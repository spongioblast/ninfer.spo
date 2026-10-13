#include "models/gemma4/load.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "core/device.h"
#include "ninfer/ops/weight_input.h"

#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

const void* device_data(const artifact::MaterializedArtifact& artifact,
                        const artifact::ParameterReference& reference) {
    const auto region = contiguous_weight_region(artifact::bind_view(reference, artifact));
    if (region.parent == nullptr || region.parent->data == nullptr ||
        region.parent->geometry.format != QType::BF16) {
        throw std::invalid_argument("gemma4 load: " + reference.name + " is not resident BF16");
    }
    return region.parent->data + region.begin * 2;
}

struct InputUse {
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::optional<float> divisor;
};

InputUse read_use(artifact::Binder& binder, const std::string& parameter,
                  const std::string& input) {
    const auto& use = binder.use(parameter, input);
    if (!use.activation_policy) {
        throw std::invalid_argument("gemma4 load: " + parameter + " has no activation policy");
    }
    InputUse out;
    switch (*use.activation_policy) {
    case artifact::ActivationPolicy::A16Only:
        out.policy = ops::LinearPolicy::A16Only;
        break;
    case artifact::ActivationPolicy::AllowA8:
        out.policy = ops::LinearPolicy::AllowA8;
        break;
    case artifact::ActivationPolicy::AllowA4:
        out.policy = ops::LinearPolicy::AllowA4;
        break;
    }
    for (const auto& [role, binding] : use.auxiliaries) {
        if (role != "activation_input_divisor") {
            throw std::invalid_argument("gemma4 load: " + parameter + " has auxiliary " + role);
        }
        out.divisor = binder.values(binding, QType::FP32).scalar_f32();
    }
    return out;
}

// One native Weight over consecutive row blocks of a single stored matrix, with the common
// activation policy of their uses. Quantized formats need the blocks to fill their parent.
ops::SingleProjectionWeight projection(const artifact::MaterializedArtifact& artifact,
                                       std::span<const artifact::ParameterReference> blocks,
                                       std::span<const InputUse> uses) {
    std::vector<WeightView> views;
    views.reserve(blocks.size());
    for (const auto& block : blocks) { views.push_back(artifact::bind_view(block, artifact)); }
    std::vector<ops::WeightInput> inputs;
    inputs.reserve(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        inputs.push_back({views[i], uses[i].policy, uses[i].divisor});
    }
    auto result = ops::prepare_linear_weight(std::span<const ops::WeightInput>(inputs));
    const QType format = result.weight.qtype;
    if (format != QType::BF16 && format != QType::NVFP4 && format != QType::FP8_E4M3FN_ROW_BF16) {
        throw std::invalid_argument("gemma4 load: " + blocks.front().name +
                                    " has an unsupported format");
    }
    return result;
}

Weight single_weight(const artifact::MaterializedArtifact& artifact,
                     const artifact::ParameterReference& reference) {
    const InputUse use;
    return projection(artifact, std::span(&reference, 1), std::span(&use, 1)).weight;
}

int config_int(const artifact::Json& config, const char* key) {
    return config.at(key).get<int>();
}

} // namespace

LoadedText::LoadedText() = default;
LoadedText::~LoadedText() = default;
LoadedText::LoadedText(LoadedText&&) noexcept = default;
LoadedText& LoadedText::operator=(LoadedText&&) noexcept = default;

LoadedText load_text(const std::string& artifact_path, DeviceContext& device) {
    artifact::Reader reader(artifact_path);
    const auto& config = reader.directory().component("text").config;
    const int layers = config_int(config, "num_hidden_layers");
    const int hidden = config_int(config, "hidden_size");
    const int vocab = config_int(config, "vocab_size");
    const int intermediate = config_int(config, "intermediate_size");
    const int heads = config_int(config, "num_attention_heads");
    const int sliding_dim = config_int(config, "head_dim");
    const int global_dim = config_int(config, "global_head_dim");
    const int sliding_kv = config_int(config, "num_key_value_heads");
    const int global_kv = config_int(config, "num_global_key_value_heads");
    const auto& kinds = config.at("layer_types");
    if (static_cast<int>(kinds.size()) != layers) {
        throw std::invalid_argument("gemma4 load: layer_types length");
    }

    artifact::Binder binder(reader);
    std::vector<artifact::ParameterReference> embedding;
    std::vector<artifact::ParameterReference> final_norm;
    std::vector<artifact::ParameterReference> head;
    std::vector<std::vector<artifact::ParameterReference>> matrices(static_cast<std::size_t>(layers));
    // Per layer, in order: query, key, (value), output, gate, up, down.
    std::vector<std::vector<InputUse>> uses(static_cast<std::size_t>(layers));
    std::vector<float> scalars(static_cast<std::size_t>(layers));
    embedding.push_back(binder.parameter("text/token_embedding", {static_cast<std::uint64_t>(vocab),
                                                                  static_cast<std::uint64_t>(hidden)}));
    final_norm.push_back(
        binder.parameter("text/final_norm", {static_cast<std::uint64_t>(hidden)}));
    head.push_back(binder.parameter("text/output_head", {static_cast<std::uint64_t>(vocab),
                                                         static_cast<std::uint64_t>(hidden)}));
    if (read_use(binder, "text/output_head", "text/final_hidden").policy !=
        ops::LinearPolicy::A16Only) {
        throw std::invalid_argument("gemma4 load: the output head runs A16Only");
    }
    for (int layer = 0; layer < layers; ++layer) {
        const bool global = kinds[layer] == "full_attention";
        const int dim = global ? global_dim : sliding_dim;
        const int kv = global ? global_kv : sliding_kv;
        const auto prefix = "text/layers/" + std::to_string(layer) + "/";
        auto& refs = matrices[static_cast<std::size_t>(layer)];
        const auto h = static_cast<std::uint64_t>(hidden);
        const auto q = static_cast<std::uint64_t>(heads * dim);
        const auto k = static_cast<std::uint64_t>(kv * dim);
        const auto inter = static_cast<std::uint64_t>(intermediate);
        auto& layer_uses = uses[static_cast<std::size_t>(layer)];
        auto matrix = [&](const char* name, artifact::Shape shape, const char* input) {
            refs.push_back(binder.parameter(prefix + name, std::move(shape)));
            layer_uses.push_back(read_use(binder, prefix + name, prefix + input));
        };
        matrix("attention/query", {q, h}, "mixer_input");
        matrix("attention/key", {k, h}, "mixer_input");
        if (!global) { matrix("attention/value", {k, h}, "mixer_input"); }
        refs.push_back(binder.parameter(prefix + "attention/query_norm", {static_cast<std::uint64_t>(dim)}));
        refs.push_back(binder.parameter(prefix + "attention/key_norm", {static_cast<std::uint64_t>(dim)}));
        matrix("attention/output", {h, q}, "attention_output");
        matrix("mlp/gate", {inter, h}, "ffn_input");
        matrix("mlp/up", {inter, h}, "ffn_input");
        matrix("mlp/down", {h, inter}, "mlp/product");
        for (const char* name : {"input_norm", "post_attention_norm", "pre_feedforward_norm",
                                 "post_feedforward_norm"}) {
            refs.push_back(binder.parameter(prefix + name, {h}));
        }
        auto scalar = binder.parameter(prefix + "layer_scalar", {1}, artifact::Residency::Host,
                                       QType::FP32);
        scalars[static_cast<std::size_t>(layer)] = binder.values(scalar.binding, QType::FP32).scalar_f32();
    }
    auto plan = std::move(binder).finish();
    LoadedText loaded;
    loaded.artifact = std::make_unique<artifact::MaterializedArtifact>(
        artifact::materialize(reader, std::move(plan), device));
    loaded.vocab = vocab;
    loaded.hidden = hidden;
    loaded.softcap = config.at("final_logit_softcapping").get<float>();
    loaded.rms_eps = config.at("rms_norm_eps").get<float>();
    loaded.token_embedding = single_weight(*loaded.artifact, embedding[0]);
    loaded.final_norm = device_data(*loaded.artifact, final_norm[0]);
    loaded.output_head = single_weight(*loaded.artifact, head[0]);
    loaded.layers.resize(static_cast<std::size_t>(layers));
    for (int layer = 0; layer < layers; ++layer) {
        const bool global = kinds[layer] == "full_attention";
        const int dim = global ? global_dim : sliding_dim;
        const int kv = global ? global_kv : sliding_kv;
        const std::span<const artifact::ParameterReference> refs =
            matrices[static_cast<std::size_t>(layer)];
        const std::span<const InputUse> layer_uses = uses[static_cast<std::size_t>(layer)];
        const std::size_t qkv_count = global ? 2 : 3;
        std::size_t cursor = qkv_count;
        std::size_t use_cursor = qkv_count;
        auto next = [&]() { return device_data(*loaded.artifact, refs[cursor++]); };
        auto take = [&](std::size_t count) {
            auto result = projection(*loaded.artifact, refs.subspan(cursor, count),
                                     layer_uses.subspan(use_cursor, count));
            cursor += count;
            use_cursor += count;
            return result;
        };
        Gemma4LayerWeights weights;
        const auto qkv = projection(*loaded.artifact, refs.first(qkv_count),
                                    layer_uses.first(qkv_count));
        weights.qkv = qkv.weight;
        weights.qkv_policy = qkv.policy;
        weights.q_norm = next();
        weights.k_norm = next();
        const auto o_proj = take(1);
        const auto gate_up = take(2);
        const auto down = take(1);
        weights.o_proj = o_proj.weight;
        weights.o_policy = o_proj.policy;
        weights.gate_up = gate_up.weight;
        weights.gate_up_policy = gate_up.policy;
        weights.down = down.weight;
        weights.down_policy = down.policy;
        weights.input_norm = next();
        weights.post_attn_norm = next();
        weights.pre_ff_norm = next();
        weights.post_ff_norm = next();
        weights.layer_scalar = scalars[static_cast<std::size_t>(layer)];
        weights.hidden = hidden;
        weights.intermediate = intermediate;
        weights.head_dim = dim;
        weights.q_heads = heads;
        weights.kv_heads = kv;
        weights.window = global ? 0 : config_int(config, "sliding_window");
        weights.rotary_dim = global ? config_int(config, "prope_rotary_dim") : dim;
        weights.rope_theta = global ? config.at("prope_theta_eff").get<float>()
                                    : config.at("rope_sliding_theta").get<float>();
        weights.rms_eps = loaded.rms_eps;
        weights.global_layer = global;
        loaded.layers[static_cast<std::size_t>(layer)] = weights;
    }
    return loaded;
}

} // namespace ninfer::models::gemma4
