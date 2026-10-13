#include "models/gemma4/load.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "core/device.h"

#include <stdexcept>
#include <string>

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
    std::vector<float> scalars(static_cast<std::size_t>(layers));
    embedding.push_back(binder.parameter("text/token_embedding", {static_cast<std::uint64_t>(vocab),
                                                                  static_cast<std::uint64_t>(hidden)}));
    final_norm.push_back(
        binder.parameter("text/final_norm", {static_cast<std::uint64_t>(hidden)}));
    head.push_back(binder.parameter("text/output_head", {static_cast<std::uint64_t>(vocab),
                                                         static_cast<std::uint64_t>(hidden)}));
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
        refs.push_back(binder.parameter(prefix + "attention/query", {q, h}));
        refs.push_back(binder.parameter(prefix + "attention/key", {k, h}));
        if (!global) { refs.push_back(binder.parameter(prefix + "attention/value", {k, h})); }
        refs.push_back(binder.parameter(prefix + "attention/query_norm", {static_cast<std::uint64_t>(dim)}));
        refs.push_back(binder.parameter(prefix + "attention/key_norm", {static_cast<std::uint64_t>(dim)}));
        refs.push_back(binder.parameter(prefix + "attention/output", {h, q}));
        refs.push_back(binder.parameter(prefix + "mlp/gate", {inter, h}));
        refs.push_back(binder.parameter(prefix + "mlp/up", {inter, h}));
        refs.push_back(binder.parameter(prefix + "mlp/down", {h, inter}));
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
    loaded.token_embedding = device_data(*loaded.artifact, embedding[0]);
    loaded.final_norm = device_data(*loaded.artifact, final_norm[0]);
    loaded.output_head = device_data(*loaded.artifact, head[0]);
    loaded.layers.resize(static_cast<std::size_t>(layers));
    for (int layer = 0; layer < layers; ++layer) {
        const bool global = kinds[layer] == "full_attention";
        const int dim = global ? global_dim : sliding_dim;
        const int kv = global ? global_kv : sliding_kv;
        auto& refs = matrices[static_cast<std::size_t>(layer)];
        std::size_t cursor = 0;
        auto next = [&]() { return device_data(*loaded.artifact, refs[cursor++]); };
        Gemma4LayerWeights weights;
        weights.q_proj = next();
        weights.k_proj = next();
        weights.v_proj = global ? nullptr : next();
        weights.q_norm = next();
        weights.k_norm = next();
        weights.o_proj = next();
        weights.gate_proj = next();
        weights.up_proj = next();
        weights.down_proj = next();
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
