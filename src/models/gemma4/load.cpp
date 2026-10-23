#include "models/gemma4/load.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "core/device.h"
#include "ninfer/ops/fp8_row_quantize.h"
#include "ninfer/ops/linear.h"
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
    if (format != QType::BF16 && format != QType::NVFP4 && format != QType::FP8_E4M3FN_ROW_BF16 &&
        format != QType::Q4_G32_FP16) {
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

bool same_object(const artifact::Binding& a, const artifact::Binding& b) {
    return a.parts.size() == 1 && b.parts.size() == 1 && a.parts[0].object == b.parts[0].object &&
           a.parts[0].begin == b.parts[0].begin && a.parts[0].end == b.parts[0].end;
}

int config_int(const artifact::Json& config, const char* key) {
    return config.at(key).get<int>();
}

// The MTP assistant's references, bound in the same plan as the target.
struct DraftRefs {
    artifact::Json config;
    // input_projection, output_projection, output_head, final_norm.
    std::vector<artifact::ParameterReference> outer;
    std::vector<InputUse> outer_uses;
    // Per layer: query, output, gate, up, down, then query_norm and the four norms.
    std::vector<std::vector<artifact::ParameterReference>> layers;
    std::vector<std::vector<InputUse>> uses;
    std::vector<float> scalars;
};

DraftRefs bind_draft(artifact::Binder& binder, const artifact::Json& config, int target_layers) {
    DraftRefs refs;
    refs.config = config;
    const auto u = [](int value) { return static_cast<std::uint64_t>(value); };
    const int layers = config_int(config, "num_hidden_layers");
    const int hidden = config_int(config, "hidden_size");
    const int backbone = config_int(config, "backbone_hidden_size");
    const int heads = config_int(config, "num_attention_heads");
    const int inter = config_int(config, "intermediate_size");
    const auto& kinds = config.at("layer_types");
    const auto& sources = config.at("kv_source_layers");
    if (static_cast<int>(kinds.size()) != layers || static_cast<int>(sources.size()) != layers) {
        throw std::invalid_argument("gemma4 load: mtp layer_types or kv_source_layers length");
    }
    auto outer = [&](const char* name, artifact::Shape shape, const char* input) {
        refs.outer.push_back(binder.parameter(name, std::move(shape)));
        if (input != nullptr) { refs.outer_uses.push_back(read_use(binder, name, input)); }
    };
    outer("mtp/input_projection", {u(hidden), u(2 * backbone)}, "mtp/stem_input");
    outer("mtp/output_projection", {u(backbone), u(hidden)}, "mtp/final_hidden");
    outer("mtp/output_head", {u(config_int(config, "vocab_size")), u(hidden)}, "mtp/final_hidden");
    outer("mtp/final_norm", {u(hidden)}, nullptr);
    for (const InputUse& use : refs.outer_uses) {
        if (use.policy != ops::LinearPolicy::A16Only || use.divisor) {
            throw std::invalid_argument("gemma4 load: the mtp projections run A16Only");
        }
    }
    refs.layers.resize(static_cast<std::size_t>(layers));
    refs.uses.resize(static_cast<std::size_t>(layers));
    for (int layer = 0; layer < layers; ++layer) {
        const int source = sources[layer].get<int>();
        if (source < 0 || source >= target_layers) {
            throw std::invalid_argument("gemma4 load: mtp KV source is outside the target");
        }
        const bool global = kinds[layer] == "full_attention";
        const int dim = config_int(config, global ? "global_head_dim" : "head_dim");
        const auto prefix = "mtp/layers/" + std::to_string(layer) + "/";
        auto& layer_refs = refs.layers[static_cast<std::size_t>(layer)];
        auto& layer_uses = refs.uses[static_cast<std::size_t>(layer)];
        auto matrix = [&](const char* name, artifact::Shape shape, const char* input) {
            layer_refs.push_back(binder.parameter(prefix + name, std::move(shape)));
            layer_uses.push_back(read_use(binder, prefix + name, prefix + input));
        };
        matrix("attention/query", {u(heads * dim), u(hidden)}, "mixer_input");
        matrix("attention/output", {u(hidden), u(heads * dim)}, "attention_output");
        matrix("mlp/gate", {u(inter), u(hidden)}, "ffn_input");
        matrix("mlp/up", {u(inter), u(hidden)}, "ffn_input");
        matrix("mlp/down", {u(hidden), u(inter)}, "mlp/product");
        layer_refs.push_back(binder.parameter(prefix + "attention/query_norm", {u(dim)}));
        for (const char* name : {"input_norm", "post_attention_norm", "pre_feedforward_norm",
                                 "post_feedforward_norm"}) {
            layer_refs.push_back(binder.parameter(prefix + name, {u(hidden)}));
        }
        auto scalar = binder.parameter(prefix + "layer_scalar", {1}, artifact::Residency::Host,
                                       QType::FP32);
        refs.scalars.push_back(binder.values(scalar.binding, QType::FP32).scalar_f32());
    }
    return refs;
}

std::unique_ptr<Gemma4DraftWeights> finish_draft(const artifact::MaterializedArtifact& artifact,
                                                 const DraftRefs& refs, cudaStream_t stream) {
    const auto& config = refs.config;
    auto draft = std::make_unique<Gemma4DraftWeights>();
    draft->hidden = config_int(config, "hidden_size");
    draft->backbone = config_int(config, "backbone_hidden_size");
    draft->vocab = config_int(config, "vocab_size");
    draft->rms_eps = config.at("rms_norm_eps").get<float>();
    draft->input_projection = single_weight(artifact, refs.outer[0]);
    draft->output_projection = single_weight(artifact, refs.outer[1]);
    draft->output_head = single_weight(artifact, refs.outer[2]);
    // The MTP head is 262144x1024. The registered FP8 GEMV is that shape; a BF16 artifact is
    // quantized here so a reconvert is not required.
    if (draft->output_head.qtype == QType::BF16 && draft->output_head.n == 262144 &&
        draft->output_head.k == 1024) {
        ops::quantize_fp8_row_bf16(draft->output_head, draft->quantized_head, stream);
    }
    // The assistant's projections are small BF16 GEMMs. Quantizing them onto the FP8 GEMV
    // cuts the draft step's weight traffic. The head above is already handled.
    auto quantize = [&](Weight& weight) {
        if (weight.qtype != QType::BF16) { return; }
        draft->quantized_weights.emplace_back();
        ops::quantize_fp8_row_bf16(weight, draft->quantized_weights.back(), stream);
    };
    quantize(draft->input_projection);
    quantize(draft->output_projection);
    draft->final_norm = device_data(artifact, refs.outer[3]);
    const auto& kinds = config.at("layer_types");
    for (std::size_t layer = 0; layer < refs.layers.size(); ++layer) {
        const std::span<const artifact::ParameterReference> layer_refs = refs.layers[layer];
        const std::span<const InputUse> layer_uses = refs.uses[layer];
        auto take = [&](std::size_t first, std::size_t count) {
            return projection(artifact, layer_refs.subspan(first, count),
                              layer_uses.subspan(first, count));
        };
        const bool global = kinds[layer] == "full_attention";
        Gemma4LayerWeights weights;
        const auto query = take(0, 1);
        const auto output = take(1, 1);
        const auto gate_up = take(2, 2);
        const auto down = take(4, 1);
        weights.qkv = query.weight;
        weights.qkv_policy = query.policy;
        weights.o_proj = output.weight;
        weights.o_policy = output.policy;
        weights.gate_up = gate_up.weight;
        weights.gate_up_policy = gate_up.policy;
        weights.down = down.weight;
        weights.down_policy = down.policy;
        weights.q_norm = device_data(artifact, layer_refs[5]);
        weights.input_norm = device_data(artifact, layer_refs[6]);
        weights.post_attn_norm = device_data(artifact, layer_refs[7]);
        weights.pre_ff_norm = device_data(artifact, layer_refs[8]);
        weights.post_ff_norm = device_data(artifact, layer_refs[9]);
        weights.layer_scalar = refs.scalars[layer];
        weights.hidden = draft->hidden;
        weights.intermediate = config_int(config, "intermediate_size");
        weights.head_dim = config_int(config, global ? "global_head_dim" : "head_dim");
        weights.q_heads = config_int(config, "num_attention_heads");
        weights.kv_heads =
            config_int(config, global ? "num_global_key_value_heads" : "num_key_value_heads");
        weights.window = global ? 0 : config_int(config, "sliding_window");
        weights.rotary_dim = global ? config_int(config, "prope_rotary_dim") : weights.head_dim;
        weights.rope_theta = global ? config.at("prope_theta_eff").get<float>()
                                    : config.at("rope_sliding_theta").get<float>();
        weights.rms_eps = draft->rms_eps;
        weights.global_layer = global;
        quantize(weights.qkv);
        quantize(weights.o_proj);
        quantize(weights.gate_up);
        quantize(weights.down);
        draft->layers.push_back(weights);
        draft->kv_source.push_back(config.at("kv_source_layers")[layer].get<int>());
    }
    return draft;
}

struct VisionRefs {
    artifact::Json config;
    artifact::ParameterReference patch;
    artifact::ParameterReference table;
    artifact::ParameterReference std_bias;
    artifact::ParameterReference std_scale;
    artifact::ParameterReference projection;
    InputUse patch_use;
    InputUse projection_use;
    std::vector<std::vector<artifact::ParameterReference>> layers;
    std::vector<std::vector<InputUse>> uses;
    std::vector<float> scalars;
};

VisionRefs bind_vision(artifact::Binder& binder, const artifact::Json& config) {
    VisionRefs refs;
    refs.config = config;
    const auto u = [](int value) { return static_cast<std::uint64_t>(value); };
    const int hidden = config_int(config, "hidden_size");
    const int text_hidden = config_int(config, "text_hidden_size");
    const int heads = config_int(config, "num_attention_heads");
    const int dim = config_int(config, "head_dim");
    const int inter = config_int(config, "intermediate_size");
    const int layers = config_int(config, "num_hidden_layers");
    const int pixels = config_int(config, "patch_pixels");
    const int positions = config_int(config, "position_embedding_size");
    const auto q = u(heads * dim);
    refs.patch = binder.parameter("vision/patch_projection", {u(hidden), u(pixels)});
    refs.patch_use = read_use(binder, "vision/patch_projection", "vision/patch_input");
    refs.table = binder.parameter("vision/position_table", {u(2), u(positions), u(hidden)});
    refs.std_bias = binder.parameter("vision/std_bias", {u(hidden)});
    refs.std_scale = binder.parameter("vision/std_scale", {u(hidden)});
    refs.projection =
        binder.parameter("vision/embedding_projection", {u(text_hidden), u(hidden)});
    refs.projection_use = read_use(binder, "vision/embedding_projection", "vision/pooled_norm");
    refs.layers.resize(static_cast<std::size_t>(layers));
    refs.uses.resize(static_cast<std::size_t>(layers));
    for (int layer = 0; layer < layers; ++layer) {
        const auto prefix = "vision/layers/" + std::to_string(layer) + "/";
        auto& layer_refs = refs.layers[static_cast<std::size_t>(layer)];
        auto& layer_uses = refs.uses[static_cast<std::size_t>(layer)];
        auto matrix = [&](const char* name, artifact::Shape shape, const char* input) {
            layer_refs.push_back(binder.parameter(prefix + name, std::move(shape)));
            layer_uses.push_back(read_use(binder, prefix + name, prefix + input));
        };
        matrix("attention/query", {q, u(hidden)}, "mixer_input");
        matrix("attention/key", {q, u(hidden)}, "mixer_input");
        matrix("attention/value", {q, u(hidden)}, "mixer_input");
        layer_refs.push_back(binder.parameter(prefix + "attention/query_norm", {u(dim)}));
        layer_refs.push_back(binder.parameter(prefix + "attention/key_norm", {u(dim)}));
        matrix("attention/output", {u(hidden), q}, "attention_output");
        matrix("mlp/gate", {u(inter), u(hidden)}, "ffn_input");
        matrix("mlp/up", {u(inter), u(hidden)}, "ffn_input");
        matrix("mlp/down", {u(hidden), u(inter)}, "mlp/product");
        for (const char* name : {"input_norm", "post_attention_norm", "pre_feedforward_norm",
                                 "post_feedforward_norm"}) {
            layer_refs.push_back(binder.parameter(prefix + name, {u(hidden)}));
        }
        auto scalar = binder.parameter(prefix + "layer_scalar", {1}, artifact::Residency::Host,
                                       QType::FP32);
        refs.scalars.push_back(binder.values(scalar.binding, QType::FP32).scalar_f32());
    }
    return refs;
}

Gemma4VisionWeights finish_vision(const artifact::MaterializedArtifact& artifact,
                                  const VisionRefs& refs) {
    const auto& config = refs.config;
    Gemma4VisionWeights out;
    out.hidden = config_int(config, "hidden_size");
    out.text_hidden = config_int(config, "text_hidden_size");
    out.intermediate = config_int(config, "intermediate_size");
    out.heads = config_int(config, "num_attention_heads");
    out.head_dim = config_int(config, "head_dim");
    out.patch_pixels = config_int(config, "patch_pixels");
    out.pooling_kernel_size = config_int(config, "pooling_kernel_size");
    out.position_table_size = config_int(config, "position_embedding_size");
    out.default_output_length = config_int(config, "default_output_length");
    out.boi_token_id = config_int(config, "boi_token_id");
    out.eoi_token_id = config_int(config, "eoi_token_id");
    out.image_token_id = config_int(config, "image_token_id");
    out.rope_theta = config.at("rope_theta").get<float>();
    out.rms_eps = config.at("rms_norm_eps").get<float>();
    const InputUse patch_uses[1] = {refs.patch_use};
    out.patch_projection = projection(artifact, std::span(&refs.patch, 1),
                                      std::span(patch_uses, 1)).weight;
    out.position_table = device_data(artifact, refs.table);
    out.std_bias = device_data(artifact, refs.std_bias);
    out.std_scale = device_data(artifact, refs.std_scale);
    const InputUse proj_uses[1] = {refs.projection_use};
    out.embedding_projection = projection(artifact, std::span(&refs.projection, 1),
                                          std::span(proj_uses, 1)).weight;
    out.layers.resize(refs.layers.size());
    for (std::size_t layer = 0; layer < refs.layers.size(); ++layer) {
        const std::span<const artifact::ParameterReference> layer_refs = refs.layers[layer];
        const std::span<const InputUse> layer_uses = refs.uses[layer];
        std::size_t cursor = 0;
        std::size_t use_cursor = 0;
        auto next = [&]() { return device_data(artifact, layer_refs[cursor++]); };
        auto take = [&](std::size_t count) {
            auto result = projection(artifact, layer_refs.subspan(cursor, count),
                                     layer_uses.subspan(use_cursor, count));
            cursor += count;
            use_cursor += count;
            return result;
        };
        Gemma4VisionLayerWeights weights;
        const auto qkv = take(3);
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
        weights.layer_scalar = refs.scalars[layer];
        out.layers[layer] = weights;
    }
    return out;
}

} // namespace

LoadedText::LoadedText() = default;
LoadedText::~LoadedText() = default;
LoadedText::LoadedText(LoadedText&&) noexcept = default;
LoadedText& LoadedText::operator=(LoadedText&&) noexcept = default;

LoadedText load_text(const std::string& artifact_path, DeviceContext& device, bool with_draft) {
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
    std::optional<DraftRefs> draft;
    if (with_draft) {
        const auto& components = reader.directory().components;
        const auto found = components.find("mtp");
        if (found == components.end()) {
            throw std::invalid_argument("gemma4 --spec mtp needs an artifact with the mtp component");
        }
        draft.emplace(bind_draft(binder, found->second.config, layers));
    }
    std::optional<VisionRefs> vision;
    if (reader.directory().components.find("vision") != reader.directory().components.end()) {
        vision = bind_vision(binder, reader.directory().component("vision").config);
    }
    auto plan = std::move(binder).finish();
    LoadedText loaded;
    loaded.artifact = std::make_unique<artifact::MaterializedArtifact>(
        artifact::materialize(reader, std::move(plan), device));
    loaded.vocab = vocab;
    loaded.hidden = hidden;
    loaded.softcap = config.at("final_logit_softcapping").get<float>();
    loaded.rms_eps = config.at("rms_norm_eps").get<float>();
    loaded.token_embedding.table = single_weight(*loaded.artifact, embedding[0]);
    loaded.final_norm = device_data(*loaded.artifact, final_norm[0]);
    loaded.output_head = single_weight(*loaded.artifact, head[0]);
    // A table the head shares is the unscaled one, so the gather applies the normalizer.
    if (same_object(embedding[0].binding, head[0].binding)) {
        loaded.token_embedding.scale = config.at("embed_normalizer").get<float>();
    }
    loaded.layers.resize(static_cast<std::size_t>(layers));
    loaded.a8_rows.reserve(static_cast<std::size_t>(layers) * 4);
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
        for (Weight* projection : {&weights.qkv, &weights.o_proj, &weights.gate_up, &weights.down}) {
            ops::linear_prepare_a8_rows(*projection, loaded.a8_rows.emplace_back(), device.stream);
        }
        loaded.layers[static_cast<std::size_t>(layer)] = weights;
    }
    if (draft) {
        loaded.draft = finish_draft(*loaded.artifact, *draft, device.stream);
        if (loaded.draft->backbone != hidden || loaded.draft->vocab != vocab) {
            throw std::invalid_argument("gemma4 load: mtp does not match the target");
        }
        for (std::size_t index = 0; index < loaded.draft->layers.size(); ++index) {
            const Gemma4LayerWeights& own = loaded.draft->layers[index];
            const Gemma4LayerWeights& source =
                loaded.layers[static_cast<std::size_t>(loaded.draft->kv_source[index])];
            if (own.global_layer != source.global_layer || own.head_dim != source.head_dim ||
                own.kv_heads != source.kv_heads || own.window != source.window) {
                throw std::invalid_argument("gemma4 load: mtp layer and its KV source disagree");
            }
        }
    }
    if (vision) { loaded.vision = finish_vision(*loaded.artifact, *vision); }
    return loaded;
}

} // namespace ninfer::models::gemma4
