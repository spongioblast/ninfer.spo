#include "models/qwen3_8_flash_next_125b_a6b/impl/variant.h"

#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next_125b_a6b::detail {
namespace {

std::vector<GraphExecutionProfile> profiles(std::uint32_t capacity) {
    if (capacity == 0) { return {}; }
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t frontier :
         {127U, 511U, 2047U, 8191U, 16383U, 32767U, 65535U, 131071U, 196607U}) {
        if (begin >= capacity) { break; }
        const std::uint32_t end = std::min(frontier, capacity - 1);
        const std::uint32_t topology_class =
            end <= 16383U ? 0U : (end <= 32767U ? 1U : (end <= 131071U ? 2U : 3U));
        out.push_back({begin, end, topology_class});
        begin = end + 1;
    }
    if (begin < capacity) { out.push_back({begin, capacity - 1, 3U}); }
    return out;
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    return profiles(capacity);
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0) { return {}; }
    auto out = profiles(capacity);
    // The four-token MTP verify/draft schedule changes CUDA Graph topology between the 65K and
    // 131K envelopes even though ordinary single-token decode can update one executable across
    // that boundary. Keep the same execution frontiers while giving the two MTP definitions
    // distinct executable owners.
    for (GraphExecutionProfile& profile : out) {
        if (profile.max <= 65535U) { continue; }
        profile.topology_class = profile.max <= 131071U ? 3U : 4U;
    }
    return out;
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value,
                                   qwen3_8_flash_next::TextPhase, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    ops::flash_next_project_query_gate(hidden, weights.query_gate, query, gate, workspace, stream);
    ops::linear(hidden, weights.key, key, stream);
    ops::linear(hidden, weights.value, value, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_8_flash_next::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope   = workspace.scope();
    Tensor delta = workspace.alloc(DType::BF16, {weight.n, attention.ne[1]});
    ops::linear(attention, weight, delta, stream);
    ops::residual_add(delta, residual, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    attention_projection(hidden, weights, query, gate, key, value,
                         qwen3_8_flash_next::TextPhase::Verify, workspace, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.key, key, stream);
    ops::linear(hidden, weights.value, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream) {
    ops::flash_next_project_query_gate(hidden, weights.query_gate, query, gate, workspace, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_8_flash_next::TextPhase,
                                   WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.query_key_value, qkv, stream);
    ops::linear(hidden, weights.output_gate, output_gate, stream);
}

void Variant::gdn_input_projection_snapshot(const Tensor&, const GdnProjectionWeights&,
                                            const Tensor&, Tensor&, const Tensor&, const Tensor&,
                                            const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                            qwen3_8_flash_next::TextPhase, WorkspaceArena&,
                                            cudaStream_t) {
    throw std::logic_error("Flash-Next GDN snapshot route is not bound to the Flash schedule");
}

void Variant::gdn_input_projection_record(const Tensor&, const GdnProjectionWeights&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                          Tensor&, Tensor&, Tensor&, Tensor&,
                                          qwen3_8_flash_next::TextPhase, WorkspaceArena&,
                                          cudaStream_t) {
    throw std::logic_error("Flash-Next GDN record route is not bound to the Flash schedule");
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_8_flash_next::TextPhase phase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    attention_output_projection(hidden, weight, residual, phase, workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace,
                                          DeviceExecutionView execution) {
    ops::rmsnorm(residual, norm_weight, eps, true, hidden, execution.stream);
    auto scope = workspace.scope();
    Tensor a   = workspace.alloc(DType::BF16, {48, hidden.ne[1]});
    Tensor b   = workspace.alloc(DType::BF16, {48, hidden.ne[1]});
    ops::linear(hidden, weights.a_projection, a, execution.stream);
    ops::linear(hidden, weights.b_projection, b, execution.stream);
    ops::gdn_gating(a, b, weights.a_log, weights.dt_bias, g, beta, execution.stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_8_flash_next::TextPhase, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    auto scope   = workspace.scope();
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, hidden.ne[1]});
    ops::flash_next_moe(hidden, weights, delta, workspace, stream);
    ops::residual_add(delta, residual, stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    post_mixer(hidden, weights, residual, qwen3_8_flash_next::TextPhase::Verify, workspace, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    return ops::flash_next_qsa_workspace_capacity_bytes(last, static_cast<std::uint32_t>(last));
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t,
                                                                    std::int32_t last) {
    return ops::flash_next_query_gate_workspace_capacity_bytes(last);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   qwen3_8_flash_next::TextPhase,
                                                                   std::int32_t,
                                                                   std::int32_t last) {
    return ops::flash_next_query_gate_workspace_capacity_bytes(last);
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile, qwen3_8_flash_next::TextPhase, std::int32_t, std::int32_t last) {
    return static_cast<std::size_t>(TextConfig::hidden) * last * sizeof(std::uint16_t);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   qwen3_8_flash_next::TextPhase,
                                                                   std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile, qwen3_8_flash_next::TextPhase, std::int32_t, std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile, qwen3_8_flash_next::TextPhase, std::int32_t, std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(
    WeightsProfile p, qwen3_8_flash_next::TextPhase ph, std::int32_t first, std::int32_t last) {
    return attention_output_projection_workspace_capacity_bytes(p, ph, first, last);
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t,
                                                                          std::int32_t last) {
    return static_cast<std::size_t>(96) * last * sizeof(std::uint16_t);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile,
                                                         qwen3_8_flash_next::TextPhase,
                                                         std::int32_t, std::int32_t last) {
    return static_cast<std::size_t>(TextConfig::hidden) * last * sizeof(std::uint16_t) +
           ops::flash_next_moe_workspace_capacity_bytes(last);
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    return post_mixer_workspace_capacity_bytes(WeightsProfile::Nvfp4,
                                               qwen3_8_flash_next::TextPhase::Verify, first, last);
}

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b::detail
