#include "models/gemma4/execution/draft_forward.h"

#include "models/gemma4/execution/projection.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gemma4_attention.h"
#include "ninfer/ops/gemma4_residual_norm.h"
#include "ninfer/ops/geglu.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::models::gemma4 {
namespace {

constexpr std::size_t kAllocations = 16;
constexpr std::size_t kAlignment   = 256;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gemma4 draft: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

// hn holds rmsnorm(x, input_norm) on entry; with next_input_norm it leaves the next layer's
// input norm there.
void draft_layer(Tensor& x, Tensor& hn, const Gemma4LayerWeights& layer,
                 const void* next_input_norm, const Tensor& position,
                 const Gemma4CacheLayer& cache, ops::Bf16GemmContext& gemm,
                 WorkspaceArena& workspace, cudaStream_t stream) {
    const int hidden = layer.hidden;
    const int dim    = layer.head_dim;
    const int heads  = layer.q_heads;
    const int inter  = layer.intermediate;
    if (layer.qkv.n != heads * dim || layer.qkv.k != hidden) {
        throw std::invalid_argument("gemma4 draft: query must be [heads*dim,hidden]");
    }
    auto scope     = workspace.scope();
    auto q         = workspace.alloc(DType::BF16, {dim, heads});
    auto qn        = workspace.alloc(DType::BF16, {dim, heads});
    auto attn      = workspace.alloc(DType::BF16, {dim, heads, 1});
    auto projected = workspace.alloc(DType::BF16, {hidden, 1});
    Tensor q_column(q.data, DType::BF16, {heads * dim, 1});
    project(hn, layer.qkv, layer.qkv_policy, q_column, gemm, workspace, stream);
    ops::rmsnorm(q, norm_weight(layer.q_norm, dim), layer.rms_eps, false, qn, stream);
    Tensor q_heads(qn.data, DType::BF16, {dim, heads, 1});
    ops::rope(position, layer.rotary_dim, layer.rope_theta, q_heads, stream);
    ops::gemma4_attention(q_heads, *cache.k_pages, *cache.v_pages, *cache.block_table, position,
                          layer.window, 1.0f, attn, workspace, stream);
    Tensor attn_column(attn.data, DType::BF16, {heads * dim, 1});
    project(attn_column, layer.o_proj, layer.o_policy, projected, gemm, workspace, stream);
    const Tensor pre_ff_norm = norm_weight(layer.pre_ff_norm, hidden);
    ops::gemma4_residual_norm(projected, norm_weight(layer.post_attn_norm, hidden), 1.0f,
                              &pre_ff_norm, layer.rms_eps, x, &hn, stream);
    auto gate_up = workspace.alloc(DType::BF16, {2 * inter, 1});
    project(hn, layer.gate_up, layer.gate_up_policy, gate_up, gemm, workspace, stream);
    const Tensor gate(gate_up.data, DType::BF16, {inter, 1});
    const Tensor up(static_cast<std::byte*>(gate_up.data) + static_cast<std::size_t>(inter) * 2,
                    DType::BF16, {inter, 1});
    auto activated = workspace.alloc(DType::BF16, {inter, 1});
    ops::geglu(gate, up, activated, stream);
    project(activated, layer.down, layer.down_policy, projected, gemm, workspace, stream);
    const Tensor next_norm = norm_weight(next_input_norm, hidden);
    ops::gemma4_residual_norm(projected, norm_weight(layer.post_ff_norm, hidden),
                              layer.layer_scalar,
                              next_input_norm != nullptr ? &next_norm : nullptr, layer.rms_eps,
                              x, next_input_norm != nullptr ? &hn : nullptr, stream);
}

} // namespace

std::size_t gemma4_draft_workspace_bytes(const Gemma4DraftWeights& draft) {
    std::size_t widest = 0;
    for (const Gemma4LayerWeights& layer : draft.layers) {
        const auto q     = static_cast<std::size_t>(layer.q_heads) * layer.head_dim;
        const auto inter = static_cast<std::size_t>(layer.intermediate);
        const std::size_t scratch = std::max(
            {projection_bytes(layer.qkv, layer.qkv_policy, 1),
             projection_bytes(layer.o_proj, layer.o_policy, 1),
             projection_bytes(layer.gate_up, layer.gate_up_policy, 1),
             projection_bytes(layer.down, layer.down_policy, 1),
             ops::gemma4_attention_workspace_capacity_bytes(layer.head_dim, layer.q_heads,
                                                            layer.kv_heads, 1)});
        widest = std::max(widest, (3 * q + 3 * inter + static_cast<std::size_t>(layer.hidden)) * 2 +
                                      scratch);
    }
    const std::size_t outer =
        (2U * draft.backbone + 3U * draft.hidden + static_cast<std::size_t>(draft.vocab)) * 2U +
        std::max({projection_bytes(draft.input_projection, ops::LinearPolicy::A16Only, 1),
                  projection_bytes(draft.output_projection, ops::LinearPolicy::A16Only, 1),
                  projection_bytes(draft.output_head, ops::LinearPolicy::A16Only, 1)});
    return outer + widest + kAllocations * kAlignment;
}

void gemma4_draft_step(const Tensor& token, const Tensor& position, Tensor& carry, Tensor& next,
                       const Weight& token_embedding, const Gemma4DraftWeights& draft,
                       std::span<const Gemma4CacheLayer> target_caches,
                       ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                       cudaStream_t stream) {
    const int backbone = draft.backbone;
    const int hidden   = draft.hidden;
    if (carry.ne[0] != backbone || carry.ne[1] != 1 || draft.layers.empty() ||
        draft.kv_source.size() != draft.layers.size()) {
        throw std::invalid_argument("gemma4 draft: carry must be [backbone,1]");
    }
    auto scope = workspace.scope();
    auto stem  = workspace.alloc(DType::BF16, {2 * backbone, 1});
    Tensor embedded(stem.data, DType::BF16, {backbone, 1});
    ops::embedding(token, token_embedding, embedded, stream);
    check(cudaMemcpyAsync(static_cast<std::byte*>(stem.data) + static_cast<std::size_t>(backbone) * 2,
                          carry.data, static_cast<std::size_t>(backbone) * 2,
                          cudaMemcpyDeviceToDevice, stream),
          "carry copy");
    auto x  = workspace.alloc(DType::BF16, {hidden, 1});
    auto hn = workspace.alloc(DType::BF16, {hidden, 1});
    project(stem, draft.input_projection, ops::LinearPolicy::A16Only, x, gemm, workspace, stream);
    ops::rmsnorm(x, norm_weight(draft.layers[0].input_norm, hidden), draft.rms_eps, false, hn,
                 stream);
    for (std::size_t index = 0; index < draft.layers.size(); ++index) {
        const void* following =
            index + 1 < draft.layers.size() ? draft.layers[index + 1].input_norm : nullptr;
        const auto source = static_cast<std::size_t>(draft.kv_source[index]);
        if (source >= target_caches.size()) {
            throw std::invalid_argument("gemma4 draft: KV source is outside the target");
        }
        draft_layer(x, hn, draft.layers[index], following, position, target_caches[source], gemm,
                    workspace, stream);
    }
    ops::rmsnorm(x, norm_weight(draft.final_norm, hidden), draft.rms_eps, false, hn, stream);
    auto logits = workspace.alloc(DType::BF16, {draft.vocab, 1});
    project(hn, draft.output_head, ops::LinearPolicy::A16Only, logits, gemm, workspace, stream);
    ops::argmax(logits, next, draft.vocab, stream);
    project(hn, draft.output_projection, ops::LinearPolicy::A16Only, carry, gemm, workspace,
            stream);
}

} // namespace ninfer::models::gemma4
