#include "models/gemma4/execution/draft_forward.h"

#include "models/gemma4/execution/projection.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/topk_columns.h"
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
                 const Gemma4CacheLayer& cache, int width, ops::Bf16GemmContext& gemm,
                 WorkspaceArena& workspace, cudaStream_t stream) {
    const int hidden = layer.hidden;
    const int dim    = layer.head_dim;
    const int heads  = layer.q_heads;
    const int inter  = layer.intermediate;
    if (layer.qkv.n != heads * dim || layer.qkv.k != hidden) {
        throw std::invalid_argument("gemma4 draft: query must be [heads*dim,hidden]");
    }
    auto scope     = workspace.scope();
    auto q         = workspace.alloc(DType::BF16, {dim, heads, width});
    auto qn        = workspace.alloc(DType::BF16, {dim, heads, width});
    auto attn      = workspace.alloc(DType::BF16, {dim, heads, width});
    auto projected = workspace.alloc(DType::BF16, {hidden, width});
    Tensor q_column(q.data, DType::BF16, {heads * dim, width});
    project(hn, layer.qkv, layer.qkv_policy, q_column, gemm, workspace, stream);
    ops::rmsnorm(q, norm_weight(layer.q_norm, dim), layer.rms_eps, false, qn, stream);
    ops::rope(position, layer.rotary_dim, layer.rope_theta, qn, stream);
    ops::gemma4_attention(qn, *cache.k_pages, *cache.v_pages, *cache.block_table, position,
                          layer.window, 1.0f, attn, workspace, stream);
    Tensor attn_column(attn.data, DType::BF16, {heads * dim, width});
    project(attn_column, layer.o_proj, layer.o_policy, projected, gemm, workspace, stream);
    const Tensor pre_ff_norm = norm_weight(layer.pre_ff_norm, hidden);
    ops::gemma4_residual_norm(projected, norm_weight(layer.post_attn_norm, hidden), 1.0f,
                              &pre_ff_norm, layer.rms_eps, x, &hn, stream);
    auto gate_up = workspace.alloc(DType::BF16, {2 * inter, width});
    project(hn, layer.gate_up, layer.gate_up_policy, gate_up, gemm, workspace, stream);
    Tensor gate(gate_up.data, DType::BF16, {inter, width});
    Tensor up(static_cast<std::byte*>(gate_up.data) + static_cast<std::size_t>(inter) * 2,
              DType::BF16, {inter, width});
    if (width > 1) {
        // gate_up is [2*inter, width], so each column holds gate then up. The two views skip
        // the other half of every column.
        const std::int64_t column = static_cast<std::int64_t>(2 * inter) * 2;
        gate.nb[1]                = column;
        up.nb[1]                  = column;
    }
    auto activated = workspace.alloc(DType::BF16, {inter, width});
    ops::geglu(gate, up, activated, stream);
    project(activated, layer.down, layer.down_policy, projected, gemm, workspace, stream);
    const Tensor next_norm = norm_weight(next_input_norm, hidden);
    ops::gemma4_residual_norm(projected, norm_weight(layer.post_ff_norm, hidden),
                              layer.layer_scalar,
                              next_input_norm != nullptr ? &next_norm : nullptr, layer.rms_eps,
                              x, next_input_norm != nullptr ? &hn : nullptr, stream);
}

} // namespace

std::size_t gemma4_draft_workspace_bytes(const Gemma4DraftWeights& draft, int columns) {
    const auto width = static_cast<std::size_t>(std::max(columns, 1));
    std::size_t widest = 0;
    for (const Gemma4LayerWeights& layer : draft.layers) {
        const auto q     = static_cast<std::size_t>(layer.q_heads) * layer.head_dim;
        const auto inter = static_cast<std::size_t>(layer.intermediate);
        const std::size_t scratch = std::max(
            {projection_bytes(layer.qkv, layer.qkv_policy, columns),
             projection_bytes(layer.o_proj, layer.o_policy, columns),
             projection_bytes(layer.gate_up, layer.gate_up_policy, columns),
             projection_bytes(layer.down, layer.down_policy, columns),
             ops::gemma4_attention_workspace_capacity_bytes(layer.head_dim, layer.q_heads,
                                                            layer.kv_heads, columns)});
        widest = std::max(widest, (3 * q + 3 * inter + static_cast<std::size_t>(layer.hidden)) * 2 *
                                          width +
                                      scratch);
    }
    const std::size_t outer =
        (2U * draft.backbone + 3U * draft.hidden + static_cast<std::size_t>(draft.vocab)) * 2U *
            width +
        std::max({projection_bytes(draft.input_projection, ops::LinearPolicy::A16Only, columns),
                  projection_bytes(draft.output_projection, ops::LinearPolicy::A16Only, columns),
                  projection_bytes(draft.output_head, ops::LinearPolicy::A16Only, columns)});
    return outer + widest + kAllocations * kAlignment;
}

void gemma4_draft_steps(const Tensor& tokens, const Tensor& positions, const Tensor& carry_in,
                        Tensor& carry_out, Tensor& top_ids, int k, const Weight& token_embedding,
                        const Gemma4DraftWeights& draft,
                        std::span<const Gemma4CacheLayer> target_caches,
                        ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                        cudaStream_t stream) {
    const int backbone = draft.backbone;
    const int hidden   = draft.hidden;
    const int width    = tokens.ne[0];
    if (width < 1 || tokens.ne[0] != positions.ne[0] || carry_in.ne[0] != backbone ||
        carry_in.ne[1] != width || carry_out.ne[0] != backbone || carry_out.ne[1] != width ||
        k < 1 || k > 8 || top_ids.ne[0] != k || top_ids.ne[1] != width || draft.layers.empty() ||
        draft.kv_source.size() != draft.layers.size()) {
        throw std::invalid_argument("gemma4 draft: batched step shape");
    }
    auto scope = workspace.scope();
    auto stem  = workspace.alloc(DType::BF16, {2 * backbone, width});
    auto embedded = workspace.alloc(DType::BF16, {backbone, width});
    ops::embedding(tokens, token_embedding, embedded, stream);
    const std::size_t half = static_cast<std::size_t>(backbone) * 2;
    check(cudaMemcpy2DAsync(stem.data, 2 * half, embedded.data, half, half,
                            static_cast<std::size_t>(width), cudaMemcpyDeviceToDevice, stream),
          "stem embed");
    check(cudaMemcpy2DAsync(static_cast<std::byte*>(stem.data) + half, 2 * half, carry_in.data, half,
                            half, static_cast<std::size_t>(width), cudaMemcpyDeviceToDevice, stream),
          "stem carry");
    auto x  = workspace.alloc(DType::BF16, {hidden, width});
    auto hn = workspace.alloc(DType::BF16, {hidden, width});
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
        draft_layer(x, hn, draft.layers[index], following, positions, target_caches[source], width,
                    gemm, workspace, stream);
    }
    ops::rmsnorm(x, norm_weight(draft.final_norm, hidden), draft.rms_eps, false, hn, stream);
    auto logits = workspace.alloc(DType::BF16, {draft.vocab, width});
    project(hn, draft.output_head, ops::LinearPolicy::A16Only, logits, gemm, workspace, stream);
    if (k == 1) {
        Tensor flat(top_ids.data, DType::I32, {width});
        ops::argmax(logits, flat, draft.vocab, stream);
    } else {
        ops::topk_columns(logits, draft.vocab, k, top_ids, stream);
    }
    project(hn, draft.output_projection, ops::LinearPolicy::A16Only, carry_out, gemm, workspace,
            stream);
}

void gemma4_draft_step(const Tensor& token, const Tensor& position, Tensor& carry, Tensor& next,
                       const Weight& token_embedding, const Gemma4DraftWeights& draft,
                       std::span<const Gemma4CacheLayer> target_caches,
                       ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                       cudaStream_t stream) {
    Tensor top(next.data, DType::I32, {1, 1});
    gemma4_draft_steps(token, position, carry, carry, top, 1, token_embedding, draft,
                       target_caches, gemm, workspace, stream);
}

} // namespace ninfer::models::gemma4
