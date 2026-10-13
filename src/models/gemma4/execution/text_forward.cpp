#include "models/gemma4/execution/text_forward.h"

#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gemma4_attention.h"
#include "ninfer/ops/gemma4_kv_append.h"
#include "ninfer/ops/geglu.h"
#include "ninfer/ops/logit_softcap.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/residual_scale.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ninfer::models::gemma4 {
namespace {

constexpr std::size_t kAllocations = 16;
constexpr std::size_t kAlignment   = 256;

Tensor norm_weight(const void* data, int width) {
    return Tensor(const_cast<void*>(data), DType::BF16, {width});
}

void project(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& out,
             ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream) {
    if (weight.qtype == QType::BF16) {
        gemm.launch(x, weight, out, stream);
    } else {
        ops::linear(x, weight, out, policy, workspace, stream);
    }
}

std::size_t projection_bytes(const Weight& weight, ops::LinearPolicy policy, int tokens) {
    if (weight.qtype == QType::BF16) { return 0; }
    return ops::linear_workspace_capacity_bytes(weight.qtype, weight.n, weight.k, policy, 1,
                                                tokens);
}

void require_matrix(const Weight& weight, int rows, int cols, const char* name) {
    if (weight.n != rows || weight.k != cols) {
        throw std::invalid_argument(std::string("gemma4_layer: ") + name + " must be [" +
                                    std::to_string(rows) + "," + std::to_string(cols) + "]");
    }
}

// Rows [first, first + rows) of every column of `fused` [R,T]. One column is already contiguous
// and is returned as a view; wider inputs are copied into workspace.
Tensor fused_rows(const Tensor& fused, int first, int rows, WorkspaceArena& workspace,
                  cudaStream_t stream) {
    const int tokens = fused.ne[1];
    auto* source     = static_cast<std::byte*>(fused.data) + static_cast<std::size_t>(first) * 2;
    if (tokens == 1) { return Tensor(source, DType::BF16, {rows, 1}); }
    auto out                 = workspace.alloc(DType::BF16, {rows, tokens});
    const std::size_t width  = static_cast<std::size_t>(rows) * 2;
    const std::size_t pitch  = static_cast<std::size_t>(fused.ne[0]) * 2;
    const cudaError_t status = cudaMemcpy2DAsync(out.data, width, source, pitch, width,
                                                 static_cast<std::size_t>(tokens),
                                                 cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gemma4_layer: fused split: ") +
                                 cudaGetErrorString(status));
    }
    return out;
}

} // namespace

Weight bf16_matrix(const void* data, int rows, int cols) {
    Weight weight;
    weight.qtype          = QType::BF16;
    weight.layout         = QuantLayout::Contiguous;
    weight.qdata          = data;
    weight.n              = rows;
    weight.k              = cols;
    weight.ndim           = 2;
    weight.shape[0]       = rows;
    weight.shape[1]       = cols;
    return weight;
}

std::size_t gemma4_layer_workspace_bytes(const Gemma4LayerWeights& weights, int tokens) {
    const auto q     = static_cast<std::size_t>(weights.q_heads) * weights.head_dim;
    const auto k     = static_cast<std::size_t>(weights.kv_heads) * weights.head_dim;
    const auto h     = static_cast<std::size_t>(weights.hidden);
    const auto inter = static_cast<std::size_t>(weights.intermediate);
    // qkv, q, qn, attn; k and v inside qkv plus k, kn, v, vn; hidden_n, projected, normed;
    // gate_up, gate, up, activated.
    const std::size_t elements = 4 * q + 6 * k + 3 * h + 5 * inter;
    const std::size_t linear =
        std::max({projection_bytes(weights.qkv, weights.qkv_policy, tokens),
                  projection_bytes(weights.o_proj, weights.o_policy, tokens),
                  projection_bytes(weights.gate_up, weights.gate_up_policy, tokens),
                  projection_bytes(weights.down, weights.down_policy, tokens)});
    return elements * 2 * static_cast<std::size_t>(tokens) + linear + kAllocations * kAlignment;
}

std::size_t gemma4_logits_workspace_bytes(const Weight& output_head, int tokens) {
    return static_cast<std::size_t>(output_head.k) * 2 * static_cast<std::size_t>(tokens) +
           projection_bytes(output_head, ops::LinearPolicy::A16Only, tokens) + 2 * kAlignment;
}

void gemma4_layer(Tensor& x, const Gemma4LayerWeights& weights, const Tensor& positions,
                  Tensor& k_pages, Tensor& v_pages, const Tensor& block_table,
                  const Tensor& v_norm_ones, ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                  cudaStream_t stream) {
    const int tokens = x.ne[1];
    const int hidden = weights.hidden;
    const int dim    = weights.head_dim;
    const int q_rows = weights.q_heads * dim;
    const int k_rows = weights.kv_heads * dim;
    const int inter  = weights.intermediate;
    if (x.ne[0] != hidden || tokens <= 0) {
        throw std::invalid_argument("gemma4_layer: x must be [hidden,T]");
    }
    const int qkv_rows = q_rows + (weights.global_layer ? 1 : 2) * k_rows;
    require_matrix(weights.qkv, qkv_rows, hidden, "qkv");
    require_matrix(weights.o_proj, hidden, q_rows, "o_proj");
    require_matrix(weights.gate_up, 2 * inter, hidden, "gate_up");
    require_matrix(weights.down, hidden, inter, "down");

    auto hidden_n  = workspace.alloc(DType::BF16, {hidden, tokens});
    auto qkv       = workspace.alloc(DType::BF16, {qkv_rows, tokens});
    auto qn        = workspace.alloc(DType::BF16, {q_rows, tokens});
    auto kn        = workspace.alloc(DType::BF16, {k_rows, tokens});
    auto vn        = workspace.alloc(DType::BF16, {k_rows, tokens});
    auto attn      = workspace.alloc(DType::BF16, {q_rows, tokens});
    auto projected = workspace.alloc(DType::BF16, {hidden, tokens});
    auto normed    = workspace.alloc(DType::BF16, {hidden, tokens});

    ops::rmsnorm(x, norm_weight(weights.input_norm, hidden), weights.rms_eps, false, hidden_n,
                 stream);
    project(hidden_n, weights.qkv, weights.qkv_policy, qkv, gemm, workspace, stream);
    const Tensor q     = fused_rows(qkv, 0, q_rows, workspace, stream);
    const Tensor k_raw = fused_rows(qkv, q_rows, k_rows, workspace, stream);
    Tensor q_heads(q.data, DType::BF16, {dim, weights.q_heads, tokens});
    Tensor qn_heads(qn.data, DType::BF16, {dim, weights.q_heads, tokens});
    Tensor k_heads(k_raw.data, DType::BF16, {dim, weights.kv_heads, tokens});
    Tensor kn_heads(kn.data, DType::BF16, {dim, weights.kv_heads, tokens});
    ops::rmsnorm(q_heads, norm_weight(weights.q_norm, dim), weights.rms_eps, false, qn_heads,
                 stream);
    ops::rmsnorm(k_heads, norm_weight(weights.k_norm, dim), weights.rms_eps, false, kn_heads,
                 stream);
    Tensor vn_heads(vn.data, DType::BF16, {dim, weights.kv_heads, tokens});
    Tensor ones(v_norm_ones.data, DType::BF16, {weights.head_dim});
    if (weights.global_layer) {
        ops::rmsnorm(k_heads, ones, weights.rms_eps, false, vn_heads, stream);
    } else {
        const Tensor v = fused_rows(qkv, q_rows + k_rows, k_rows, workspace, stream);
        Tensor v_heads(v.data, DType::BF16, {dim, weights.kv_heads, tokens});
        ops::rmsnorm(v_heads, ones, weights.rms_eps, false, vn_heads, stream);
    }
    ops::rope(positions, weights.rotary_dim, weights.rope_theta, qn_heads, kn_heads, stream);
    ops::gemma4_kv_append(kn_heads, vn_heads, positions, k_pages, v_pages, block_table, stream);
    Tensor attn_heads(attn.data, DType::BF16, {dim, weights.q_heads, tokens});
    ops::gemma4_attention(qn_heads, k_pages, v_pages, block_table, positions, weights.window, 1.0f,
                          attn_heads, stream);
    project(attn, weights.o_proj, weights.o_policy, projected, gemm, workspace, stream);
    ops::rmsnorm(projected, norm_weight(weights.post_attn_norm, hidden), weights.rms_eps, false,
                 normed, stream);
    ops::residual_add(normed, x, stream);

    ops::rmsnorm(x, norm_weight(weights.pre_ff_norm, hidden), weights.rms_eps, false, hidden_n,
                 stream);
    auto gate_up = workspace.alloc(DType::BF16, {2 * inter, tokens});
    project(hidden_n, weights.gate_up, weights.gate_up_policy, gate_up, gemm, workspace, stream);
    const Tensor gate = fused_rows(gate_up, 0, inter, workspace, stream);
    const Tensor up   = fused_rows(gate_up, inter, inter, workspace, stream);
    auto activated    = workspace.alloc(DType::BF16, {inter, tokens});
    ops::geglu(gate, up, activated, stream);
    project(activated, weights.down, weights.down_policy, projected, gemm, workspace, stream);
    ops::rmsnorm(projected, norm_weight(weights.post_ff_norm, hidden), weights.rms_eps, false,
                 normed, stream);
    ops::residual_add(normed, x, stream);
    ops::residual_scale(x, weights.layer_scalar, stream);
}

void gemma4_layers(const Tensor& ids, Tensor& hidden, const Weight& token_embedding,
                   std::span<const Gemma4LayerWeights> layers,
                   std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                   const Tensor& v_norm_ones, ops::Bf16GemmContext& gemm,
                   WorkspaceArena& workspace, cudaStream_t stream) {
    if (layers.size() != caches.size()) {
        throw std::invalid_argument("gemma4_layers: one cache per layer");
    }
    ops::embedding(ids, token_embedding, hidden, stream);
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        auto scope = workspace.scope();
        gemma4_layer(hidden, layers[layer], positions, *caches[layer].k_pages,
                     *caches[layer].v_pages, *caches[layer].block_table, v_norm_ones, gemm,
                     workspace, stream);
    }
}

void gemma4_logits(const Tensor& hidden, Tensor& logits, const void* final_norm,
                   const Weight& output_head, float softcap, float rms_eps,
                   ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream) {
    if (hidden.ne[1] != logits.ne[1]) {
        throw std::invalid_argument("gemma4_logits: one logits column per hidden column");
    }
    if (output_head.n != logits.ne[0] || output_head.k != hidden.ne[0]) {
        throw std::invalid_argument("gemma4_logits: output head must be [vocab,hidden]");
    }
    auto scope  = workspace.scope();
    auto normed = workspace.alloc(DType::BF16, {hidden.ne[0], hidden.ne[1]});
    ops::rmsnorm(hidden, norm_weight(final_norm, hidden.ne[0]), rms_eps, false, normed, stream);
    project(normed, output_head, ops::LinearPolicy::A16Only, logits, gemm, workspace, stream);
    ops::logit_softcap(logits, softcap, stream);
}

void gemma4_prefill(const Tensor& ids, Tensor& hidden, Tensor& logits,
                    const Weight& token_embedding, const void* final_norm,
                    const Weight& output_head, float softcap, float rms_eps,
                    std::span<const Gemma4LayerWeights> layers,
                    std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                    const Tensor& v_norm_ones, ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                    cudaStream_t stream) {
    gemma4_layers(ids, hidden, token_embedding, layers, caches, positions, v_norm_ones, gemm,
                  workspace, stream);
    gemma4_logits(hidden, logits, final_norm, output_head, softcap, rms_eps, gemm, workspace,
                  stream);
}

} // namespace ninfer::models::gemma4
