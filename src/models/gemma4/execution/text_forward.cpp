#include "models/gemma4/execution/text_forward.h"

#include "models/gemma4/execution/projection.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gemma4_attention.h"
#include "ninfer/ops/gemma4_qkv_prep.h"
#include "ninfer/ops/geglu.h"
#include "ninfer/ops/logit_softcap.h"
#include "ninfer/ops/gemma4_residual_norm.h"
#include "ninfer/ops/rmsnorm.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ninfer::models::gemma4 {
namespace {

constexpr std::size_t kAllocations = 18;
constexpr std::size_t kAlignment   = 256;

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
    auto out                = workspace.alloc(DType::BF16, {rows, tokens});
    const std::size_t width = static_cast<std::size_t>(rows) * 2;
    const std::size_t pitch = static_cast<std::size_t>(fused.ne[0]) * 2;
    const cudaError_t status =
        cudaMemcpy2DAsync(out.data, width, source, pitch, width, static_cast<std::size_t>(tokens),
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
    weight.qtype    = QType::BF16;
    weight.layout   = QuantLayout::Contiguous;
    weight.qdata    = data;
    weight.n        = rows;
    weight.k        = cols;
    weight.ndim     = 2;
    weight.shape[0] = rows;
    weight.shape[1] = cols;
    return weight;
}

std::size_t gemma4_layer_workspace_bytes(const Gemma4LayerWeights& weights, int tokens) {
    const auto q     = static_cast<std::size_t>(weights.q_heads) * weights.head_dim;
    const auto k     = static_cast<std::size_t>(weights.kv_heads) * weights.head_dim;
    const auto h     = static_cast<std::size_t>(weights.hidden);
    const auto inter = static_cast<std::size_t>(weights.intermediate);
    // qkv (with k and v), qn, attn; hidden_n, projected; gate_up, gate, up, activated.
    const std::size_t elements = 3 * q + 2 * k + 2 * h + 5 * inter;
    const std::size_t scratch =
        std::max({projection_bytes(weights.qkv, weights.qkv_policy, tokens),
                  projection_bytes(weights.o_proj, weights.o_policy, tokens),
                  projection_bytes(weights.gate_up, weights.gate_up_policy, tokens),
                  projection_bytes(weights.down, weights.down_policy, tokens),
                  ops::gemma4_attention_workspace_capacity_bytes(weights.head_dim, weights.q_heads,
                                                                 weights.kv_heads, tokens)});
    return elements * 2 * static_cast<std::size_t>(tokens) + scratch + kAllocations * kAlignment;
}

std::size_t gemma4_logits_workspace_bytes(const Weight& output_head, int tokens) {
    return static_cast<std::size_t>(output_head.k) * 2 * static_cast<std::size_t>(tokens) +
           projection_bytes(output_head, ops::LinearPolicy::A16Only, tokens) + 2 * kAlignment;
}

namespace {

// hidden_n holds rmsnorm(x, input_norm) on entry. With next_input_norm the FFN tail leaves the
// next layer's input norm there, otherwise hidden_n is scratch.
void layer_body(Tensor& x, Tensor& hidden_n, const Gemma4LayerWeights& weights,
                const void* next_input_norm, const Tensor& positions, Tensor& k_pages,
                Tensor& v_pages, const Tensor& block_table, ops::Bf16GemmContext& gemm,
                WorkspaceArena& workspace, cudaStream_t stream) {
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

    auto qkv       = workspace.alloc(DType::BF16, {qkv_rows, tokens});
    auto qn        = workspace.alloc(DType::BF16, {q_rows, tokens});
    auto attn      = workspace.alloc(DType::BF16, {q_rows, tokens});
    auto projected = workspace.alloc(DType::BF16, {hidden, tokens});

    project(hidden_n, weights.qkv, weights.qkv_policy, qkv, gemm, workspace, stream);
    Tensor qn_heads(qn.data, DType::BF16, {dim, weights.q_heads, tokens});
    ops::gemma4_qkv_prep(qkv, norm_weight(weights.q_norm, dim), norm_weight(weights.k_norm, dim),
                         positions, weights.rotary_dim, weights.rope_theta, weights.rms_eps,
                         weights.global_layer, qn_heads, k_pages, v_pages, block_table, stream);
    Tensor attn_heads(attn.data, DType::BF16, {dim, weights.q_heads, tokens});
    ops::gemma4_attention(qn_heads, k_pages, v_pages, block_table, positions, weights.window, 1.0f,
                          attn_heads, workspace, stream);
    project(attn, weights.o_proj, weights.o_policy, projected, gemm, workspace, stream);
    const Tensor pre_ff_norm = norm_weight(weights.pre_ff_norm, hidden);
    ops::gemma4_residual_norm(projected, norm_weight(weights.post_attn_norm, hidden), 1.0f,
                              &pre_ff_norm, weights.rms_eps, x, &hidden_n, stream);

    auto gate_up = workspace.alloc(DType::BF16, {2 * inter, tokens});
    project(hidden_n, weights.gate_up, weights.gate_up_policy, gate_up, gemm, workspace, stream);
    const Tensor gate = fused_rows(gate_up, 0, inter, workspace, stream);
    const Tensor up   = fused_rows(gate_up, inter, inter, workspace, stream);
    auto activated    = workspace.alloc(DType::BF16, {inter, tokens});
    ops::geglu(gate, up, activated, stream);
    project(activated, weights.down, weights.down_policy, projected, gemm, workspace, stream);
    const Tensor next_norm = norm_weight(next_input_norm, hidden);
    ops::gemma4_residual_norm(projected, norm_weight(weights.post_ff_norm, hidden),
                              weights.layer_scalar,
                              next_input_norm != nullptr ? &next_norm : nullptr, weights.rms_eps, x,
                              next_input_norm != nullptr ? &hidden_n : nullptr, stream);
}

} // namespace

void gemma4_layer(Tensor& x, const Gemma4LayerWeights& weights, const Tensor& positions,
                  Tensor& k_pages, Tensor& v_pages, const Tensor& block_table,
                  ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream) {
    if (x.ne[0] != weights.hidden || x.ne[1] <= 0) {
        throw std::invalid_argument("gemma4_layer: x must be [hidden,T]");
    }
    auto hidden_n = workspace.alloc(DType::BF16, {weights.hidden, x.ne[1]});
    ops::rmsnorm(x, norm_weight(weights.input_norm, weights.hidden), weights.rms_eps, false,
                 hidden_n, stream);
    layer_body(x, hidden_n, weights, nullptr, positions, k_pages, v_pages, block_table, gemm,
               workspace, stream);
}

void gemma4_layers(const Tensor& ids, Tensor& hidden, const Weight& token_embedding,
                   std::span<const Gemma4LayerWeights> layers,
                   std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                   ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream) {
    if (layers.size() != caches.size()) {
        throw std::invalid_argument("gemma4_layers: one cache per layer");
    }
    ops::embedding(ids, token_embedding, hidden, stream);
    if (layers.empty()) { return; }
    auto scope    = workspace.scope();
    auto hidden_n = workspace.alloc(DType::BF16, {hidden.ne[0], hidden.ne[1]});
    ops::rmsnorm(hidden, norm_weight(layers[0].input_norm, hidden.ne[0]), layers[0].rms_eps, false,
                 hidden_n, stream);
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        auto layer_scope = workspace.scope();
        const void* next = layer + 1 < layers.size() ? layers[layer + 1].input_norm : nullptr;
        layer_body(hidden, hidden_n, layers[layer], next, positions, *caches[layer].k_pages,
                   *caches[layer].v_pages, *caches[layer].block_table, gemm, workspace, stream);
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
                    ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream) {
    gemma4_layers(ids, hidden, token_embedding, layers, caches, positions, gemm, workspace, stream);
    gemma4_logits(hidden, logits, final_norm, output_head, softcap, rms_eps, gemm, workspace,
                  stream);
}

} // namespace ninfer::models::gemma4
