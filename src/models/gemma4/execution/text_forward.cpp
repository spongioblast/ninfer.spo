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

#include <stdexcept>

namespace ninfer::models::gemma4 {
namespace {

Weight matrix(const void* data, int rows, int cols) {
    Weight weight;
    weight.qtype          = QType::BF16;
    weight.layout         = QuantLayout::Contiguous;
    weight.qdata          = data;
    weight.n              = rows;
    weight.k              = cols;
    weight.ndim           = 2;
    weight.shape[0]       = cols;
    weight.shape[1]       = rows;
    return weight;
}

Tensor norm_weight(const void* data, int width) {
    return Tensor(const_cast<void*>(data), DType::BF16, {width});
}

void project(const Tensor& x, const void* data, int rows, int cols, Tensor& out,
             ops::Bf16GemmContext& gemm, cudaStream_t stream) {
    gemm.launch(x, matrix(data, rows, cols), out, stream);
}

} // namespace

void gemma4_layer(Tensor& x, const Gemma4LayerWeights& weights, const Tensor& positions,
                  Tensor& k_pages, Tensor& v_pages, const Tensor& block_table,
                  const Tensor& v_norm_ones, ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                  cudaStream_t stream) {
    const int tokens = x.ne[1];
    const int hidden = weights.hidden;
    const int dim    = weights.head_dim;
    const int q_rows = weights.q_heads * dim;
    const int k_rows = weights.kv_heads * dim;
    if (x.ne[0] != hidden || tokens <= 0) {
        throw std::invalid_argument("gemma4_layer: x must be [hidden,T]");
    }
    auto hidden_n = workspace.alloc(DType::BF16, {hidden, tokens});
    auto q        = workspace.alloc(DType::BF16, {q_rows, tokens});
    auto qn       = workspace.alloc(DType::BF16, {q_rows, tokens});
    auto k_raw    = workspace.alloc(DType::BF16, {k_rows, tokens});
    auto kn       = workspace.alloc(DType::BF16, {k_rows, tokens});
    auto v        = workspace.alloc(DType::BF16, {k_rows, tokens});
    auto vn       = workspace.alloc(DType::BF16, {k_rows, tokens});
    auto attn     = workspace.alloc(DType::BF16, {q_rows, tokens});
    auto projected = workspace.alloc(DType::BF16, {hidden, tokens});
    auto normed   = workspace.alloc(DType::BF16, {hidden, tokens});
    auto gate     = workspace.alloc(DType::BF16, {weights.intermediate, tokens});
    auto up       = workspace.alloc(DType::BF16, {weights.intermediate, tokens});
    auto activated = workspace.alloc(DType::BF16, {weights.intermediate, tokens});

    ops::rmsnorm(x, norm_weight(weights.input_norm, hidden), weights.rms_eps, false, hidden_n,
                 stream);
    project(hidden_n, weights.q_proj, q_rows, hidden, q, gemm, stream);
    project(hidden_n, weights.k_proj, k_rows, hidden, k_raw, gemm, stream);
    Tensor q_heads(q.data, DType::BF16, {dim, weights.q_heads, tokens});
    Tensor qn_heads(qn.data, DType::BF16, {dim, weights.q_heads, tokens});
    Tensor k_heads(k_raw.data, DType::BF16, {dim, weights.kv_heads, tokens});
    Tensor kn_heads(kn.data, DType::BF16, {dim, weights.kv_heads, tokens});
    ops::rmsnorm(q_heads, norm_weight(weights.q_norm, dim), weights.rms_eps, false, qn_heads,
                 stream);
    ops::rmsnorm(k_heads, norm_weight(weights.k_norm, dim), weights.rms_eps, false, kn_heads,
                 stream);
    Tensor v_heads(v.data, DType::BF16, {dim, weights.kv_heads, tokens});
    Tensor vn_heads(vn.data, DType::BF16, {dim, weights.kv_heads, tokens});
    Tensor ones(v_norm_ones.data, DType::BF16, {weights.head_dim});
    if (weights.global_layer) {
        ops::rmsnorm(k_heads, ones, weights.rms_eps, false, vn_heads, stream);
    } else {
        project(hidden_n, weights.v_proj, k_rows, hidden, v, gemm, stream);
        ops::rmsnorm(v_heads, ones, weights.rms_eps, false, vn_heads, stream);
    }
    ops::rope(positions, weights.rotary_dim, weights.rope_theta, qn_heads, kn_heads, stream);
    ops::gemma4_kv_append(kn_heads, vn_heads, positions, k_pages, v_pages, block_table, stream);
    Tensor attn_heads(attn.data, DType::BF16, {dim, weights.q_heads, tokens});
    ops::gemma4_attention(qn_heads, k_pages, v_pages, block_table, positions, weights.window, 1.0f,
                          attn_heads, stream);
    project(attn, weights.o_proj, hidden, q_rows, projected, gemm, stream);
    ops::rmsnorm(projected, norm_weight(weights.post_attn_norm, hidden), weights.rms_eps, false,
                 normed, stream);
    ops::residual_add(normed, x, stream);

    ops::rmsnorm(x, norm_weight(weights.pre_ff_norm, hidden), weights.rms_eps, false, hidden_n,
                 stream);
    project(hidden_n, weights.gate_proj, weights.intermediate, hidden, gate, gemm, stream);
    project(hidden_n, weights.up_proj, weights.intermediate, hidden, up, gemm, stream);
    ops::geglu(gate, up, activated, stream);
    project(activated, weights.down_proj, hidden, weights.intermediate, projected, gemm, stream);
    ops::rmsnorm(projected, norm_weight(weights.post_ff_norm, hidden), weights.rms_eps, false,
                 normed, stream);
    ops::residual_add(normed, x, stream);
    ops::residual_scale(x, weights.layer_scalar, stream);
}

void gemma4_layers(const Tensor& ids, Tensor& hidden, const void* token_embedding, int vocab,
                   std::span<const Gemma4LayerWeights> layers,
                   std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                   const Tensor& v_norm_ones, ops::Bf16GemmContext& gemm,
                   WorkspaceArena& workspace, cudaStream_t stream) {
    if (layers.size() != caches.size()) {
        throw std::invalid_argument("gemma4_layers: one cache per layer");
    }
    Weight table;
    table.qtype        = QType::BF16;
    table.layout       = QuantLayout::Contiguous;
    table.qdata        = token_embedding;
    table.ndim         = 2;
    table.shape[0]     = vocab;
    table.shape[1]     = hidden.ne[0];
    ops::embedding(ids, table, hidden, stream);
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        auto scope = workspace.scope();
        gemma4_layer(hidden, layers[layer], positions, *caches[layer].k_pages,
                     *caches[layer].v_pages, *caches[layer].block_table, v_norm_ones, gemm,
                     workspace, stream);
    }
}

void gemma4_logits(const Tensor& hidden, Tensor& logits, const void* final_norm,
                   const void* output_head, float softcap, float rms_eps,
                   ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream) {
    if (hidden.ne[1] != logits.ne[1]) {
        throw std::invalid_argument("gemma4_logits: one logits column per hidden column");
    }
    auto scope  = workspace.scope();
    auto normed = workspace.alloc(DType::BF16, {hidden.ne[0], hidden.ne[1]});
    ops::rmsnorm(hidden, norm_weight(final_norm, hidden.ne[0]), rms_eps, false, normed, stream);
    project(normed, output_head, logits.ne[0], hidden.ne[0], logits, gemm, stream);
    ops::logit_softcap(logits, softcap, stream);
}

void gemma4_prefill(const Tensor& ids, Tensor& hidden, Tensor& logits, const void* token_embedding,
                    int vocab, const void* final_norm, const void* output_head, float softcap,
                    float rms_eps, std::span<const Gemma4LayerWeights> layers,
                    std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                    const Tensor& v_norm_ones, ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                    cudaStream_t stream) {
    gemma4_layers(ids, hidden, token_embedding, vocab, layers, caches, positions, v_norm_ones, gemm,
                  workspace, stream);
    gemma4_logits(hidden, logits, final_norm, output_head, softcap, rms_eps, gemm, workspace,
                  stream);
}

} // namespace ninfer::models::gemma4
