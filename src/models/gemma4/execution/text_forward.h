#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <span>

namespace ninfer::models::gemma4 {

// Contiguous BF16 matrix [rows, cols] stored with K fastest: index = k + cols * n.
[[nodiscard]] Weight bf16_matrix(const void* data, int rows, int cols);

// One Gemma 4 text block. x is contiguous BF16 [hidden, T] and is updated in place.
// Projections are native Weights in any format ops::linear registers (BF16 runs through cuBLAS).
// qkv holds the query rows, then the key rows, then (sliding layers only) the value rows;
// gate_up holds the gate rows, then the up rows. Each policy is the artifact's permission for
// that projection's input.
struct Gemma4LayerWeights {
    const void* input_norm = nullptr;
    Weight qkv;
    const void* q_norm = nullptr;
    const void* k_norm = nullptr;
    Weight o_proj;
    const void* post_attn_norm = nullptr;
    const void* pre_ff_norm    = nullptr;
    Weight gate_up;
    Weight down;
    ops::LinearPolicy qkv_policy     = ops::LinearPolicy::A16Only;
    ops::LinearPolicy o_policy       = ops::LinearPolicy::A16Only;
    ops::LinearPolicy gate_up_policy = ops::LinearPolicy::A16Only;
    ops::LinearPolicy down_policy    = ops::LinearPolicy::A16Only;
    const void* post_ff_norm         = nullptr;
    float layer_scalar               = 1.0f;
    int hidden                       = 0;
    int intermediate                 = 0;
    int head_dim                     = 0;
    int q_heads                      = 0;
    int kv_heads                     = 0;
    int window                       = 0;
    int rotary_dim                   = 0;
    float rope_theta                 = 10000.0f;
    float rms_eps                    = 1.0e-6f;
    bool global_layer                = false;
};

void gemma4_layer(Tensor& x, const Gemma4LayerWeights& weights, const Tensor& positions,
                  Tensor& k_pages, Tensor& v_pages, const Tensor& block_table,
                  ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);

// Workspace bytes gemma4_layer and gemma4_logits need for at most `tokens` columns.
[[nodiscard]] std::size_t gemma4_layer_workspace_bytes(const Gemma4LayerWeights& weights,
                                                       int tokens);
[[nodiscard]] std::size_t gemma4_logits_workspace_bytes(const Weight& output_head, int tokens);

struct Gemma4CacheLayer {
    Tensor* k_pages           = nullptr;
    Tensor* v_pages           = nullptr;
    const Tensor* block_table = nullptr;
};

// Embed ids into `hidden` [hidden,T] and run every layer, appending K/V at `positions`.
void gemma4_layers(const Tensor& ids, Tensor& hidden, const Weight& token_embedding,
                   std::span<const Gemma4LayerWeights> layers,
                   std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                   ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);

// Final norm, output head, and logit softcap for the columns of `hidden` [hidden,C] into
// `logits` [vocab,C]. `hidden` is not modified.
void gemma4_logits(const Tensor& hidden, Tensor& logits, const void* final_norm,
                   const Weight& output_head, float softcap, float rms_eps,
                   ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);

// Embed, run every layer, final norm, output head, then logit softcap.
// `hidden` and `logits` are caller storage, [hidden,T] and [vocab,T].
void gemma4_prefill(const Tensor& ids, Tensor& hidden, Tensor& logits,
                    const Weight& token_embedding, const void* final_norm,
                    const Weight& output_head, float softcap, float rms_eps,
                    std::span<const Gemma4LayerWeights> layers,
                    std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                    ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::gemma4
