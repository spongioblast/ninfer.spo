#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <span>

namespace ninfer::models::gemma4 {

// One Gemma 4 text block. x is contiguous BF16 [hidden, T] and is updated in place.
// Projections are contiguous BF16 matrices stored with K fastest: index = k + K * n.
struct Gemma4LayerWeights {
    const void* input_norm      = nullptr;
    const void* q_proj          = nullptr;
    const void* k_proj          = nullptr;
    const void* v_proj          = nullptr;
    const void* q_norm          = nullptr;
    const void* k_norm          = nullptr;
    const void* o_proj          = nullptr;
    const void* post_attn_norm  = nullptr;
    const void* pre_ff_norm     = nullptr;
    const void* gate_proj       = nullptr;
    const void* up_proj         = nullptr;
    const void* down_proj       = nullptr;
    const void* post_ff_norm    = nullptr;
    float layer_scalar          = 1.0f;
    int hidden                  = 0;
    int intermediate            = 0;
    int head_dim                = 0;
    int q_heads                 = 0;
    int kv_heads                = 0;
    int window                  = 0;
    int rotary_dim              = 0;
    float rope_theta            = 10000.0f;
    float rms_eps               = 1.0e-6f;
    bool global_layer           = false;
};

void gemma4_layer(Tensor& x, const Gemma4LayerWeights& weights, const Tensor& positions,
                  Tensor& k_pages, Tensor& v_pages, const Tensor& block_table,
                  const Tensor& v_norm_ones, ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                  cudaStream_t stream);

struct Gemma4CacheLayer {
    Tensor* k_pages          = nullptr;
    Tensor* v_pages          = nullptr;
    const Tensor* block_table = nullptr;
};

// Embed ids into `hidden` [hidden,T] and run every layer, appending K/V at `positions`.
void gemma4_layers(const Tensor& ids, Tensor& hidden, const void* token_embedding, int vocab,
                   std::span<const Gemma4LayerWeights> layers,
                   std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                   const Tensor& v_norm_ones, ops::Bf16GemmContext& gemm,
                   WorkspaceArena& workspace, cudaStream_t stream);

// Final norm, output head, and logit softcap for the columns of `hidden` [hidden,C] into
// `logits` [vocab,C]. `hidden` is not modified.
void gemma4_logits(const Tensor& hidden, Tensor& logits, const void* final_norm,
                   const void* output_head, float softcap, float rms_eps,
                   ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);

// Embed, run every layer, final norm, output head, then logit softcap.
// `hidden` and `logits` are caller storage, [hidden,T] and [vocab,T].
void gemma4_prefill(const Tensor& ids, Tensor& hidden, Tensor& logits, const void* token_embedding,
                    int vocab, const void* final_norm, const void* output_head, float softcap,
                    float rms_eps, std::span<const Gemma4LayerWeights> layers,
                    std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                    const Tensor& v_norm_ones, ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                    cudaStream_t stream);

} // namespace ninfer::models::gemma4
