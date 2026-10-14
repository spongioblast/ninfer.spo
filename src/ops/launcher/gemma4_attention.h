#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Number of key chunks per query; 1 means no split and no workspace.
int gemma4_attention_chunks(std::int32_t tokens, std::int32_t kv_heads);

// partial is FP32 [D, chunks, Hq, T] and stats FP32 [2, chunks, Hq, T]; both unused when
// gemma4_attention_chunks(...) is 1.
void gemma4_attention_launch(const Tensor& q, const Tensor& k_pages, const Tensor& v_pages,
                             const Tensor& block_table, const Tensor& positions, int window,
                             float scale, Tensor& out, float* partial, float* stats,
                             cudaStream_t stream);

// Tensor-core path for wide prefill chunks (D=256 or 512); needs no workspace.
void gemma4_attention_mma_launch(const Tensor& q, const Tensor& k_pages, const Tensor& v_pages,
                                 const Tensor& block_table, const Tensor& positions, int window,
                                 float scale, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
