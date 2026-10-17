#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Dense non-causal attention for the Gemma 4 vision encoder.
 *
 *   score[j]   = fp32(dot(q[:,h,i], k[:,h,j]))      (cuBLAS bf16 GEMM, fp32 store)
 *   weight[j] = bf16(softmax(score)[j])             (fp32 softmax, then round)
 *   out[:,h,i] = sum_j weight[j] * v[:,h,j]         (cuBLAS bf16 GEMM)
 *
 * q/k/v/out are contiguous BF16 [72,16,T]. scores is contiguous FP32 with at least 16*T*T
 * elements (col-major T×T per head); logits stay fp32 through the softmax. weights is
 * contiguous BF16 of the same count and receives the rounded probabilities. Every query
 * attends every key. An optional contiguous I32 mask [T] (nonzero = valid) drops pad keys
 * from every softmax; pad queries still attend. Without a mask every token is valid.
 * Hq = Hkv = 16. The oracle evaluates the visible set in FP64. The Op writes out, scores,
 * and weights, and leaves q/k/v unchanged.
 */
[[nodiscard]] inline std::size_t gemma4_vision_attention_workspace_bytes(int tokens) {
    const auto cells = static_cast<std::size_t>(16) * static_cast<std::size_t>(tokens) *
                       static_cast<std::size_t>(tokens);
    return cells * 4 + cells * 2;
}

void gemma4_vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask,
                             Tensor& scores, Tensor& weights, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
