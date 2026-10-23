#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Dense non-causal attention for the Gemma 4 vision encoder.
 *
 *   score[j]   = fp32(dot(q[:,h,i], k[:,h,j]))
 *   weight[j] = bf16(softmax(score)[j])             (fp32 softmax, then round)
 *   out[:,h,i] = sum_j weight[j] * v[:,h,j]
 *
 * q/k/v/out are contiguous BF16 [72,16,T]. Logits stay fp32 through an exact softmax (the row
 * maximum and the sum are complete before any probability is formed); the probabilities are
 * rounded to bf16 before the value multiply. No T×T buffer is stored, so the op needs no
 * workspace. Every query attends every key. An optional contiguous I32 mask [T] (nonzero =
 * valid) drops pad keys from every softmax; pad queries still attend, and a query without a
 * valid key gets zeros. Hq = Hkv = 16. The oracle evaluates the visible set in FP64. The Op
 * writes out and leaves q/k/v unchanged.
 */
void gemma4_vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask,
                             Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
