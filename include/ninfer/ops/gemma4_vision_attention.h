#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Dense non-causal attention for the Gemma 4 vision encoder.
 *
 *   score[j]   = 1.0 * dot(q[:,h,i], k[:,h,j])     (valid keys only)
 *   out[:,h,i] = softmax(score) * v[:,h,j]
 *
 * q/k/v/out are contiguous BF16 [72,16,T]. Every query attends every key. An optional contiguous
 * I32 mask [T] (nonzero = valid) drops pad keys from every softmax; pad queries write zeros.
 * Without a mask every token is valid. Hq = Hkv = 16. The oracle evaluates the visible set in
 * FP64. The Op writes all of out, leaves inputs unchanged, and uses no workspace.
 */
void gemma4_vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask,
                             Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
