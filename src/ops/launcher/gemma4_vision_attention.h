#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gemma4_vision_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor* mask, Tensor& scores, Tensor& weights, Tensor& out,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
