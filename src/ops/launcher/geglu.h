#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void geglu_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
