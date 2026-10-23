#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gemma4_vision_scale_patches_launch(const Tensor& patches, Tensor& out, cudaStream_t stream);

void gemma4_vision_axial_rope_launch(const Tensor& table, Tensor& q, Tensor& k,
                                     cudaStream_t stream);

void gemma4_vision_pool_launch(const Tensor& hidden, const Tensor& offsets, const Tensor& members,
                               const Tensor& bias, const Tensor& scale, double gain, Tensor& out,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
