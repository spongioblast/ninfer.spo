#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gemma4_qkv_prep_launch(const Tensor& qkv, const Tensor& q_norm, const Tensor& k_norm,
                            const Tensor& positions, int rotary_dim, float theta, float eps,
                            bool v_from_k, Tensor& q_out, Tensor& k_pages, Tensor& v_pages,
                            const Tensor& block_table, cudaStream_t stream);

} // namespace ninfer::ops::detail
