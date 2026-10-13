#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gemma4_attention_launch(const Tensor& q, const Tensor& k_pages, const Tensor& v_pages,
                             const Tensor& block_table, const Tensor& positions, int window,
                             float scale, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
