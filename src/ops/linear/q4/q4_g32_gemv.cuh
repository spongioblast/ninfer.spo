#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_g32_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
