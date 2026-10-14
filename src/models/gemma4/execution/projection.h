#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::models::gemma4 {

[[nodiscard]] inline Tensor norm_weight(const void* data, int width) {
    return Tensor(const_cast<void*>(data), DType::BF16, {width});
}

// out = weight * x for contiguous BF16 x [K,T] and out [N,T]. BF16 runs through cuBLAS, wide FP8
// A16 inputs through cuBLAS with per-call widening, everything else through ops::linear.
void project(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& out,
             ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);

// Arena bytes `project` takes for at most `tokens` columns.
[[nodiscard]] std::size_t projection_bytes(const Weight& weight, ops::LinearPolicy policy,
                                           int tokens);

} // namespace ninfer::models::gemma4
