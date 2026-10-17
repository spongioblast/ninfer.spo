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

// out = weight * x for contiguous BF16 x [K,T] and out [N,T]. BF16 goes through cuBLAS. Wide FP8
// A16 inputs go through cuBLAS with per-call widening; everything else through ops::linear.
void project(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& out,
             ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);

// Codes for one A8 activation [rows, tokens]. The producer writes them; project_down_ready
// consumes them and does not quantize again. Only the 5376x21504 down projection.
struct DownA8 {
    std::uint8_t* codes  = nullptr;
    float* scales        = nullptr;
};

[[nodiscard]] DownA8 prepare_down_a8(WorkspaceArena& workspace, int rows, int tokens);

void project_down_ready(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& out,
                        DownA8 ready, ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                        cudaStream_t stream);

// Arena bytes `project` takes for at most `tokens` columns.
[[nodiscard]] std::size_t projection_bytes(const Weight& weight, ops::LinearPolicy policy,
                                           int tokens);

} // namespace ninfer::models::gemma4
