#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Row-scaled FP8 projection through cuBLAS for wide inputs:
 *
 *   out[n,t] = BF16(scale[n] * sum_k e4m3[n,k] * x[k,t])
 *
 * The E4M3 codes are widened to BF16 (exact) in workspace, cuBLAS accumulates and stores FP32,
 * and the BF16 row scale is applied once before the BF16 store: the arithmetic of the FP8 A16
 * kernels behind ops::linear, in cuBLAS's summation order.
 *
 * w is FP8_E4M3FN_ROW_BF16 in the RowScale layout with BF16 row scales, x is contiguous BF16
 * [K,T] and out contiguous BF16 [N,T]; out must not overlap x or the weight. The workspace holds
 * fp8_row_cublas_workspace_bytes(N, K, T) and must not overlap x, w or out.
 */
[[nodiscard]] std::size_t fp8_row_cublas_workspace_bytes(std::int32_t n, std::int32_t k,
                                                         std::int32_t tokens);

void fp8_row_cublas(const Tensor& x, const Weight& w, Tensor& out, Bf16GemmContext& gemm,
                    WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
