#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream);

// codes and scales already hold the A8 activation. Only Gemma's 5376x21504 down projection.
void fp8_dispatch_ready(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                        std::uint8_t* codes, float* scales, WorkspaceArena* workspace,
                        cudaStream_t stream);

} // namespace ninfer::ops::detail
