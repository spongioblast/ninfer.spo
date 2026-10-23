#pragma once

#include "ops/linear/q4/q4_operands.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

// A16 q4_g32 for more than 16 tokens. Activations stay BF16.
void q4_g32_wide(const Q4LinearOperands& operands, __nv_bfloat16* out, cudaStream_t stream);

} // namespace ninfer::ops::detail
