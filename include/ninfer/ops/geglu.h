#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * GeGLU with the tanh GELU approximation:
 *
 *   ideal[i] = gelu_tanh(gate[i]) * up[i]
 *   gelu_tanh(z) = 0.5*z*(1 + tanh(sqrt(2/pi)*(z + 0.044715*z^3))).
 *
 * `gate`, `up`, and `out` are same-shaped contiguous BF16 tensors. out must not overlap gate or
 * up. The oracle evaluates `ideal` in FP64 from the represented inputs. The BF16 output is
 * promoted and compared directly with that result; output storage rounding belongs to the Op's
 * numerical criterion, not the oracle. The Op writes all of out and uses no workspace.
 */
void geglu(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
