#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Gemma 4 vision encoder element passes. Each matches the float (or double) host arithmetic
 * it replaced operation for operation, without contraction, so the results are the same bits.
 *
 * gemma4_vision_scale_patches: out = bf16(2 * (x - 0.5)) for contiguous BF16 x/out of equal
 * shape (processor pixels in [0, 1] to [-1, 1]).
 *
 * gemma4_vision_axial_rope: q and k are contiguous BF16 [72,16,T], rotated in place. table is
 * contiguous FP32 [72,T]: per token cos and sin of the 18 x angles, then cos and sin of the 18 y
 * angles. Dims [0,36) rotate-half with the x angles and [36,72) with the y angles:
 *   x[o+i] = bf16(a*c - b*s), x[o+18+i] = bf16(b*c + a*s), a = x[o+i], b = x[o+18+i].
 *
 * gemma4_vision_pool: hidden is contiguous BF16 [W,T], out contiguous BF16 [W,P]. Bin b sums
 * the tokens members[offsets[b] .. offsets[b+1]) in that order in double, then
 *   out[d,b] = bf16(float((sum * gain - bias[d]) * scale[d])).
 * offsets is I32 [P+1], members I32 [offsets[P]], bias and scale BF16 [W].
 */
void gemma4_vision_scale_patches(const Tensor& patches, Tensor& out, cudaStream_t stream);

void gemma4_vision_axial_rope(const Tensor& table, Tensor& q, Tensor& k, cudaStream_t stream);

void gemma4_vision_pool(const Tensor& hidden, const Tensor& offsets, const Tensor& members,
                        const Tensor& bias, const Tensor& scale, double gain, Tensor& out,
                        cudaStream_t stream);

} // namespace ninfer::ops
