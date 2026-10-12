#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Computes one vocabulary argmax per column:
 *
 *   out[t] = min argmax_{0 <= v < valid_rows} float(logits[v,t]).
 *
 * `logits` is contiguous BF16 [physical_rows,T], `out` is contiguous I32 [T], and
 * 1 <= valid_rows <= physical_rows. Physical rows [valid_rows,physical_rows) do not
 * participate. Equal maxima select the lowest row index. `out` must not overlap `logits`.
 * The Op has no workspace and changes no state other than writing all of `out`.
 */
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

// Exact BF16-head argmax over an approximate-logit shortlist. candidate_* are caller scratch
// of shape [2*ceil(valid_rows/512), T].
void shortlist_exact_argmax(const Tensor& hidden, const Tensor& approximate_logits,
                            std::int32_t valid_rows, const Weight& exact_head,
                            const std::int32_t* id_map, Tensor& candidate_ids,
                            Tensor& candidate_scores, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
