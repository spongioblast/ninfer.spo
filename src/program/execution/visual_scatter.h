#pragma once

// Shifted visual scatter for MTP prefill, shared by the model families.

#include "core/arena.h"
#include "core/tensor.h"
#include "program/speculative/mtp_alignment.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::program {

// Composes the generic scatter Op from the family-provided shifted-window interpretation.
void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       const MtpVisualOverlap& overlap,
                                       Tensor& destination_indices, cudaStream_t stream);

} // namespace ninfer::program
