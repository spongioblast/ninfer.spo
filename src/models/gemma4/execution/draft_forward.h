#pragma once

#include "models/gemma4/execution/text_forward.h"

#include <cstddef>
#include <span>
#include <vector>

namespace ninfer::models::gemma4 {

// The MTP assistant. Its layers reuse Gemma4LayerWeights: qkv holds only the query rows and
// k_norm is unused. Layer i attends to the target cache of layer kv_source[i] and writes no K/V.
struct Gemma4DraftWeights {
    Weight input_projection;  // [hidden, 2*backbone] over [token embedding; carried state]
    Weight output_projection; // [backbone, hidden]
    Weight output_head;       // [vocab, hidden]
    const void* final_norm = nullptr;
    std::vector<Gemma4LayerWeights> layers;
    std::vector<int> kv_source;
    int hidden     = 0;
    int backbone   = 0;
    int vocab      = 0;
    float rms_eps  = 1.0e-6f;
};

/**
 * One draft step for a single sequence at the last target position.
 *
 * token and position are device I32 [1]; position is the last position whose K/V the target has
 * written, and every step of a round uses the same one. carry is BF16 [backbone,1]: on entry the
 * target's final-normed hidden state at `position` (first step) or the previous step's
 * projection, on exit this step's projection. next receives the argmax draft token. The token
 * embedding is the target's, already scaled by sqrt(backbone).
 */
void gemma4_draft_step(const Tensor& token, const Tensor& position, Tensor& carry, Tensor& next,
                       const Weight& token_embedding, const Gemma4DraftWeights& draft,
                       std::span<const Gemma4CacheLayer> target_caches,
                       ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                       cudaStream_t stream);

[[nodiscard]] std::size_t gemma4_draft_workspace_bytes(const Gemma4DraftWeights& draft);

} // namespace ninfer::models::gemma4
