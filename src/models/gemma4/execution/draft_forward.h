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
    Weight output_head;       // [vocab, hidden]; FP8 when the registered 262144x1024 GEMV exists
    DeviceBuffer quantized_head; // owns the FP8 payload if the artifact stored the head as BF16
    // Owns the FP8 payloads of the assistant projections quantized at load.
    std::vector<DeviceBuffer> quantized_weights;
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
 * embedding is the target's; its gather scales the rows by sqrt(backbone).
 */
void gemma4_draft_step(const Tensor& token, const Tensor& position, Tensor& carry, Tensor& next,
                       const Gemma4Embedding& token_embedding, const Gemma4DraftWeights& draft,
                       std::span<const Gemma4CacheLayer> target_caches,
                       ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                       cudaStream_t stream);

/**
 * w draft steps in one pass, one per column. tokens and positions are I32 [w]; every position is
 * the last target position. carry_in and carry_out are BF16 [backbone,w] and may alias. top_ids
 * is I32 [k,w], the top-k draft tokens of each column. k == 1 is the argmax.
 */
void gemma4_draft_steps(const Tensor& tokens, const Tensor& positions, const Tensor& carry_in,
                        Tensor& carry_out, Tensor& top_ids, int k, const Gemma4Embedding& token_embedding,
                        const Gemma4DraftWeights& draft,
                        std::span<const Gemma4CacheLayer> target_caches,
                        ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                        cudaStream_t stream);

[[nodiscard]] std::size_t gemma4_draft_workspace_bytes(const Gemma4DraftWeights& draft,
                                                       int columns = 1);

} // namespace ninfer::models::gemma4
