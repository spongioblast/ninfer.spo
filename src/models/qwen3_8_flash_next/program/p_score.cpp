// Causal scoring for the perplexity engine.
// The sequence plan has no MTP, vision, or CUDA graphs. One idle state slot and one KV row are enough.

#include "models/qwen3_8_flash_next/program/p3.h"
#include "program/execution/causal_score.h"

namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime {

std::vector<float> P3Program::causal_score(qwen3_8_flash_next::PreparedPromptData prompt,
                                           std::uint32_t first_target) {
    if (!causal_scoring || !score_hidden || !score_logprobs_host ||
        workspace_plan.causal_score == 0 || backend != SpeculativeBackend::None) {
        throw std::logic_error("Program was not constructed for causal scoring");
    }
    for (const Lane& slot : lanes_) {
        if (slot.bound) { throw std::logic_error("causal score requires an idle program"); }
    }
    program::validate_causal_score_request(prompt.token_ids.size(), capacity, first_target,
                                           prompt.has_media());

    const auto token_count              = static_cast<std::uint32_t>(prompt.token_ids.size());
    const std::uint32_t predictor_count = token_count - 1U;
    const std::uint32_t entitlement =
        1U + (predictor_count - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);

    program::CausalScoreTiles tiles(prompt.token_ids, first_target, *score_hidden, kCausalScoreTile,
                                    *score_logprobs_host, device);
    program::CausalScoreLane lane(*state_store, *text_kv_addresses, entitlement, predictor_count,
                                  device.stream);
    try {
        const std::int32_t state_slot = lane.state_slot();
        const auto flush              = [&] {
            if (tiles.staged_columns() == 0) { return; }
            work.reset();
            const auto columns = static_cast<std::int32_t>(tiles.staged_columns());
            Tensor logits      = work.alloc(DType::BF16, {TextConfig::output_rows, columns});
            Tensor target_ids  = work.alloc(DType::I32, {columns});
            Tensor logprobs    = work.alloc(DType::FP32, {columns});
            ops::linear(tiles.staged_hidden(), model.output_head, logits, device.stream);
            tiles.score(logits, target_ids, logprobs, TextConfig::token_domain);
            work.reset();
        };
        const auto prefill = [&](std::uint32_t cursor, std::uint32_t nominal) {
            schedule::PrefillContext schedule_state{
                .execution = execution_core(),
                .text_kv =
                    decoder->text_kv.execution_view(text_kv_addresses->execution_row(lane.address())),
                .mtp_kv                        = {},
                .text_cache                    = decoder->text_kv,
                .mtp_cache                     = nullptr,
                .text_kv_base                  = cursor,
                .sampling                      = nullptr,
                .rewrite_checkpoint_hidden     = nullptr,
                .rewrite_checkpoint_mtp_hidden = nullptr,
                .state_source_slot             = state_slot,
                .state_destination_slot        = state_slot,
                .mtp_proposal_extent           = 0};
            return schedule::prefill_text_chunk(schedule_state,
                                                std::span<const TokenId>(prompt.token_ids), nominal,
                                                std::nullopt, false);
        };
        program::prefill_scored_columns(tiles, lane, *text_kv_addresses, predictor_count,
                                        prefill_chunk, prefill_hidden, prefill, flush);
        auto output = std::move(tiles).finish();
        lane.release();
        return output;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        throw;
    }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime
