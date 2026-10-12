// Grammar masks for one prefill sample or one decode row.

#include "models/qwen3_8_flash_next/program/p3.h"

namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime {

std::uint32_t P3Program::apply_mask(runtime::TokenMaskProvider* masks, std::size_t compact_row,
                                    std::uint32_t lane_index, std::span<const TokenId> drafts) {
    if (masks == nullptr || !masks->constrained(compact_row)) {
        lane().request.sampling.mask = {};
        lane().grammar_dead          = 0;
        return 0;
    }
    if (lane_index >= max_concurrency) {
        throw std::logic_error("grammar mask lane is outside the plan");
    }
    if (!grammar_masks_device) {
        grammar_mask_words =
            (static_cast<std::uint32_t>(TextConfig::token_domain) + 31U) / 32U;
        const std::size_t positions =
            static_cast<std::size_t>(qwen3_8_flash_next::kMtpDecodeMaximumDrafts) + 1U;
        const std::size_t bytes = static_cast<std::size_t>(max_concurrency) * positions *
                                  grammar_mask_words * sizeof(std::uint32_t);
        grammar_masks_device.emplace(bytes);
        grammar_masks_host.emplace(bytes);
    }
    const std::size_t positions =
        static_cast<std::size_t>(qwen3_8_flash_next::kMtpDecodeMaximumDrafts) + 1U;
    const std::size_t offset = static_cast<std::size_t>(lane_index) * positions * grammar_mask_words;
    auto* host = static_cast<std::uint32_t*>(grammar_masks_host->data()) + offset;
    const std::size_t used = (drafts.size() + 1U) * grammar_mask_words;
    std::span<std::uint32_t> words(host, used);
    const std::uint32_t dead = masks->fill(compact_row, drafts, words);
    auto* device_words       = static_cast<std::uint32_t*>(grammar_masks_device->p) + offset;
    CUDA_CHECK(cudaMemcpyAsync(device_words, host, words.size_bytes(), cudaMemcpyHostToDevice,
                               device.stream));
    masks->uploaded(compact_row, words.size_bytes());
    lane().request.sampling.mask = {device_words, static_cast<std::int32_t>(grammar_mask_words)};
    lane().grammar_dead          = dead;
    return dead;
}

} // namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime
