#pragma once

#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include "runtime/contract/resources.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::runtime_support {

// Short template pieces between rewrite execution frontiers (the generation prompt's assistant
// opener and think markers) run as one pass. Pieces of 2..8 tokens take the per-token small-T
// kernels, so the target model computes the same bits for the group as for its pieces; a 1-token
// piece takes different kernels and is never grouped. The MTP head is not split-invariant, so the
// grouping depends only on the frontier list: a replay of the same history groups exactly like
// the live request did.
inline constexpr std::uint32_t kRewriteGroupMaxTokens = 8;
inline constexpr std::uint32_t kRewriteGroupMinPiece  = 2;

// Next pass end forced by `frontiers` (sorted, unique) for a pass starting at `cursor`, or
// nullopt when no rewrite frontier splits the interval before `limit`.
inline std::optional<std::uint32_t> rewrite_split_after(std::span<const std::uint32_t> frontiers,
                                                        std::uint32_t cursor, std::uint32_t limit) {
    auto next = std::upper_bound(frontiers.begin(), frontiers.end(), cursor);
    if (next == frontiers.end() || *next >= limit) { return std::nullopt; }
    const std::uint32_t first = *next;
    std::uint32_t end         = cursor;
    for (;;) {
        const std::uint32_t piece_end = next == frontiers.end() ? limit : std::min(*next, limit);
        const std::uint32_t piece     = piece_end - end;
        if (piece < kRewriteGroupMinPiece || piece_end - cursor > kRewriteGroupMaxTokens) { break; }
        end = piece_end;
        if (end == limit) { return std::nullopt; }
        ++next;
    }
    return end == cursor ? first : end;
}

// Every rewrite frontier is honoured. The old tail-only measurement changed tokens and stayed off.
inline std::span<const std::uint32_t> honoured_rewrite_frontiers(
    std::span<const std::uint32_t> frontiers) {
    return frontiers;
}

inline void include_rebuild_boundary(std::uint32_t& tail_begin, std::uint32_t boundary,
                                     std::uint32_t frontier) noexcept {
    if (boundary <= frontier) { tail_begin = std::max(tail_begin, boundary); }
}

inline void advance_segmented_rebuild_work(runtime::PrefillWork& work, std::uint32_t tail_begin,
                                           std::uint32_t previous_frontier, std::uint32_t frontier,
                                           std::uint32_t prefill_chunk) {
    if (frontier < previous_frontier || work.tokens != previous_frontier ||
        tail_begin > previous_frontier || prefill_chunk == 0) {
        throw std::logic_error("sequence rebuild work is not aligned with its frontier");
    }
    if (frontier == previous_frontier) { return; }

    const std::uint64_t old_tail = previous_frontier - tail_begin;
    const std::uint64_t new_tail = frontier - tail_begin;
    const auto chunks_for        = [prefill_chunk](std::uint64_t tokens) {
        return tokens == 0 ? 0ULL : 1ULL + (tokens - 1ULL) / prefill_chunk;
    };
    const std::uint64_t old_tail_chunks = chunks_for(old_tail);
    const std::uint64_t new_tail_chunks = chunks_for(new_tail);
    if (work.chunks < old_tail_chunks) {
        throw std::logic_error("sequence rebuild chunk accounting is invalid");
    }

    const runtime::PrefillWork total = runtime::make_prefill_work(
        0, frontier, work.vision_items, work.vision_patches, prefill_chunk);
    work.chunks          = work.chunks - old_tail_chunks + new_tail_chunks;
    work.tokens          = total.tokens;
    work.attention_pairs = total.attention_pairs;
}

} // namespace ninfer::models::qwen3_8_flash_next::runtime_support
