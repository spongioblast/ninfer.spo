#pragma once

#include "artifact/reader.h"
#include "core/tensor.h"
#include "core/host_worker_pool.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_8_flash_next {

inline constexpr std::int32_t kPleHeads        = 16;
inline constexpr std::int32_t kPleHeadWidth    = 160;
inline constexpr std::int32_t kPleEmbeddingDim = kPleHeads * kPleHeadWidth;
inline constexpr std::int32_t kPleEosToken     = 248044;
inline constexpr std::uint64_t kPleRows        = 320001536;

using PleIds = std::array<std::uint32_t, kPleHeads>;

/** Tokens before a position that its IDs depend on (the trigram reach). */
inline constexpr std::size_t kPleHistoryTokens = 2;

/** Compute the exact bigram/trigram row IDs for one complete token sequence. */
void compute_ple_ids(std::span<const std::int32_t> tokens, std::span<PleIds> output);

/**
 * First token to hash so that IDs from `position` on equal those of the complete sequence. The
 * EOS segment reset only changes the trigram of the token after an EOS, which the window holds.
 */
constexpr std::size_t ple_history_begin(std::size_t position) {
    return position > kPleHistoryTokens ? position - kPleHistoryTokens : 0;
}

/** Gather the 16 FP8 or BF16 head rows for each token into token-major storage. */
void gather_ple(const artifact::MappedRange& table, DType dtype,
                std::span<const PleIds> ids, std::span<std::byte> output,
                HostWorkerPool* workers = nullptr);

} // namespace ninfer::models::qwen3_8_flash_next
