#pragma once

#include "models/gemma4/frontend/image.h"
#include "models/gemma4/program/prefix_store.h"
#include "ninfer/types.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::gemma4 {

// A prefix shorter than this is cheaper to recompute than to keep or restore.
inline constexpr std::uint32_t kMinCheckpointTokens = 256;
// A host checkpoint is restored over PCIe (its ring snapshot alone is about 1 GiB for the 31B),
// so it must save at least this many prompt tokens over the live state.
inline constexpr std::uint32_t kMinHostRestoreGain = 1024;

// The key of every prompt position (see PositionKey).
[[nodiscard]] std::vector<PositionKey> prompt_keys(std::span<const TokenId> tokens,
                                                   std::span<const Gemma4PreparedImage> images,
                                                   std::span<const std::int32_t> image_scatter);

[[nodiscard]] inline PositionKey token_key(TokenId token) noexcept {
    return static_cast<std::uint32_t>(token);
}

// The live KV of the Gemma program and the checkpoints kept beside it.
//
// The live sequence holds `resident` positions. Its global pages come from the PrefixStore pool
// through the page table at `table`; the sliding layers write the ring. Before a request binds,
// the live state is kept as a checkpoint when the request would overwrite it, and the request
// then resumes from whichever of the live state and the checkpoints shares the longest prefix.
class PrefixCache {
public:
    struct Config {
        PrefixStore::Config store;
        int table_pages       = 0;
        std::int32_t* table   = nullptr;
        bool history          = true;
    };

    struct Source {
        std::uint32_t reuse = 0;
        std::optional<PrefixStore::Match> match;
    };

    explicit PrefixCache(Config config);
    ~PrefixCache();
    PrefixCache(const PrefixCache&)            = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    [[nodiscard]] const std::vector<PositionKey>& resident() const noexcept { return resident_; }

    // Makes the pages of positions [first, first + count) present and owned by the live
    // sequence alone, and uploads the page table when it changed. Call before the forward.
    void prepare_write(std::uint32_t first, std::uint32_t count);
    // Records that positions [first, first + count) now hold `keys`.
    void wrote(std::uint32_t first, std::span<const PositionKey> keys);
    void wrote_tokens(std::uint32_t first, std::span<const TokenId> tokens);
    void truncate(std::uint32_t length);

    // The longest resume for `keys`, keeping at least the last key to recompute. A checkpoint
    // wins over the live state only when it saves enough to pay for its restore.
    [[nodiscard]] Source best(std::span<const PositionKey> keys) const;
    // Loads `source` into the live state for `keys`, first keeping the displaced live state.
    void bind(const Source& source, std::span<const PositionKey> keys);
    // Keeps the first `length` live positions as a checkpoint, never evicting checkpoint `keep`.
    void capture(std::uint32_t length, std::optional<std::uint64_t> keep = std::nullopt);
    void clear() noexcept;

    [[nodiscard]] PrefixStore::Stats stats() const noexcept { return store_.stats(); }

private:
    [[nodiscard]] std::uint32_t live_reuse(std::span<const PositionKey> keys) const;
    void upload_table();

    PrefixStore store_;
    Config config_;
    std::vector<PositionKey> resident_;
    std::vector<int> live_pages_;
    PinnedHostBuffer table_host_;
    bool table_dirty_ = true;
    // The ring holds positions [ring_low_, ring_high_], at most ring_pages * 64 back.
    std::int64_t ring_low_  = 0;
    std::int64_t ring_high_ = -1;
};

} // namespace ninfer::models::gemma4
