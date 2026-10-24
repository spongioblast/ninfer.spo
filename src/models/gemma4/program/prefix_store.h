#pragma once

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::gemma4 {

// Identity of one prompt position: the token id, or for an image position a digest of the image
// and the position inside it (bit 63 set), so prompts that differ only in an image never share
// a prefix.
using PositionKey = std::uint64_t;

// One paged K or V tensor: `pages` pages of `page_bytes` each, page p at base + p * page_bytes.
struct PagedTensor {
    std::uint8_t* base     = nullptr;
    std::size_t page_bytes = 0;
};

// Retained prompt prefixes ("checkpoints") of the Gemma program.
//
// The global layers' K/V is a pool of reference-counted 64-token pages. The live sequence maps
// its logical pages onto the pool; a checkpoint keeps references to the pages of its prefix, so
// capture copies nothing and a shared page is copied only when the live sequence writes it.
//
// The sliding layers keep K/V in a ring, which the next tokens overwrite, so a checkpoint
// snapshots the ring pages of the last window - 1 positions plus some slack. The slack lets a
// prompt that diverges a little before the frontier still resume. Snapshots sit in device slots;
// the least recently used checkpoint moves to a pinned host arena (pages and snapshot) when the
// slots or the pool run out, and is dropped when the arena is full too.
class PrefixStore {
public:
    struct Config {
        std::vector<PagedTensor> global;
        std::vector<PagedTensor> ring;
        int pool_pages                = 0;
        int ring_pages                = 0;
        int window                    = 0;
        std::uint32_t device_slots    = 0;
        std::size_t host_bytes        = 0;
        cudaStream_t stream           = nullptr;
    };

    struct Match {
        std::uint64_t id    = 0;
        std::uint32_t reuse = 0;
        bool on_host        = false;
    };

    struct Stats {
        std::uint64_t captures      = 0;
        std::uint64_t device_hits   = 0;
        std::uint64_t host_hits     = 0;
        std::uint64_t demotions     = 0;
        std::uint64_t drops         = 0;
        std::uint32_t device_points = 0;
        std::uint32_t host_points   = 0;
    };

    explicit PrefixStore(Config config);
    ~PrefixStore();
    PrefixStore(const PrefixStore&)            = delete;
    PrefixStore& operator=(const PrefixStore&) = delete;

    // Snapshot pages per checkpoint and its device bytes, for sizing before construction.
    [[nodiscard]] static int snapshot_pages(int ring_pages, int window);
    [[nodiscard]] static std::size_t snapshot_bytes(std::span<const std::size_t> ring_page_bytes,
                                                    int ring_pages, int window);

    // A free pool page with one reference. Demotes or drops checkpoints to free one; throws when
    // only the live sequence holds pages.
    [[nodiscard]] int allocate_page();
    void retain(int page) noexcept;
    void release(int page) noexcept;
    [[nodiscard]] bool shared(int page) const noexcept;
    // Copies page `from` to `to` in every global tensor.
    void copy_page(int from, int to);

    // The checkpoint that resumes the most of `prompt`, at most `limit` positions.
    [[nodiscard]] std::optional<Match> best_match(std::span<const PositionKey> prompt,
                                                  std::uint32_t limit) const;
    // True when a checkpoint already ends exactly after `keys`.
    [[nodiscard]] bool covers(std::span<const PositionKey> keys) const;
    // Marks every checkpoint whose keys are a prefix of `prompt` as just used, so a checkpoint
    // that each request extends (the system prompt and tools) is not the first one evicted.
    void touch(std::span<const PositionKey> prompt);

    // Keeps the live state at frontier keys.size(): references to live_pages[0, pages of the
    // frontier) and a snapshot of the ring. The ring holds positions (ring_low, ring_high] at
    // most ring_pages * 64 back from ring_high; ring_high >= frontier - 1. Making room never
    // demotes or drops checkpoint `keep` (the one about to be restored). Returns false when the
    // snapshot could not resume anything or found no room.
    bool capture(std::vector<PositionKey> keys, std::span<const int> live_pages,
                 std::int64_t ring_low, std::int64_t ring_high,
                 std::optional<std::uint64_t> keep = std::nullopt);

    // Loads `match` into an empty live state: live_pages receives the pages of the first
    // match.reuse positions and the ring gets the snapshot. Returns the lowest position the
    // ring now holds.
    [[nodiscard]] std::int64_t restore(const Match& match, std::vector<int>& live_pages);

    [[nodiscard]] Stats stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::gemma4
