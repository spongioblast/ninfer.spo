#pragma once

#include <cstdint>
#include <vector>

namespace ninfer::models::gemma4 {

constexpr int kKvPageTokens = 64;

// Sliding layers keep their K/V in a ring of pages: logical page p lives in physical page
// p % ring. A forward of up to `chunk` tokens appends its K/V before it attends, so the ring
// must hold chunk + window - 1 positions for the oldest query's window to survive the appends.
[[nodiscard]] inline int sliding_ring_pages(std::uint32_t chunk, int window) {
    const std::uint32_t tokens = chunk + static_cast<std::uint32_t>(window) - 1U;
    return static_cast<int>((tokens + kKvPageTokens - 1U) / kKvPageTokens);
}

[[nodiscard]] inline std::vector<std::int32_t> ring_block_table(int logical_pages, int ring_pages) {
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages));
    for (int page = 0; page < logical_pages; ++page) {
        table[static_cast<std::size_t>(page)] = page % ring_pages;
    }
    return table;
}

} // namespace ninfer::models::gemma4
