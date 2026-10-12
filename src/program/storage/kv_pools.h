#pragma once

// Fixed description of the KV pools a model owns. Flash-Next has a main pool and an MTP pool.
// qwen3_5 has one pool, or two when MTP is on. Sliding-window retention is not implemented here.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::program {

enum class KvRetention : std::uint8_t { Full };

struct KvPoolSpec {
    std::string_view name;
    std::uint32_t page_tokens    = 0;
    std::uint32_t head_dim       = 0;
    std::size_t bytes_per_page   = 0;
    KvRetention retention        = KvRetention::Full;
};

} // namespace ninfer::program
