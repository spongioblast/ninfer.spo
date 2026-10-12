#pragma once

#include <cstdint>

namespace ninfer::runtime {

// One process, one engine worker. The tally is exchanged onto the request that just finished
// a prefill unit.
struct PrefillGraphTally {
    std::uint64_t hits        = 0;
    std::uint64_t captures    = 0;
    std::uint64_t eager_first = 0;
    std::uint64_t ineligible  = 0;
    std::uint64_t audited          = 0;
    std::uint64_t audit_mismatches = 0;
    std::uint32_t last_reason = 0;
};

enum class PrefillGraphNote : std::uint8_t {
    Hit,
    Capture,
    EagerFirst,
    Ineligible,
    AuditMatched,
    AuditMismatch,
};

void note_prefill_graph(PrefillGraphNote note, std::uint32_t reason = 0);
[[nodiscard]] PrefillGraphTally consume_prefill_graph_tally();

} // namespace ninfer::runtime
