#include "runtime/engine/prefill_graph_stats.h"

#include <atomic>

namespace ninfer::runtime {
namespace {
std::atomic_flag tally_lock = ATOMIC_FLAG_INIT;
PrefillGraphTally process_tally;
} // namespace

void note_prefill_graph(PrefillGraphNote note, std::uint32_t reason) {
    while (tally_lock.test_and_set(std::memory_order_acquire)) {}
    switch (note) {
    case PrefillGraphNote::Hit: ++process_tally.hits; break;
    case PrefillGraphNote::Capture: ++process_tally.captures; break;
    case PrefillGraphNote::EagerFirst: ++process_tally.eager_first; break;
    case PrefillGraphNote::Ineligible: ++process_tally.ineligible; break;
    case PrefillGraphNote::AuditMatched: ++process_tally.audited; break;
    case PrefillGraphNote::AuditMismatch:
        ++process_tally.audited;
        ++process_tally.audit_mismatches;
        break;
    }
    if (reason != 0) { process_tally.last_reason = reason; }
    tally_lock.clear(std::memory_order_release);
}

PrefillGraphTally consume_prefill_graph_tally() {
    while (tally_lock.test_and_set(std::memory_order_acquire)) {}
    const PrefillGraphTally out = process_tally;
    process_tally = {};
    tally_lock.clear(std::memory_order_release);
    return out;
}

} // namespace ninfer::runtime
