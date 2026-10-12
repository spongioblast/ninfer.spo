#pragma once

// Every NINFER_* switch is read here, once. Callers use runtime_switches().
// Option switches can also be set by serve flags, which win over the environment.
// NINFER_DEBUG_* switches are diagnostics only and have no flag.

#include <cstdint>
#include <optional>
#include <string>

namespace ninfer {

struct RuntimeSwitches {
    // Unset means the device default (spin).
    std::optional<std::string> cuda_sync;
    std::optional<int> numa_node;
    bool high_priority = true;
    bool thread_tuning = true;
    std::uint64_t pin_ceiling_bytes = 0;

    bool planner_trace = false;
    std::string logits_dir;
    std::string state_dir;
    std::string state_frontier;

    int prefill_graph_max_tokens = 64;
    int prefill_graph_cache      = 64;
    bool prefill_graph_audit     = false;
    bool prefill_graphs_forced_off = false;
    bool prefill_graph_memlog    = false;

    // Empty keeps the platform default (spin on Windows, sleep elsewhere).
    std::string decode_wait;
    std::optional<int> wake_margin_us;
    int round_stats = 0;
};

struct RuntimeSwitchFlags {
    std::optional<std::string> cuda_sync;
    std::optional<bool> high_priority;
    std::optional<bool> thread_tuning;
    std::optional<std::uint64_t> pin_ceiling_gib;
    std::optional<std::string> decode_wait;
    std::optional<int> wake_margin_us;
    std::optional<int> prefill_graph_max_tokens;
    std::optional<int> prefill_graph_cache;
    std::optional<bool> prefill_graphs_off;
};

[[nodiscard]] const RuntimeSwitches& runtime_switches();

// Call on the main thread before the engine, process tuning, or any device starts.
void apply_runtime_switch_flags(const RuntimeSwitchFlags& flags);

} // namespace ninfer
