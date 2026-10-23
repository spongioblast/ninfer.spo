#include "core/runtime_options.h"

#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>

namespace ninfer {
namespace {

std::string env_string(const char* name) {
    const char* text = std::getenv(name);
    return text == nullptr ? std::string{} : std::string{text};
}

void warn_ignored(const char* name, const char* text) {
    std::fprintf(stderr, "ninfer: ignoring %s=%s (not a valid value)\n", name, text);
}

int env_int(const char* name, int fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') { return fallback; }
    char* end         = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        warn_ignored(name, text);
        return fallback;
    }
    return static_cast<int>(parsed);
}

bool env_flag(const char* name) {
    const char* text = std::getenv(name);
    return text != nullptr && text[0] == '1' && text[1] == '\0';
}

bool env_present(const char* name) { return std::getenv(name) != nullptr; }

std::optional<int> env_numa() {
    const char* text = std::getenv("NINFER_NUMA_NODE");
    if (text == nullptr || *text == '\0') { return std::nullopt; }
    char* end         = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < -1 || parsed > 63) {
        warn_ignored("NINFER_NUMA_NODE", text);
        return std::nullopt;
    }
    return static_cast<int>(parsed);
}

std::uint64_t env_pin_ceiling() {
    const char* text = std::getenv("NINFER_PIN_CEILING_GIB");
    if (text == nullptr || *text == '\0') { return 0; }
    char* end                     = nullptr;
    const unsigned long long gib = std::strtoull(text, &end, 10);
    // strtoull accepts a sign and wraps "-1"; a ceiling past 2^34 GiB would overflow the shift.
    if (end == text || *end != '\0' || *text == '-' || *text == '+' ||
        gib > (std::numeric_limits<std::uint64_t>::max() >> 30U)) {
        warn_ignored("NINFER_PIN_CEILING_GIB", text);
        return 0;
    }
    return static_cast<std::uint64_t>(gib) << 30U;
}

void log_if_set(const RuntimeSwitches& switches) {
    const bool quiet = !switches.cuda_sync && !switches.numa_node && switches.high_priority &&
                       switches.thread_tuning && switches.pin_ceiling_bytes == 0 &&
                       !switches.planner_trace && switches.logits_dir.empty() &&
                       switches.state_dir.empty() && switches.state_frontier.empty() &&
                       switches.prefill_graph_max_tokens == 64 && switches.prefill_graph_cache == 64 &&
                       !switches.prefill_graph_audit && !switches.prefill_graphs_forced_off &&
                       !switches.prefill_graph_memlog && switches.decode_wait.empty() &&
                       !switches.wake_margin_us && switches.round_stats == 0;
    if (quiet) { return; }
    std::fprintf(stderr, "ninfer switches:");
    if (switches.cuda_sync) { std::fprintf(stderr, " cuda_sync=%s", switches.cuda_sync->c_str()); }
    if (switches.numa_node) { std::fprintf(stderr, " numa=%d", *switches.numa_node); }
    if (!switches.high_priority) { std::fprintf(stderr, " priority=normal"); }
    if (!switches.thread_tuning) { std::fprintf(stderr, " thread_tuning=off"); }
    if (switches.pin_ceiling_bytes != 0) {
        std::fprintf(stderr, " pin_ceiling_gib=%llu",
                     static_cast<unsigned long long>(switches.pin_ceiling_bytes >> 30U));
    }
    if (switches.planner_trace) { std::fprintf(stderr, " planner_trace"); }
    if (!switches.logits_dir.empty()) {
        std::fprintf(stderr, " logits_dir=%s", switches.logits_dir.c_str());
    }
    if (!switches.state_dir.empty()) {
        std::fprintf(stderr, " state_dir=%s", switches.state_dir.c_str());
    }
    if (!switches.state_frontier.empty()) {
        std::fprintf(stderr, " state_frontier=%s", switches.state_frontier.c_str());
    }
    if (switches.prefill_graph_max_tokens != 64) {
        std::fprintf(stderr, " prefill_graph_max_tokens=%d", switches.prefill_graph_max_tokens);
    }
    if (switches.prefill_graph_cache != 64) {
        std::fprintf(stderr, " prefill_graph_cache=%d", switches.prefill_graph_cache);
    }
    if (switches.prefill_graph_audit) { std::fprintf(stderr, " prefill_graph_audit"); }
    if (switches.prefill_graphs_forced_off) { std::fprintf(stderr, " prefill_graphs=off"); }
    if (switches.prefill_graph_memlog) { std::fprintf(stderr, " prefill_graph_memlog"); }
    if (!switches.decode_wait.empty()) {
        std::fprintf(stderr, " decode_wait=%s", switches.decode_wait.c_str());
    }
    if (switches.wake_margin_us) {
        std::fprintf(stderr, " wake_margin_us=%d", *switches.wake_margin_us);
    }
    if (switches.round_stats != 0) { std::fprintf(stderr, " round_stats=%d", switches.round_stats); }
    if (switches.q4_exact_prefill) { std::fprintf(stderr, " q4_exact_prefill=1"); }
    std::fprintf(stderr, "\n");
}

RuntimeSwitches& mutable_switches() {
    static RuntimeSwitches switches = [] {
        RuntimeSwitches out;
        if (env_present("NINFER_CUDA_SYNC")) { out.cuda_sync = env_string("NINFER_CUDA_SYNC"); }
        out.numa_node           = env_numa();
        const std::string priority = env_string("NINFER_PRIORITY");
        out.high_priority       = priority.empty() || priority != "normal";
        const std::string tuning = env_string("NINFER_THREAD_TUNING");
        out.thread_tuning       = !(tuning.size() == 1 && tuning[0] == '0');
        out.pin_ceiling_bytes   = env_pin_ceiling();
        out.planner_trace       = env_present("NINFER_DEBUG_PLANNER_TRACE");
        out.logits_dir          = env_string("NINFER_DEBUG_LOGITS_DIR");
        out.state_dir           = env_string("NINFER_DEBUG_STATE_DIR");
        out.state_frontier      = env_string("NINFER_DEBUG_STATE_FRONTIER");
        out.prefill_graph_max_tokens = env_int("NINFER_FLASH_NEXT_PREFILL_GRAPH_MAX_TOKENS", 64);
        out.prefill_graph_cache      = env_int("NINFER_FLASH_NEXT_PREFILL_GRAPH_CACHE", 64);
        out.prefill_graph_audit      = env_flag("NINFER_DEBUG_PREFILL_GRAPH_AUDIT");
        out.prefill_graphs_forced_off =
            env_string("NINFER_FLASH_NEXT_PREFILL_GRAPHS") == "0";
        out.prefill_graph_memlog = env_flag("NINFER_DEBUG_PREFILL_GRAPH_MEMLOG");
        const std::string decode_wait = env_string("NINFER_DECODE_WAIT");
        if (decode_wait == "spin" || decode_wait == "sleep" || decode_wait == "block" ||
            decode_wait == "stock") {
            out.decode_wait = decode_wait;
        } else if (!decode_wait.empty()) {
            warn_ignored("NINFER_DECODE_WAIT", decode_wait.c_str());
        }
        if (env_present("NINFER_WAKE_MARGIN_US")) {
            const int margin = env_int("NINFER_WAKE_MARGIN_US", 0);
            out.wake_margin_us = margin < 0 ? 0 : (margin > 5000 ? 5000 : margin);
        }
        const int stats = env_int("NINFER_DEBUG_ROUND_STATS", 0);
        out.round_stats = stats < 0 ? 0 : (stats > 10000 ? 10000 : stats);
        out.q4_exact_prefill = env_present("NINFER_Q4_EXACT");
        return out;
    }();
    return switches;
}

void log_once() {
    static const bool logged = [] {
        log_if_set(mutable_switches());
        return true;
    }();
    (void)logged;
}

} // namespace

const RuntimeSwitches& runtime_switches() {
    log_once();
    return mutable_switches();
}

void apply_runtime_switch_flags(const RuntimeSwitchFlags& flags) {
    RuntimeSwitches& out = mutable_switches();
    if (flags.cuda_sync) { out.cuda_sync = *flags.cuda_sync; }
    if (flags.high_priority) { out.high_priority = *flags.high_priority; }
    if (flags.thread_tuning) { out.thread_tuning = *flags.thread_tuning; }
    if (flags.pin_ceiling_gib) { out.pin_ceiling_bytes = *flags.pin_ceiling_gib << 30U; }
    if (flags.decode_wait) { out.decode_wait = *flags.decode_wait; }
    if (flags.wake_margin_us) { out.wake_margin_us = *flags.wake_margin_us; }
    if (flags.prefill_graph_max_tokens) {
        out.prefill_graph_max_tokens = *flags.prefill_graph_max_tokens;
    }
    if (flags.prefill_graph_cache) { out.prefill_graph_cache = *flags.prefill_graph_cache; }
    if (flags.prefill_graphs_off) { out.prefill_graphs_forced_off = *flags.prefill_graphs_off; }
}

} // namespace ninfer
