#pragma once

#include "core/process_tuning.h"
#include "ninfer/types.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

inline constexpr std::string_view kHostOptionsUsage =
    "[--ple-residency mapped|resident] [--ple-large-pages] [--numa-node N|off] "
    "[--host-ram-budget-gib N] [--kv-headroom-mib N]";

inline std::string host_options_help(std::string_view indent) {
    std::string text;
    for (const std::string_view line : {
             "--ple-residency defaults to resident on Windows (mapped PLE reads are hard faults "
             "there) and mapped elsewhere",
             "--ple-large-pages needs the \"Lock pages in memory\" right; falls back to locked "
             "normal pages",
             "--numa-node defaults to NINFER_NUMA_NODE, else the GPU's NUMA node",
             "--host-ram-budget-gib defaults to 200; resident PLE + host KV + host state + 8 GiB "
             "must fit; 0 skips the check",
             "--kv-headroom-mib sets the device memory --kv-capacity auto leaves free",
         }) {
        text.append(indent).append(line).append("\n");
    }
    return text;
}

inline std::uint64_t parse_host_u64(const char* text, const char* flag) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + flag);
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + flag + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

// Host placement flags shared by ninfer-spo, ninfer-spo-serve, and ninfer_bench. Returns false
// when `arg` is not one of them; `value()` consumes the flag's argument. --kv-headroom-mib is
// stored in `kv_headroom_bytes` and applied by the caller where it resolves --kv-capacity.
template <class NextValue>
bool parse_host_option(std::string_view arg, NextValue&& value, HostResidencyOptions& options,
                       std::optional<std::size_t>& kv_headroom_bytes) {
    const auto text_value = [&] { return std::string(value()); };
    if (arg == "--ple-residency") {
        const std::string mode = text_value();
        if (mode == "mapped") {
            options.ple = PleResidency::Mapped;
        } else if (mode == "resident") {
            options.ple = PleResidency::Resident;
        } else {
            throw std::invalid_argument("--ple-residency must be mapped or resident");
        }
    } else if (arg == "--ple-large-pages") {
        options.ple_large_pages = true;
    } else if (arg == "--numa-node") {
        const std::string text = text_value();
        if (text == "off") {
            options.numa_node = -1;
        } else {
            const std::uint64_t node = parse_host_u64(text.c_str(), "--numa-node");
            if (node > 63) { throw std::invalid_argument("--numa-node must be in [0,63] or off"); }
            options.numa_node = static_cast<int>(node);
        }
    } else if (arg == "--host-ram-budget-gib") {
        const std::uint64_t gib = parse_host_u64(text_value().c_str(), "--host-ram-budget-gib");
        if (gib > (std::numeric_limits<std::size_t>::max() >> 30U)) {
            throw std::invalid_argument("--host-ram-budget-gib is out of range");
        }
        options.host_ram_budget_bytes = static_cast<std::size_t>(gib << 30U);
    } else if (arg == "--kv-headroom-mib") {
        const std::uint64_t mib = parse_host_u64(text_value().c_str(), "--kv-headroom-mib");
        if (mib > (std::numeric_limits<std::size_t>::max() >> 20U)) {
            throw std::invalid_argument("--kv-headroom-mib is out of range");
        }
        kv_headroom_bytes = static_cast<std::size_t>(mib << 20U);
    } else {
        return false;
    }
    return true;
}

// Resolves the host NUMA node, applies process and main-thread tuning, and records the node for
// resident host tables. Call before any large host allocation and before the engine starts.
inline void apply_host_tuning(HostResidencyOptions& options, int cuda_device) {
    std::string source;
    const int node = resolve_numa_node(options.numa_node, cuda_device, &source);
    apply_process_tuning(node);
    options.numa_node = node;
    std::clog << "host tuning: NUMA node " << node << " (" << source << ")\n";
}

} // namespace ninfer::product
