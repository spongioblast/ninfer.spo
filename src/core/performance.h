#pragma once

// No device reads, events, synchronization, or argument evaluation in ordinary builds.
// Bytes are a cold, once-per-Op useful-traffic model, not measured DRAM transactions.
#ifdef NINFER_PERFORMANCE_TRACE
#    include <nvtx3/nvToolsExt.h>

#    include <cstdint>
#    include <cstdio>

namespace ninfer::performance {
struct Work {
    std::uint64_t bytes_min   = 0;
    std::uint64_t bytes_max   = 0;
    std::uint64_t bf16_flops  = 0;
    std::uint64_t nvfp4_flops = 0;
    std::uint64_t fp32_flops  = 0;
};

// Use a separate domain so collecting work metadata does not depend on other NVTX ranges.
inline nvtxDomainHandle_t domain() noexcept {
    static const auto value = nvtxDomainCreateA("ninfer.performance");
    return value;
}

class Scope {
public:
    Scope(const char* stage, std::uint64_t tokens, std::uint64_t batch, std::uint64_t context,
          Work work) noexcept {
        char message[512];
        std::snprintf(
            message, sizeof(message), "ninfer.work/1|%s|%llu|%llu|%llu|%llu|%llu|%llu|%llu|%llu",
            stage, static_cast<unsigned long long>(tokens), static_cast<unsigned long long>(batch),
            static_cast<unsigned long long>(context),
            static_cast<unsigned long long>(work.bytes_min),
            static_cast<unsigned long long>(work.bytes_max),
            static_cast<unsigned long long>(work.bf16_flops),
            static_cast<unsigned long long>(work.nvfp4_flops),
            static_cast<unsigned long long>(work.fp32_flops));
        push(message);
    }

    explicit Scope(const char* region) noexcept { push(region); }

    ~Scope() noexcept { nvtxDomainRangePop(domain()); }

    Scope(const Scope&)            = delete;
    Scope& operator=(const Scope&) = delete;

private:
    static void push(const char* message) noexcept {
        nvtxEventAttributes_t event{};
        event.version       = NVTX_VERSION;
        event.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        event.messageType   = NVTX_MESSAGE_TYPE_ASCII;
        event.message.ascii = message;
        nvtxDomainRangePushEx(domain(), &event);
    }
};
} // namespace ninfer::performance

#    define NINFER_PERF_JOIN_IMPL(a, b) a##b
#    define NINFER_PERF_JOIN(a, b)      NINFER_PERF_JOIN_IMPL(a, b)
#    define NINFER_PERF_SCOPE(...)                                                                 \
        ::ninfer::performance::Scope NINFER_PERF_JOIN(performance_scope_, __LINE__)(__VA_ARGS__)
#else
#    define NINFER_PERF_SCOPE(...) ((void)0)
#endif
