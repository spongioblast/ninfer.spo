#include "core/device.h"
#include "core/precise_sleep.h"
#include "core/runtime_options.h"

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_M_ARM64)
#include <intrin.h>
#endif

namespace ninfer {
namespace {

struct SyncSchedule {
    std::string_view name;
    unsigned int flags;
};

constexpr SyncSchedule kSyncSchedules[] = {
    {"spin", cudaDeviceScheduleSpin},
    {"blocking", cudaDeviceScheduleBlockingSync},
    {"yield", cudaDeviceScheduleYield},
    {"auto", cudaDeviceScheduleAuto},
};

unsigned int sync_schedule_from_environment() {
    const std::optional<std::string>& value = runtime_switches().cuda_sync;
    if (!value) { return cudaDeviceScheduleSpin; }
    for (const auto& schedule : kSyncSchedules) {
        if (schedule.name == *value) { return schedule.flags; }
    }
    throw std::invalid_argument("NINFER_CUDA_SYNC must be spin, blocking, yield, or auto");
}

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

bool context_lost(cudaError_t err) noexcept {
    switch (err) {
    case cudaErrorDeviceUninitialized:
    case cudaErrorContextIsDestroyed:
    case cudaErrorIllegalAddress:
    case cudaErrorNoDevice:
        return true;
    default:
        return false;
    }
}

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    const char* name = cudaGetErrorName(err);
    const char* text = cudaGetErrorString(err);
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr, name, text);
    // Sleep or a device reset drops the context. Throw so the engine can finish held
    // requests and stop. Other CUDA failures still abort.
    if (context_lost(err)) {
        throw std::runtime_error(std::string("CUDA device context lost: ") + name + ": " + text);
    }
    std::abort();
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    const unsigned int sync_flags = sync_schedule_from_environment();
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid CUDA device id"); }

    bind_to_current_thread();

    err = cudaSetDeviceFlags(sync_flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDeviceFlags failed", err));
    }

    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err                  = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
    }

    stream          = compute;
    transfer_stream = load;
}

DeviceContext::~DeviceContext() {
    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), transfer_stream(other.transfer_stream),
      props(other.props) {
    other.stream          = nullptr;
    other.transfer_stream = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);

    device          = other.device;
    props           = other.props;
    stream          = other.stream;
    transfer_stream = other.transfer_stream;

    other.stream          = nullptr;
    other.transfer_stream = nullptr;
    return *this;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device));
}

int DeviceContext::compute_capability() const noexcept { return props.major * 10 + props.minor; }

int DeviceContext::multiprocessor_count() const noexcept { return props.multiProcessorCount; }

DeviceExecutionView DeviceContext::execution_view() const noexcept {
    return {.stream = stream, .multiprocessor_count = multiprocessor_count()};
}

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

const char* DeviceContext::sync_mode() const {
    bind_to_current_thread();
    unsigned int flags    = 0;
    const cudaError_t err = cudaGetDeviceFlags(&flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceFlags failed", err));
    }
    for (const auto& schedule : kSyncSchedules) {
        if ((flags & cudaDeviceScheduleMask) == schedule.flags) { return schedule.name.data(); }
    }
    throw std::runtime_error("unknown CUDA synchronization schedule");
}

void DeviceContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream)); }

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : CudaEventTimer(ctx, ctx.stream) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    if (stream == nullptr) { throw std::invalid_argument("CUDA timer stream is null"); }
    ctx.bind_to_current_thread();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    cudaError_t err   = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx) : device_(ctx.device) {
    ctx.bind_to_current_thread();
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    device_ = other.device_;
    event_  = std::exchange(other.event_, nullptr);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
}

void CudaCompletionEvent::record_external(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecordWithFlags(event_, stream, cudaEventRecordExternal));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return false;
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    CUDA_CHECK(cudaEventSynchronize(event_));
}

namespace {

std::int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

float percentile_us(float* samples, int count, int percent) {
    std::vector<float> copy(samples, samples + count);
    std::sort(copy.begin(), copy.end());
    const int index = std::min(count - 1, (count * percent) / 100);
    return copy[static_cast<std::size_t>(index)];
}

void cpu_pause() {
#if defined(_M_X64) || defined(__x86_64__)
    _mm_pause();
#elif defined(_M_ARM64)
    __yield();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
}

} // namespace

LowLatencyStreamWait::LowLatencyStreamWait(const DeviceContext& ctx) {
    ctx.bind_to_current_thread();
#ifdef _WIN32
    mode_ = Mode::Spin;
#else
    mode_ = Mode::Sleep;
#endif
    const std::string& selected = runtime_switches().decode_wait;
    if (selected == "spin") {
        mode_ = Mode::Spin;
    } else if (selected == "sleep") {
        mode_ = Mode::Sleep;
    } else if (selected == "block") {
        mode_ = Mode::Block;
    } else if (selected == "stock") {
        mode_ = Mode::Stock;
    }
    if (const auto margin = runtime_switches().wake_margin_us) {
        wake_margin_us_ = *margin;
    } else if (mode_ == Mode::Sleep) {
#ifdef _WIN32
        wake_margin_us_ = 1000;
#endif
    }
    stats_every_ = runtime_switches().round_stats;
    if (stats_every_ > 0) {
        host_prep_us_ = new float[static_cast<std::size_t>(stats_every_)];
        device_us_    = new float[static_cast<std::size_t>(stats_every_)];
        wake_us_      = new float[static_cast<std::size_t>(stats_every_)];
    }
    std::fprintf(stderr, "decode wait: %s, wake margin %d us\n",
                 mode_ == Mode::Spin    ? "spin"
                 : mode_ == Mode::Sleep ? "sleep"
                 : mode_ == Mode::Block ? "block"
                                        : "stock",
                 wake_margin_us_);
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&launch_));
    if (mode_ == Mode::Block) {
        CUDA_CHECK(cudaEventCreateWithFlags(&stop_, cudaEventBlockingSync));
    } else {
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
}

LowLatencyStreamWait::~LowLatencyStreamWait() {
    delete[] host_prep_us_;
    delete[] device_us_;
    delete[] wake_us_;
    if (stop_ != nullptr) { (void)cudaEventDestroy(stop_); }
    if (launch_ != nullptr) { (void)cudaEventDestroy(launch_); }
    if (start_ != nullptr) { (void)cudaEventDestroy(start_); }
}

void LowLatencyStreamWait::mark_launch(cudaStream_t stream) {
    if (stats_every_ <= 0) { return; }
    CUDA_CHECK(cudaEventRecord(launch_, stream));
    launch_marked_ = true;
}

void LowLatencyStreamWait::begin(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
    begin_host_ns_ = steady_ns();
    started_       = true;
}

void LowLatencyStreamWait::wait(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(stop_, stream));
    if (mode_ == Mode::Block) {
        CUDA_CHECK(cudaEventSynchronize(stop_));
    } else {
        if ((mode_ == Mode::Sleep || mode_ == Mode::Stock) && started_ &&
            expected_us_ > static_cast<double>(wake_margin_us_)) {
            const auto deadline =
                begin_host_ns_ +
                static_cast<std::int64_t>((expected_us_ - wake_margin_us_) * 1000.0);
            const auto remaining = deadline - steady_ns();
            if (remaining > 0) {
                if (mode_ == Mode::Stock) {
                    std::this_thread::sleep_for(std::chrono::nanoseconds(remaining));
                } else {
                    precise_sleep_ns(remaining);
                }
            }
        }
        for (;;) {
            const cudaError_t status = cudaEventQuery(stop_);
            if (status == cudaSuccess) { break; }
            if (status != cudaErrorNotReady) { CUDA_CHECK(status); }
            cpu_pause();
        }
    }
    if (started_) {
        float ms = 0.0F;
        if (cudaEventElapsedTime(&ms, start_, stop_) == cudaSuccess) {
            const double measured = static_cast<double>(ms) * 1000.0;
            expected_us_ = expected_us_ == 0.0 ? measured : 0.75 * expected_us_ + 0.25 * measured;
        }
        if (stats_every_ > 0 && launch_marked_) {
            float host_ms  = 0.0F;
            float dev_ms   = 0.0F;
            float total_ms = 0.0F;
            if (cudaEventElapsedTime(&host_ms, start_, launch_) == cudaSuccess &&
                cudaEventElapsedTime(&dev_ms, launch_, stop_) == cudaSuccess &&
                cudaEventElapsedTime(&total_ms, start_, stop_) == cudaSuccess) {
                const double wake_us = static_cast<double>(steady_ns() - begin_host_ns_) / 1000.0 -
                                       static_cast<double>(total_ms) * 1000.0;
                host_prep_us_[stats_i_] = host_ms * 1000.0F;
                device_us_[stats_i_]    = dev_ms * 1000.0F;
                wake_us_[stats_i_]      = static_cast<float>(wake_us);
                ++stats_i_;
                if (stats_i_ == stats_every_) {
                    const float host_p50 = percentile_us(host_prep_us_, stats_every_, 50);
                    const float host_p99 = percentile_us(host_prep_us_, stats_every_, 99);
                    const float dev_p50  = percentile_us(device_us_, stats_every_, 50);
                    const float dev_p99  = percentile_us(device_us_, stats_every_, 99);
                    const float wake_p50 = percentile_us(wake_us_, stats_every_, 50);
                    const float wake_p99 = percentile_us(wake_us_, stats_every_, 99);
                    std::clog << "round-stats n=" << stats_every_ << " host_prep " << host_p50 << "/"
                              << host_p99 << " device " << dev_p50 << "/" << dev_p99 << " wake "
                              << wake_p50 << "/" << wake_p99 << " (us)\n";
                    stats_i_ = 0;
                }
            }
        }
        launch_marked_ = false;
        started_       = false;
    }
}

} // namespace ninfer
