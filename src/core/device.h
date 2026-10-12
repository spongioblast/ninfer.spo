#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;

    [[nodiscard]] constexpr DeviceExecutionView
    on_stream(cudaStream_t target_stream) const noexcept {
        return {target_stream, multiprocessor_count};
    }
};

// Flash-Next call sites still pass a stream. The SM count is the current device's.
inline DeviceExecutionView execution_for_stream(cudaStream_t stream) {
    int device = 0;
    int sms    = 1;
    if (cudaGetDevice(&device) == cudaSuccess) {
        (void)cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    }
    if (sms < 1) { sms = 1; }
    return DeviceExecutionView{stream, sms};
}

struct DeviceContext {
    int device                   = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    int compute_capability() const noexcept;
    int multiprocessor_count() const noexcept;
    DeviceExecutionView execution_view() const noexcept;
    std::size_t total_vram() const noexcept;
    const char* sync_mode() const;
    void synchronize() const;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(cudaStream_t stream);
    // Capture a record node observable by the host while later graph nodes are still running.
    // Only call during stream capture; this event must outlive every graph referencing it.
    void record_external(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

// Low-latency completion wait for steady per-step work such as decode rounds. With the
// blocking-sync schedule a synchronize costs an interrupt wake-up after the GPU goes idle.
// This waiter sleeps until shortly before the expected completion, an exponential average of
// measured GPU durations, and polls the completion event for the remainder.
class LowLatencyStreamWait {
public:
    explicit LowLatencyStreamWait(const DeviceContext& ctx);
    ~LowLatencyStreamWait();

    LowLatencyStreamWait(const LowLatencyStreamWait&)            = delete;
    LowLatencyStreamWait& operator=(const LowLatencyStreamWait&) = delete;

    // Marks the start of the step's device work on `stream`; call before enqueueing it.
    void begin(cudaStream_t stream);
    // Records the end of host preparation. No-op unless round stats are enabled.
    void mark_launch(cudaStream_t stream);
    // Waits for all work enqueued on `stream`; throws on a device error.
    void wait(cudaStream_t stream);

private:
    enum class Mode { Spin, Sleep, Block, Stock };
    cudaEvent_t start_          = nullptr;
    cudaEvent_t launch_         = nullptr;
    cudaEvent_t stop_           = nullptr;
    double expected_us_         = 0.0;
    bool started_               = false;
    bool launch_marked_         = false;
    int stats_every_            = 0;
    int stats_i_                = 0;
    float* host_prep_us_        = nullptr;
    float* device_us_           = nullptr;
    float* wake_us_             = nullptr;
    Mode mode_                  = Mode::Sleep;
    int wake_margin_us_         = 250;
    std::int64_t begin_host_ns_ = 0;
};

} // namespace ninfer
