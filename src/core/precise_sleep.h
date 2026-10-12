#pragma once

#include <cstdint>

namespace ninfer {

// Sleep for at least `ns`. On Windows this uses a high-resolution waitable timer.
// std::this_thread::sleep_for rounds up to whole milliseconds and is not used here.
void precise_sleep_ns(std::int64_t ns);

// Asks Windows for a 1 ms timer tick for this object's lifetime. Condition-variable timeouts
// otherwise wake on the default 15.6 ms tick. Other platforms do nothing.
class ProcessTimerResolution {
public:
    explicit ProcessTimerResolution(unsigned period_ms = 1) noexcept;
    ~ProcessTimerResolution();

    ProcessTimerResolution(const ProcessTimerResolution&)            = delete;
    ProcessTimerResolution& operator=(const ProcessTimerResolution&) = delete;

    [[nodiscard]] bool active() const noexcept { return active_; }

private:
    unsigned period_ms_ = 0;
    bool active_        = false;
};

} // namespace ninfer
