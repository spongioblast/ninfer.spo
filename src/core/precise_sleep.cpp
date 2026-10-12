#include "core/precise_sleep.h"

#include <algorithm>
#include <chrono>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <timeapi.h>
#endif

namespace ninfer {

void precise_sleep_ns(std::int64_t ns) {
    if (ns <= 0) { return; }
#ifdef _WIN32
    struct HighResolutionTimer {
        HANDLE handle = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                               TIMER_ALL_ACCESS);
        ~HighResolutionTimer() {
            if (handle != nullptr) { CloseHandle(handle); }
        }
    };
    thread_local HighResolutionTimer timer;
    if (timer.handle == nullptr) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
        return;
    }
    LARGE_INTEGER due{};
    due.QuadPart = -std::max<std::int64_t>(1, ns / 100);
    if (!SetWaitableTimer(timer.handle, &due, 0, nullptr, nullptr, FALSE)) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
        return;
    }
    WaitForSingleObject(timer.handle, INFINITE);
#else
    std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
#endif
}

ProcessTimerResolution::ProcessTimerResolution(unsigned period_ms) noexcept {
#ifdef _WIN32
    if (period_ms == 0) { return; }
    if (timeBeginPeriod(period_ms) == TIMERR_NOERROR) {
        period_ms_ = period_ms;
        active_    = true;
    }
#else
    (void)period_ms;
#endif
}

ProcessTimerResolution::~ProcessTimerResolution() {
#ifdef _WIN32
    if (active_) { timeEndPeriod(period_ms_); }
#endif
}

} // namespace ninfer
