#include "core/precise_sleep.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

int main() {
    using clock_type = std::chrono::steady_clock;
    std::vector<double> overshoot_us;
    overshoot_us.reserve(200);
    for (int i = 0; i < 200; ++i) {
        constexpr auto requested = std::chrono::microseconds(500);
        const auto start         = clock_type::now();
        ninfer::precise_sleep_ns(
            std::chrono::duration_cast<std::chrono::nanoseconds>(requested).count());
        const auto elapsed = std::chrono::duration<double, std::micro>(clock_type::now() - start);
        overshoot_us.push_back(elapsed.count() - 500.0);
    }
    std::sort(overshoot_us.begin(), overshoot_us.end());
    const double p99 = overshoot_us[static_cast<std::size_t>(0.99 * (overshoot_us.size() - 1))];
    std::cout << "precise_sleep 500us p99 overshoot " << p99 << " us\n";
#ifdef _WIN32
    if (p99 > 2500.0) {
        std::cerr << "precise sleep p99 overshoot exceeds 2.5 ms\n";
        return 1;
    }
#endif
    return 0;
}
