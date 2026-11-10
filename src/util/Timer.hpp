#pragma once

#include <chrono>

namespace vkp {

// Wall clock stopwatch over steady_clock. It starts running when it is constructed.
class Timer {
public:
    [[nodiscard]] double elapsedSeconds() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }

private:
    using Clock = std::chrono::steady_clock;

    Clock::time_point start_ = Clock::now();
};

} // namespace vkp
