#pragma once

#include <chrono>

namespace vkp {

// Thin wrapper over steady_clock for wall clock measurement of pipeline runs.
class Timer {
public:
    Timer();

    void reset();
    double elapsedSeconds() const;
    double elapsedMillis() const;

private:
    std::chrono::steady_clock::time_point start_;
};

} // namespace vkp
