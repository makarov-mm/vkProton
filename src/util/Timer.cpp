#include "Timer.hpp"

namespace vkp {

Timer::Timer() {
    reset();
}

void Timer::reset() {
    start_ = std::chrono::steady_clock::now();
}

double Timer::elapsedSeconds() const {
    return elapsedMillis() / 1000.0;
}

double Timer::elapsedMillis() const {
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> delta = now - start_;
    return delta.count();
}

} // namespace vkp
