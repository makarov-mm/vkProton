#pragma once

#include <cstdio>
#include <format>
#include <print>
#include <utility>

namespace vkp {

// Prints one line to stdout with the program prefix.
template <typename... Args>
void logInfo(std::format_string<Args...> format, Args&&... args) {
    std::println(stdout, "[vkPhoton] {}", std::format(format, std::forward<Args>(args)...));
}

// Prints one line to stderr. stdout is flushed first so the error lands after the
// output that led up to it, even when both streams are redirected to the same file.
template <typename... Args>
void logError(std::format_string<Args...> format, Args&&... args) {
    std::fflush(stdout);
    std::println(stderr, "[vkPhoton] error: {}", std::format(format, std::forward<Args>(args)...));
}

} // namespace vkp
