#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>

namespace vkp {

// Everything that can be set from the command line, with its default.
struct Options {
    std::uint32_t width = 1024;
    std::uint32_t height = 1024;
    std::uint32_t frames = 30;
    std::uint32_t seed = 1;
    std::uint32_t maxImages = 8;
    int annulusOuter = 7;
    int annulusInner = 4;
    float detectSigma = 6.0f;
    float shapeRatio = 0.35f;
    bool writeImages = true;
    bool runBenchmark = true;
    bool validation = false;
    bool view = false;
    std::uint32_t viewFps = 8;
    std::filesystem::path outputDir = "output";
    bool showHelp = false;
};

// Parses the arguments that follow the program name and checks their ranges.
// Returns the options, or a message describing the first problem found.
[[nodiscard]] std::expected<Options, std::string> parseOptions(
    std::span<const char* const> arguments);

void printUsage();

} // namespace vkp
