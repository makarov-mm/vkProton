#include "Options.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <format>
#include <print>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

#include "../pipeline/FrameSource.hpp"

namespace vkp {

namespace {

constexpr std::uint32_t kMaxFrameSide = 16384;
// The noise stage visits every pixel of the annulus bounding square twice, so its
// cost grows with the square of the radius. This keeps a single dispatch short.
constexpr int kMaxAnnulusRadius = 32;

constexpr std::string_view kUsage =
    "vkPhoton - Vulkan compute pipeline for streaming transient detection\n\n"
    "Options:\n"
    "  --width N          frame width in pixels (default 1024)\n"
    "  --height N         frame height in pixels (default 1024)\n"
    "  --frames N         number of frames to stream (default 30)\n"
    "  --seed N           random seed for the synthetic sky (default 1)\n"
    "  --detect-sigma F   detection threshold in local noise units (default 6.0)\n"
    "  --annulus-outer N  outer radius of the noise annulus (default 7)\n"
    "  --annulus-inner N  inner radius excluded from the annulus (default 4)\n"
    "  --shape-ratio F    minimum neighbour to centre ratio, 0 disables (default 0.35)\n"
    "  --max-images N     cap on annotated frames written (default 8)\n"
    "  --output DIR       output directory, created if missing (default output)\n"
    "  --view             open a live window and stream the pipeline into it\n"
    "  --view-fps N       target frames per second in the viewer (default 8)\n"
    "  --no-images        skip writing annotated frames\n"
    "  --no-benchmark     skip the CPU baseline comparison\n"
    "  --validation       enable the Vulkan validation layer\n"
    "  --help             show this message\n";

using Status = std::expected<void, std::string>;

// Converts the whole of text to a number. Anything left over is an error.
template <typename T>
    requires std::is_arithmetic_v<T>
Status parseValue(std::string_view option, std::string_view text, T& target) {
    const char* const end = text.data() + text.size();
    const auto [parsed, error] = std::from_chars(text.data(), end, target);
    if (error != std::errc{} || parsed != end) {
        return std::unexpected(std::format("invalid value '{}' for {}", text, option));
    }
    return {};
}

Status parseValue(std::string_view /*option*/, std::string_view text,
                  std::filesystem::path& target) {
    target = text;
    return {};
}

std::expected<Options, std::string> validated(Options options) {
    const auto inRange = [](auto value, auto low, auto high) {
        return value >= low && value <= high;
    };

    if (!inRange(options.width, FrameSource::kMinSide, kMaxFrameSide) ||
        !inRange(options.height, FrameSource::kMinSide, kMaxFrameSide)) {
        return std::unexpected(std::format("--width and --height must be between {} and {}",
                                           FrameSource::kMinSide, kMaxFrameSide));
    }
    if (options.frames == 0) {
        return std::unexpected("--frames must be at least 1");
    }
    if (!std::isfinite(options.detectSigma) || options.detectSigma <= 0.0f) {
        return std::unexpected("--detect-sigma must be a positive number");
    }
    if (!std::isfinite(options.shapeRatio) || options.shapeRatio < 0.0f) {
        return std::unexpected("--shape-ratio must be zero or a positive number");
    }
    if (!inRange(options.annulusOuter, 1, kMaxAnnulusRadius)) {
        return std::unexpected(
            std::format("--annulus-outer must be between 1 and {}", kMaxAnnulusRadius));
    }
    if (!inRange(options.annulusInner, 0, options.annulusOuter - 1)) {
        return std::unexpected("--annulus-inner must be at least 0 and below --annulus-outer");
    }
    return options;
}

} // namespace

std::expected<Options, std::string> parseOptions(std::span<const char* const> arguments) {
    Options options;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const std::string_view name = arguments[i];
        if (name == "--help") {
            options.showHelp = true;
            return options;
        }

        // Reads the value that follows the current option into the given field.
        const auto value = [&](auto& field) -> Status {
            if (i + 1 >= arguments.size()) {
                return std::unexpected(std::format("{} needs a value", name));
            }
            return parseValue(name, arguments[++i], field);
        };

        Status status;
        if (name == "--width") status = value(options.width);
        else if (name == "--height") status = value(options.height);
        else if (name == "--frames") status = value(options.frames);
        else if (name == "--seed") status = value(options.seed);
        else if (name == "--detect-sigma") status = value(options.detectSigma);
        else if (name == "--annulus-outer") status = value(options.annulusOuter);
        else if (name == "--annulus-inner") status = value(options.annulusInner);
        else if (name == "--shape-ratio") status = value(options.shapeRatio);
        else if (name == "--max-images") status = value(options.maxImages);
        else if (name == "--output") status = value(options.outputDir);
        else if (name == "--view-fps") status = value(options.viewFps);
        else if (name == "--view") options.view = true;
        else if (name == "--no-images") options.writeImages = false;
        else if (name == "--no-benchmark") options.runBenchmark = false;
        else if (name == "--validation") options.validation = true;
        else status = std::unexpected(std::format("unknown argument: {}", name));

        if (!status) {
            return std::unexpected(std::move(status).error());
        }
    }
    return validated(std::move(options));
}

void printUsage() {
    std::print(stdout, "{}", kUsage);
}

} // namespace vkp
