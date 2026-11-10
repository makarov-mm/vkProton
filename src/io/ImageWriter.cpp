#include "ImageWriter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <ostream>
#include <print>
#include <stdexcept>

namespace vkp {

namespace {

std::uint8_t toByte(float value) {
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 255.0f)));
}

struct Stretch {
    float low;
    float span;
};

// Anchors the stretch at the 5th and 99.5th percentile of the frame. The low
// anchor removes the sky, the high anchor keeps bright stars from washing out the
// mid tones.
Stretch percentileStretch(std::span<const float> data) {
    // Selection reorders its input, so both anchors come from one scratch copy.
    // Selecting the high one first leaves every smaller value in front of it, and
    // the low one is then picked from that part alone.
    std::vector<float> scratch(data.begin(), data.end());
    const auto position = [&](float fraction) {
        const auto last = static_cast<float>(scratch.size() - 1);
        return scratch.begin() + static_cast<std::ptrdiff_t>(fraction * last);
    };
    const auto high = position(0.995f);
    const auto low = position(0.05f);
    std::nth_element(scratch.begin(), high, scratch.end());
    std::nth_element(scratch.begin(), low, high);
    return {.low = *low, .span = std::max(*high - *low, 1e-3f)};
}

// Marks a detection with four short arms that leave the source itself visible.
void drawCross(std::span<std::uint8_t> pixels, std::uint32_t width, std::uint32_t height,
               std::size_t channels, std::uint32_t cx, std::uint32_t cy) {
    constexpr int arm = 4;
    constexpr int gap = 2;
    constexpr std::array<std::uint8_t, 3> red{255, 40, 40};

    const auto plot = [&](int x, int y) {
        if (x < 0 || x >= static_cast<int>(width) || y < 0 || y >= static_cast<int>(height)) {
            return;
        }
        const std::size_t pixel = static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x);
        std::ranges::copy(red, pixels.begin() + static_cast<std::ptrdiff_t>(pixel * channels));
    };

    const int x = static_cast<int>(cx);
    const int y = static_cast<int>(cy);
    for (int d = gap; d <= arm; ++d) {
        plot(x + d, y);
        plot(x - d, y);
        plot(x, y + d);
        plot(x, y - d);
    }
}

// Renders the annotated frame with 3 bytes per pixel for RGB or 4 for RGBA.
std::vector<std::uint8_t> render(std::span<const float> calibrated,
                                 std::uint32_t width,
                                 std::uint32_t height,
                                 std::span<const Detection> detections,
                                 std::size_t channels) {
    const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
    if (pixelCount == 0 || calibrated.size() != pixelCount) {
        throw std::invalid_argument("calibrated frame does not match the image size");
    }

    const Stretch stretch = percentileStretch(calibrated);
    const float asinhRange = std::asinh(8.0f);

    // Every byte starts at 255, which is the final value of the alpha channel.
    std::vector<std::uint8_t> pixels(pixelCount * channels, 255);
    for (std::size_t i = 0; i < pixelCount; ++i) {
        // asinh stretch gives faint sources a fair share of the tonal range.
        const float norm = (calibrated[i] - stretch.low) / stretch.span;
        const std::uint8_t grey = toByte(std::asinh(norm * 8.0f) / asinhRange * 255.0f);
        std::fill_n(pixels.begin() + static_cast<std::ptrdiff_t>(i * channels), 3, grey);
    }

    for (const Detection& d : detections) {
        drawCross(pixels, width, height, channels, d.x, d.y);
    }
    return pixels;
}

} // namespace

std::vector<std::uint8_t> buildRgba(std::span<const float> calibrated,
                                    std::uint32_t width,
                                    std::uint32_t height,
                                    std::span<const Detection> detections) {
    return render(calibrated, width, height, detections, 4);
}

void writeAnnotatedPpm(const std::filesystem::path& path,
                       std::span<const float> calibrated,
                       std::uint32_t width,
                       std::uint32_t height,
                       std::span<const Detection> detections) {
    const std::vector<std::uint8_t> rgb = render(calibrated, width, height, detections, 3);

    std::ofstream file(path, std::ios::binary);
    std::print(file, "P6\n{} {}\n255\n", width, height);
    file.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    file.close();
    if (!file) {
        throw std::runtime_error(std::format("cannot write image {}", path.string()));
    }
}

} // namespace vkp
