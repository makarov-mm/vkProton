#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "../pipeline/Detection.hpp"

namespace vkp {

// Turns a calibrated frame into a displayable image. Astronomical frames have a
// huge dynamic range, so a percentile anchored asinh stretch is applied before
// mapping to 8 bit, otherwise everything but the brightest stars looks black.
// Detections are drawn as red crosses so transients are easy to spot. The file
// writer and the live viewer share one renderer, so both show exactly the same
// picture.

// Builds a tightly packed RGBA8 image, four bytes per pixel in R, G, B, A order.
[[nodiscard]] std::vector<std::uint8_t> buildRgba(std::span<const float> calibrated,
                                                  std::uint32_t width,
                                                  std::uint32_t height,
                                                  std::span<const Detection> detections);

// Writes the same image as a binary PPM, without the alpha channel. Throws if the
// file cannot be written.
void writeAnnotatedPpm(const std::filesystem::path& path,
                       std::span<const float> calibrated,
                       std::uint32_t width,
                       std::uint32_t height,
                       std::span<const Detection> detections);

} // namespace vkp
