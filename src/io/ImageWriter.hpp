#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../pipeline/Detection.hpp"

namespace vkp {

// Turns a calibrated frame into a displayable image and writes it to disk.
// Astronomical frames have a huge dynamic range, so a percentile anchored asinh
// stretch is applied before mapping to 8 bit, otherwise everything but the
// brightest stars looks black. Detections are drawn as red crosses so transients
// are easy to spot. The RGBA builder is shared by the file writer and the live
// viewer so both show exactly the same picture.
class ImageWriter {
public:
    // Builds a tightly packed RGBA8 image, four bytes per pixel in R, G, B, A order.
    static std::vector<uint8_t> buildRgba(const std::vector<float>& calibrated,
                                          uint32_t width,
                                          uint32_t height,
                                          const std::vector<Detection>& detections);

    // Writes the same image as a binary PPM, dropping the alpha channel.
    static bool writeAnnotatedPpm(const std::string& path,
                                  const std::vector<float>& calibrated,
                                  uint32_t width,
                                  uint32_t height,
                                  const std::vector<Detection>& detections);
};

} // namespace vkp
