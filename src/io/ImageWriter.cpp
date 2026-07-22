#include "ImageWriter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>

namespace vkp {

namespace {
uint8_t toByte(float value) {
    float clamped = std::clamp(value, 0.0f, 255.0f);
    return static_cast<uint8_t>(std::lround(clamped));
}

// Returns the value at the given percentile of a copy of the data.
float percentile(const std::vector<float>& data, float fraction) {
    if (data.empty()) {
        return 0.0f;
    }
    std::vector<float> sorted = data;
    size_t index = static_cast<size_t>(fraction * static_cast<float>(sorted.size() - 1));
    std::nth_element(sorted.begin(), sorted.begin() + index, sorted.end());
    return sorted[index];
}

void drawCross(std::vector<uint8_t>& rgba, uint32_t width, uint32_t height,
               uint32_t cx, uint32_t cy) {
    const int arm = 4;
    const int gap = 2;
    for (int d = gap; d <= arm; ++d) {
        const int offsets[4][2] = {{d, 0}, {-d, 0}, {0, d}, {0, -d}};
        for (const auto& o : offsets) {
            int x = static_cast<int>(cx) + o[0];
            int y = static_cast<int>(cy) + o[1];
            if (x < 0 || x >= static_cast<int>(width) || y < 0 || y >= static_cast<int>(height)) {
                continue;
            }
            size_t p = (static_cast<size_t>(y) * width + x) * 4;
            rgba[p + 0] = 255;
            rgba[p + 1] = 40;
            rgba[p + 2] = 40;
            rgba[p + 3] = 255;
        }
    }
}
} // namespace

std::vector<uint8_t> ImageWriter::buildRgba(const std::vector<float>& calibrated,
                                            uint32_t width,
                                            uint32_t height,
                                            const std::vector<Detection>& detections) {
    const size_t pixelCount = static_cast<size_t>(width) * height;
    std::vector<uint8_t> rgba(pixelCount * 4, 0);
    if (calibrated.size() != pixelCount) {
        return rgba;
    }

    // Percentile stretch. The low anchor removes the sky, the high anchor keeps
    // bright stars from washing out the mid tones.
    float low = percentile(calibrated, 0.05f);
    float high = percentile(calibrated, 0.995f);
    float span = std::max(high - low, 1e-3f);

    for (size_t i = 0; i < pixelCount; ++i) {
        // asinh stretch gives faint sources a fair share of the tonal range.
        float norm = (calibrated[i] - low) / span;
        float stretched = std::asinh(norm * 8.0f) / std::asinh(8.0f);
        uint8_t g = toByte(stretched * 255.0f);
        rgba[i * 4 + 0] = g;
        rgba[i * 4 + 1] = g;
        rgba[i * 4 + 2] = g;
        rgba[i * 4 + 3] = 255;
    }

    for (const Detection& d : detections) {
        drawCross(rgba, width, height, d.x, d.y);
    }
    return rgba;
}

bool ImageWriter::writeAnnotatedPpm(const std::string& path,
                                    const std::vector<float>& calibrated,
                                    uint32_t width,
                                    uint32_t height,
                                    const std::vector<Detection>& detections) {
    const size_t pixelCount = static_cast<size_t>(width) * height;
    if (calibrated.size() != pixelCount) {
        return false;
    }

    std::vector<uint8_t> rgba = buildRgba(calibrated, width, height, detections);

    std::vector<uint8_t> rgb(pixelCount * 3);
    for (size_t i = 0; i < pixelCount; ++i) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(rgb.data()),
               static_cast<std::streamsize>(rgb.size()));
    return file.good();
}

} // namespace vkp
