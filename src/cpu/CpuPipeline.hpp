#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "../pipeline/PipelineConfig.hpp"

namespace vkp {

// A straightforward single threaded version of the same four stage pipeline.
// It exists as a reference: to sanity check the GPU output and to give the
// benchmark an honest CPU baseline. It is deliberately not vectorised or
// threaded, so it represents the naive starting point a GPU port improves on.
// Each stage is one method that mirrors the shader of the same name.
class CpuPipeline {
public:
    explicit CpuPipeline(const PipelineConfig& config);

    void uploadCalibration(std::span<const float> bias, std::span<const float> flatInv);
    FrameResult processFrame(std::span<const float> raw);

private:
    struct RingStats {
        float mean;
        float sigma;
    };

    void calibrate(std::span<const float> raw, std::span<float> calibrated) const;
    void difference(std::span<const float> calibrated, std::span<const float> previous);
    void background();
    [[nodiscard]] FrameResult detect() const;

    [[nodiscard]] RingStats ringStats(int x, int y, bool clip, RingStats estimate) const;
    [[nodiscard]] bool isLocalMaximum(int x, int y, float center) const;
    [[nodiscard]] float diffAt(int x, int y) const;
    [[nodiscard]] std::size_t pixelIndex(int x, int y) const noexcept;

    PipelineConfig config_;
    int width_;
    int height_;
    std::size_t pixelCount_;
    std::uint32_t frameCounter_ = 0;

    std::vector<float> bias_;
    std::vector<float> flatInv_;
    // Ping pong pair, as on the GPU, so keeping the previous frame costs no copy.
    std::array<std::vector<float>, 2> calibrated_;
    std::vector<float> diff_;
    std::vector<float> diffMean_;
    std::vector<float> diffStd_;
};

} // namespace vkp
