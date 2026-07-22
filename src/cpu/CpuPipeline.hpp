#pragma once

#include <vector>

#include "../pipeline/Detection.hpp"
#include "../pipeline/TransientPipeline.hpp"

namespace vkp {

// A straightforward single threaded version of the same four stage pipeline.
// It exists as a reference: to sanity check the GPU output and to give the
// benchmark an honest CPU baseline. It is deliberately not vectorised or
// threaded, so it represents the naive starting point a GPU port improves on.
class CpuPipeline {
public:
    explicit CpuPipeline(const PipelineConfig& config);

    void uploadCalibration(const std::vector<float>& bias, const std::vector<float>& flatInv);
    FrameResult processFrame(const std::vector<float>& raw);

    const std::vector<float>& lastCalibrated() const { return calibrated_; }
    void resetHistory() { frameCounter_ = 0; }

private:
    PipelineConfig config_;
    size_t pixelCount_ = 0;
    uint32_t frameCounter_ = 0;

    std::vector<float> bias_;
    std::vector<float> flatInv_;
    std::vector<float> calibrated_;
    std::vector<float> prevCalibrated_;
    std::vector<float> diff_;
    std::vector<float> diffMean_;
    std::vector<float> diffStd_;
};

} // namespace vkp
