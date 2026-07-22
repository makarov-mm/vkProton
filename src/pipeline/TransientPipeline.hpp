#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "Detection.hpp"

namespace vkp {

class VulkanContext;
class Buffer;
class ComputeProgram;

struct PipelineConfig {
    uint32_t width = 0;
    uint32_t height = 0;
    int annulusOuter = 7;       // outer radius of the noise annulus in pixels
    int annulusInner = 4;       // inner radius excluded from the annulus
    float detectSigma = 6.0f;   // detection threshold in units of local difference noise
    float shapeRatio = 0.35f;   // minimum neighbour to centre ratio in the difference
    uint32_t maxDetections = 8192;
};

struct FrameResult {
    std::vector<Detection> detections; // capped at config.maxDetections
    uint32_t reportedCount = 0;        // total count from the atomic counter
};

// Runs the four stage transient detection pipeline entirely on the GPU. It owns
// every buffer and compute program, uploads calibration once, then processes
// frames one at a time. Difference imaging uses a ping pong pair of calibrated
// buffers so the previous frame is always available without an extra copy.
class TransientPipeline {
public:
    TransientPipeline(const VulkanContext& context, const PipelineConfig& config);
    ~TransientPipeline();

    TransientPipeline(const TransientPipeline&) = delete;
    TransientPipeline& operator=(const TransientPipeline&) = delete;

    void uploadCalibration(const std::vector<float>& bias, const std::vector<float>& flatInv);

    // Uploads a raw frame, runs all stages and reads the detections back.
    FrameResult processFrame(const std::vector<float>& raw);

    // The calibrated pixels of the most recent frame, kept for visualisation.
    const std::vector<float>& lastCalibrated() const { return calibratedHost_; }

    void resetHistory() { frameCounter_ = 0; }

private:
    void createBuffers();
    void createPrograms();

    const VulkanContext& context_;
    PipelineConfig config_;
    size_t pixelCount_ = 0;
    uint32_t frameCounter_ = 0;

    std::unique_ptr<Buffer> stagingUpload_;
    std::unique_ptr<Buffer> readbackCalibrated_;
    std::unique_ptr<Buffer> readbackCounter_;
    std::unique_ptr<Buffer> readbackDetections_;

    std::unique_ptr<Buffer> rawBuffer_;
    std::unique_ptr<Buffer> biasBuffer_;
    std::unique_ptr<Buffer> flatInvBuffer_;
    std::unique_ptr<Buffer> calibrated_[2];
    std::unique_ptr<Buffer> diffBuffer_;
    std::unique_ptr<Buffer> diffMeanBuffer_;
    std::unique_ptr<Buffer> diffStdBuffer_;
    std::unique_ptr<Buffer> counterBuffer_;
    std::unique_ptr<Buffer> detectionsBuffer_;

    std::unique_ptr<ComputeProgram> calibrateProgram_;
    std::unique_ptr<ComputeProgram> backgroundProgram_;
    std::unique_ptr<ComputeProgram> differenceProgram_;
    std::unique_ptr<ComputeProgram> detectProgram_;

    std::vector<float> calibratedHost_;
};

} // namespace vkp
