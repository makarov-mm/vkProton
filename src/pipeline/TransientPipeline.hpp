#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <vulkan/vulkan.h>

#include "PipelineConfig.hpp"
#include "../core/Buffer.hpp"
#include "../core/ComputeProgram.hpp"

namespace vkp {

class VulkanContext;

// Runs the four stage transient detection pipeline entirely on the GPU. It owns
// every buffer and compute program, uploads calibration once, then processes
// frames one at a time. Difference imaging uses a ping pong pair of calibrated
// buffers so the previous frame is always available without an extra copy.
class TransientPipeline {
public:
    TransientPipeline(const VulkanContext& context, const PipelineConfig& config);

    void uploadCalibration(std::span<const float> bias, std::span<const float> flatInv);

    // Uploads a raw frame, runs all stages and reads the detections back. The
    // calibrated image is by far the largest transfer, so it is only read back
    // when the caller wants it.
    FrameResult processFrame(std::span<const float> raw, bool readCalibrated);

    // Forgets the previous frame, so the next one starts a fresh sequence.
    void resetHistory() noexcept { frameCounter_ = 0; }

private:
    void requireFrameSized(std::span<const float> data, const char* what) const;
    void bindStages();

    const VulkanContext& context_;
    PipelineConfig config_;
    std::size_t pixelCount_;
    VkDeviceSize frameBytes_;
    VkDeviceSize detectionBytes_;
    std::uint32_t frameCounter_ = 0;

    Buffer stagingUpload_;
    Buffer readbackCalibrated_;
    Buffer readbackCounter_;
    Buffer readbackDetections_;

    Buffer rawBuffer_;
    Buffer biasBuffer_;
    Buffer flatInvBuffer_;
    std::array<Buffer, 2> calibrated_;
    Buffer diffBuffer_;
    Buffer diffMeanBuffer_;
    Buffer diffStdBuffer_;
    Buffer counterBuffer_;
    Buffer detectionsBuffer_;

    ComputeProgram calibrateProgram_;
    ComputeProgram differenceProgram_;
    ComputeProgram backgroundProgram_;
    ComputeProgram detectProgram_;
};

} // namespace vkp
