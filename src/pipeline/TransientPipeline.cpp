#include "TransientPipeline.hpp"

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "PushConstants.hpp"
#include "../core/Buffer.hpp"
#include "../core/ComputeProgram.hpp"
#include "../core/VulkanContext.hpp"
#include "../util/Log.hpp"
#include "../util/Paths.hpp"

#ifndef VKPHOTON_SHADER_DIR
#define VKPHOTON_SHADER_DIR "shaders"
#endif

namespace vkp {

namespace {
constexpr uint32_t kLocalSize = 16;

uint32_t groupCount(uint32_t extent) {
    return (extent + kLocalSize - 1) / kLocalSize;
}

// Records a global memory barrier between two pipeline stages. Using a single
// full barrier rather than per buffer barriers keeps the orchestration readable;
// the buffers are small enough that the coarser sync costs nothing measurable.
void memoryBarrier(VkCommandBuffer cmd,
                   VkPipelineStageFlags srcStage,
                   VkPipelineStageFlags dstStage,
                   VkAccessFlags srcAccess,
                   VkAccessFlags dstAccess) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void copyBuffer(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
}

bool directoryHasShaders(const std::string& dir) {
    std::ifstream probe(dir + "/calibrate.comp.spv", std::ios::binary);
    return probe.good();
}

// Finds the directory holding the compiled shaders. It checks the path baked in
// at build time first, then the folder next to the executable, then a couple of
// relative locations. This keeps both the CMake build and the Visual Studio build
// working no matter which directory the program is launched from.
std::string resolveShaderDir() {
    std::vector<std::string> candidates;
#ifdef VKPHOTON_SHADER_DIR
    candidates.push_back(VKPHOTON_SHADER_DIR);
#endif
    std::string exeDir = executableDirectory();
    if (!exeDir.empty()) {
        candidates.push_back(exeDir + "/shaders");
    }
    candidates.push_back("shaders");
    candidates.push_back("../shaders");

    for (const std::string& candidate : candidates) {
        if (directoryHasShaders(candidate)) {
            return candidate;
        }
    }
#ifdef VKPHOTON_SHADER_DIR
    return VKPHOTON_SHADER_DIR;
#else
    return "shaders";
#endif
}
} // namespace

TransientPipeline::TransientPipeline(const VulkanContext& context, const PipelineConfig& config)
    : context_(context), config_(config) {
    pixelCount_ = static_cast<size_t>(config_.width) * config_.height;
    calibratedHost_.resize(pixelCount_);
    createBuffers();
    createPrograms();
}

TransientPipeline::~TransientPipeline() = default;

void TransientPipeline::createBuffers() {
    const VkDeviceSize floatBytes = static_cast<VkDeviceSize>(pixelCount_) * sizeof(float);
    const VkDeviceSize detectionBytes =
        static_cast<VkDeviceSize>(config_.maxDetections) * sizeof(Detection);

    const VkBufferUsageFlags computeUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkMemoryPropertyFlags deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const VkMemoryPropertyFlags hostVisible =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    stagingUpload_ = std::make_unique<Buffer>(context_, floatBytes,
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT, hostVisible);
    readbackCalibrated_ = std::make_unique<Buffer>(context_, floatBytes,
                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostVisible);
    readbackCounter_ = std::make_unique<Buffer>(context_, sizeof(uint32_t),
                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostVisible);
    readbackDetections_ = std::make_unique<Buffer>(context_, detectionBytes,
                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostVisible);

    rawBuffer_ = std::make_unique<Buffer>(context_, floatBytes, computeUsage, deviceLocal);
    biasBuffer_ = std::make_unique<Buffer>(context_, floatBytes, computeUsage, deviceLocal);
    flatInvBuffer_ = std::make_unique<Buffer>(context_, floatBytes, computeUsage, deviceLocal);
    calibrated_[0] = std::make_unique<Buffer>(context_, floatBytes, computeUsage, deviceLocal);
    calibrated_[1] = std::make_unique<Buffer>(context_, floatBytes, computeUsage, deviceLocal);
    diffBuffer_ = std::make_unique<Buffer>(context_, floatBytes, computeUsage, deviceLocal);
    diffMeanBuffer_ = std::make_unique<Buffer>(context_, floatBytes, computeUsage, deviceLocal);
    diffStdBuffer_ = std::make_unique<Buffer>(context_, floatBytes, computeUsage, deviceLocal);
    counterBuffer_ = std::make_unique<Buffer>(context_, sizeof(uint32_t), computeUsage, deviceLocal);
    detectionsBuffer_ = std::make_unique<Buffer>(context_, detectionBytes, computeUsage, deviceLocal);
}

void TransientPipeline::createPrograms() {
    const std::string dir = resolveShaderDir() + "/";
    const uint32_t pushSize = static_cast<uint32_t>(sizeof(PushConstants));

    calibrateProgram_ = std::make_unique<ComputeProgram>(context_, dir + "calibrate.comp.spv", 4, pushSize);
    differenceProgram_ = std::make_unique<ComputeProgram>(context_, dir + "difference.comp.spv", 3, pushSize);
    backgroundProgram_ = std::make_unique<ComputeProgram>(context_, dir + "background.comp.spv", 3, pushSize);
    detectProgram_ = std::make_unique<ComputeProgram>(context_, dir + "detect.comp.spv", 5, pushSize);
}

void TransientPipeline::uploadCalibration(const std::vector<float>& bias,
                                          const std::vector<float>& flatInv) {
    const VkDeviceSize floatBytes = static_cast<VkDeviceSize>(pixelCount_) * sizeof(float);

    stagingUpload_->upload(bias.data(), floatBytes);
    context_.runCommands([&](VkCommandBuffer cmd) {
        copyBuffer(cmd, stagingUpload_->handle(), biasBuffer_->handle(), floatBytes);
    });

    stagingUpload_->upload(flatInv.data(), floatBytes);
    context_.runCommands([&](VkCommandBuffer cmd) {
        copyBuffer(cmd, stagingUpload_->handle(), flatInvBuffer_->handle(), floatBytes);
    });
}

FrameResult TransientPipeline::processFrame(const std::vector<float>& raw) {
    const VkDeviceSize floatBytes = static_cast<VkDeviceSize>(pixelCount_) * sizeof(float);
    const VkDeviceSize detectionBytes =
        static_cast<VkDeviceSize>(config_.maxDetections) * sizeof(Detection);

    const uint32_t current = frameCounter_ % 2;
    const uint32_t previous = 1u - current;
    const bool hasPrev = frameCounter_ > 0;

    // Point each stage at the right buffers for this frame. The calibrated output
    // alternates between the two ping pong buffers so the previous frame survives.
    // Noise is measured on the difference image, so the background stage reads
    // diff and the stages run calibrate, difference, background, detect in order.
    calibrateProgram_->bindBuffers({rawBuffer_.get(), biasBuffer_.get(),
                                    flatInvBuffer_.get(), calibrated_[current].get()});
    differenceProgram_->bindBuffers({calibrated_[current].get(), calibrated_[previous].get(),
                                     diffBuffer_.get()});
    backgroundProgram_->bindBuffers({diffBuffer_.get(), diffMeanBuffer_.get(),
                                     diffStdBuffer_.get()});
    detectProgram_->bindBuffers({diffBuffer_.get(), diffMeanBuffer_.get(),
                                 diffStdBuffer_.get(), counterBuffer_.get(),
                                 detectionsBuffer_.get()});

    PushConstants pc{};
    pc.width = config_.width;
    pc.height = config_.height;
    pc.annulusOuter = config_.annulusOuter;
    pc.annulusInner = config_.annulusInner;
    pc.detectSigma = config_.detectSigma;
    pc.shapeRatio = config_.shapeRatio;
    pc.maxDetections = config_.maxDetections;
    pc.hasPrev = hasPrev ? 1u : 0u;

    stagingUpload_->upload(raw.data(), floatBytes);

    const uint32_t gx = groupCount(config_.width);
    const uint32_t gy = groupCount(config_.height);

    context_.runCommands([&](VkCommandBuffer cmd) {
        // Upload raw frame and clear the detection counter.
        copyBuffer(cmd, stagingUpload_->handle(), rawBuffer_->handle(), floatBytes);
        vkCmdFillBuffer(cmd, counterBuffer_->handle(), 0, sizeof(uint32_t), 0);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        calibrateProgram_->recordDispatch(cmd, gx, gy, &pc);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        differenceProgram_->recordDispatch(cmd, gx, gy, &pc);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        backgroundProgram_->recordDispatch(cmd, gx, gy, &pc);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        detectProgram_->recordDispatch(cmd, gx, gy, &pc);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

        // Pull back the calibrated frame, the counter and the detection list.
        copyBuffer(cmd, calibrated_[current]->handle(), readbackCalibrated_->handle(), floatBytes);
        copyBuffer(cmd, counterBuffer_->handle(), readbackCounter_->handle(), sizeof(uint32_t));
        copyBuffer(cmd, detectionsBuffer_->handle(), readbackDetections_->handle(), detectionBytes);
    });

    // Copy results out of the host visible readback buffers.
    readbackCalibrated_->download(calibratedHost_.data(), floatBytes);

    uint32_t reported = 0;
    readbackCounter_->download(&reported, sizeof(uint32_t));

    FrameResult result;
    result.reportedCount = reported;
    const uint32_t stored = reported < config_.maxDetections ? reported : config_.maxDetections;
    result.detections.resize(stored);
    if (stored > 0) {
        readbackDetections_->download(result.detections.data(),
                                      static_cast<VkDeviceSize>(stored) * sizeof(Detection));
    }
    (void)detectionBytes;

    ++frameCounter_;
    return result;
}

} // namespace vkp
