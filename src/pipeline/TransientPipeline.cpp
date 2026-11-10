#include "TransientPipeline.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "PushConstants.hpp"
#include "../core/VulkanContext.hpp"
#include "../util/Paths.hpp"

namespace vkp {

namespace {

// Edge length of the square tiles the shaders work in. Must match common.glsl.
constexpr std::uint32_t kTileSize = 16;
constexpr auto kPushConstantSize = static_cast<std::uint32_t>(sizeof(PushConstants));

constexpr std::uint32_t tileCount(std::uint32_t extent) {
    return (extent + kTileSize - 1) / kTileSize;
}

// Rejects a configuration the shaders or the device cannot handle, before any
// memory is allocated for it.
PipelineConfig checked(const VulkanContext& context, const PipelineConfig& config) {
    if (config.width == 0 || config.height == 0 || config.maxDetections == 0) {
        throw std::invalid_argument("frame size and detection capacity must not be zero");
    }

    const VkPhysicalDeviceLimits& limits = context.limits();
    if (limits.maxComputeWorkGroupInvocations < kTileSize * kTileSize ||
        limits.maxComputeWorkGroupSize[0] < kTileSize ||
        limits.maxComputeWorkGroupSize[1] < kTileSize) {
        throw std::runtime_error("the device does not support 16x16 compute work groups");
    }
    if (tileCount(config.width) > limits.maxComputeWorkGroupCount[0] ||
        tileCount(config.height) > limits.maxComputeWorkGroupCount[1]) {
        throw std::runtime_error("the frame is too large to dispatch on this device");
    }

    // Every frame sized buffer is bound whole as one storage buffer, and the
    // shaders index pixels with signed 32 bit integers.
    const std::uint64_t pixelCount = std::uint64_t{config.width} * config.height;
    const std::uint64_t frameBytes = pixelCount * sizeof(float);
    if (frameBytes > limits.maxStorageBufferRange ||
        pixelCount > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error(std::format(
            "a {}x{} frame needs {} MiB per buffer, the device allows {} MiB", config.width,
            config.height, frameBytes >> 20, limits.maxStorageBufferRange >> 20));
    }
    return config;
}

// Finds a compiled shader. The directory baked in by the CMake build is checked
// first, then the folder next to the executable, where the Visual Studio build
// puts the shaders, then the working directory and its parent. That keeps both
// builds working no matter which directory the program is launched from.
std::filesystem::path shaderPath(const std::string& fileName) {
    std::vector<std::filesystem::path> directories;
#ifdef VKPHOTON_SHADER_DIR
    directories.emplace_back(VKPHOTON_SHADER_DIR);
#endif
    if (const std::filesystem::path exeDir = executableDirectory(); !exeDir.empty()) {
        directories.push_back(exeDir / "shaders");
    }
    directories.emplace_back("shaders");
    directories.emplace_back("../shaders");

    for (const std::filesystem::path& directory : directories) {
        const std::filesystem::path path = directory / fileName;
        if (std::error_code error; std::filesystem::exists(path, error)) {
            return path;
        }
    }
    throw std::runtime_error(std::format(
        "compiled shader {} not found next to the executable or in the working directory",
        fileName));
}

// One end of a memory barrier: a pipeline stage and what it does to memory.
struct Access {
    VkPipelineStageFlags stage;
    VkAccessFlags mask;
};

constexpr Access kTransferWrite{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
constexpr Access kTransferRead{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT};
constexpr Access kShaderWrite{VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT};
constexpr Access kShaderRead{VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT};
constexpr Access kShaderReadWrite{VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};

// Records a global memory barrier between two pipeline stages. Using a single
// full barrier rather than per buffer barriers keeps the orchestration readable;
// the buffers are small enough that the coarser sync costs nothing measurable.
void memoryBarrier(VkCommandBuffer commands, Access from, Access to) {
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = from.mask,
        .dstAccessMask = to.mask,
    };
    vkCmdPipelineBarrier(commands, from.stage, to.stage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void copyBuffer(VkCommandBuffer commands, const Buffer& source, const Buffer& target,
                VkDeviceSize size) {
    const VkBufferCopy region{.size = size};
    vkCmdCopyBuffer(commands, source.handle(), target.handle(), 1, &region);
}

} // namespace

TransientPipeline::TransientPipeline(const VulkanContext& context, const PipelineConfig& config)
    : context_(context),
      config_(checked(context, config)),
      pixelCount_(static_cast<std::size_t>(config_.width) * config_.height),
      frameBytes_(static_cast<VkDeviceSize>(pixelCount_) * sizeof(float)),
      detectionBytes_(static_cast<VkDeviceSize>(config_.maxDetections) * sizeof(Detection)),
      stagingUpload_(context, frameBytes_, Buffer::Kind::Upload),
      readbackCalibrated_(context, frameBytes_, Buffer::Kind::Readback),
      readbackCounter_(context, sizeof(std::uint32_t), Buffer::Kind::Readback),
      readbackDetections_(context, detectionBytes_, Buffer::Kind::Readback),
      rawBuffer_(context, frameBytes_, Buffer::Kind::Storage),
      biasBuffer_(context, frameBytes_, Buffer::Kind::Storage),
      flatInvBuffer_(context, frameBytes_, Buffer::Kind::Storage),
      calibrated_{Buffer(context, frameBytes_, Buffer::Kind::Storage),
                  Buffer(context, frameBytes_, Buffer::Kind::Storage)},
      diffBuffer_(context, frameBytes_, Buffer::Kind::Storage),
      diffMeanBuffer_(context, frameBytes_, Buffer::Kind::Storage),
      diffStdBuffer_(context, frameBytes_, Buffer::Kind::Storage),
      counterBuffer_(context, sizeof(std::uint32_t), Buffer::Kind::Storage),
      detectionsBuffer_(context, detectionBytes_, Buffer::Kind::Storage),
      calibrateProgram_(context, shaderPath("calibrate.comp.spv"), 4, kPushConstantSize, 2),
      differenceProgram_(context, shaderPath("difference.comp.spv"), 3, kPushConstantSize, 2),
      backgroundProgram_(context, shaderPath("background.comp.spv"), 3, kPushConstantSize),
      detectProgram_(context, shaderPath("detect.comp.spv"), 5, kPushConstantSize) {
    bindStages();
}

void TransientPipeline::bindStages() {
    // The calibrated output alternates between the two ping pong buffers so the
    // previous frame survives. Calibrate and difference therefore keep one
    // descriptor set per arrangement, indexed by which buffer is the current one,
    // and nothing has to be rewritten per frame.
    for (std::uint32_t current = 0; current < 2; ++current) {
        const std::uint32_t previous = 1 - current;
        calibrateProgram_.bindBuffers(
            current, {&rawBuffer_, &biasBuffer_, &flatInvBuffer_, &calibrated_[current]});
        differenceProgram_.bindBuffers(
            current, {&calibrated_[current], &calibrated_[previous], &diffBuffer_});
    }

    // Noise is measured on the difference image, so the background stage reads diff.
    backgroundProgram_.bindBuffers(0, {&diffBuffer_, &diffMeanBuffer_, &diffStdBuffer_});
    detectProgram_.bindBuffers(0, {&diffBuffer_, &diffMeanBuffer_, &diffStdBuffer_,
                                   &counterBuffer_, &detectionsBuffer_});
}

void TransientPipeline::requireFrameSized(std::span<const float> data, const char* what) const {
    if (data.size() != pixelCount_) {
        throw std::invalid_argument(std::format("{} has {} pixels, the pipeline expects {}", what,
                                                data.size(), pixelCount_));
    }
}

void TransientPipeline::uploadCalibration(std::span<const float> bias,
                                          std::span<const float> flatInv) {
    requireFrameSized(bias, "bias field");
    requireFrameSized(flatInv, "inverse flat field");

    const auto upload = [this](std::span<const float> data, const Buffer& target) {
        stagingUpload_.write(data);
        context_.submitAndWait([&](VkCommandBuffer commands) {
            copyBuffer(commands, stagingUpload_, target, frameBytes_);
        });
    };
    upload(bias, biasBuffer_);
    upload(flatInv, flatInvBuffer_);
}

FrameResult TransientPipeline::processFrame(std::span<const float> raw, bool readCalibrated) {
    requireFrameSized(raw, "raw frame");

    const std::uint32_t current = frameCounter_ % 2;
    const PushConstants pc{
        .width = config_.width,
        .height = config_.height,
        .annulusOuter = config_.annulusOuter,
        .annulusInner = config_.annulusInner,
        .detectSigma = config_.detectSigma,
        .shapeRatio = config_.shapeRatio,
        .maxDetections = config_.maxDetections,
        .hasPrev = frameCounter_ > 0 ? 1u : 0u,
    };
    const std::span<const std::byte> pcBytes = std::as_bytes(std::span(&pc, 1));
    const std::uint32_t tilesX = tileCount(config_.width);
    const std::uint32_t tilesY = tileCount(config_.height);

    stagingUpload_.write(raw);

    context_.submitAndWait([&](VkCommandBuffer commands) {
        // Upload the raw frame and clear the detection counter.
        copyBuffer(commands, stagingUpload_, rawBuffer_, frameBytes_);
        vkCmdFillBuffer(commands, counterBuffer_.handle(), 0, sizeof(std::uint32_t), 0);
        memoryBarrier(commands, kTransferWrite, kShaderReadWrite);

        // The stages run calibrate, difference, background, detect in order, each
        // reading what the one before it wrote.
        calibrateProgram_.dispatch(commands, current, tilesX, tilesY, pcBytes);
        memoryBarrier(commands, kShaderWrite, kShaderRead);

        differenceProgram_.dispatch(commands, current, tilesX, tilesY, pcBytes);
        memoryBarrier(commands, kShaderWrite, kShaderRead);

        backgroundProgram_.dispatch(commands, 0, tilesX, tilesY, pcBytes);
        memoryBarrier(commands, kShaderWrite, kShaderRead);

        detectProgram_.dispatch(commands, 0, tilesX, tilesY, pcBytes);
        memoryBarrier(commands, kShaderWrite, kTransferRead);

        // Pull back the counter and the detection list, and the calibrated frame
        // if it is wanted.
        copyBuffer(commands, counterBuffer_, readbackCounter_, sizeof(std::uint32_t));
        copyBuffer(commands, detectionsBuffer_, readbackDetections_, detectionBytes_);
        if (readCalibrated) {
            copyBuffer(commands, calibrated_[current], readbackCalibrated_, frameBytes_);
        }
    });

    FrameResult result;
    result.reportedCount = readbackCounter_.view<std::uint32_t>(1).front();

    const std::uint32_t stored = std::min(result.reportedCount, config_.maxDetections);
    const std::span<const Detection> found = readbackDetections_.view<Detection>(stored);
    result.detections.assign(found.begin(), found.end());
    // Invocations append in whatever order they happen to run. Sorting by row and
    // column makes the catalogue reproducible and matches the CPU reference.
    std::ranges::sort(result.detections, std::ranges::less{},
                      [](const Detection& d) { return std::pair(d.y, d.x); });

    if (readCalibrated) {
        result.calibrated = readbackCalibrated_.view<float>(pixelCount_);
    }

    ++frameCounter_;
    return result;
}

} // namespace vkp
