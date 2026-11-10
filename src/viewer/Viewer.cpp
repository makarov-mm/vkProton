#include "Viewer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>

#include "Window.hpp"
#include "../core/VulkanContext.hpp"
#include "../core/VulkanUtil.hpp"

namespace vkp {

namespace {

constexpr VkImageSubresourceRange kColorRange{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
};

constexpr VkImageSubresourceLayers kColorLayer{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .mipLevel = 0,
    .baseArrayLayer = 0,
    .layerCount = 1,
};

// What an image is doing on one side of a barrier: its layout, plus the pipeline
// stage and access that touch it there.
struct ImageState {
    VkImageLayout layout;
    VkPipelineStageFlags stage;
    VkAccessFlags access;
};

// Contents not needed. The transfer stage orders the layout change after whatever
// used the image there before: the previous frame's blit for the display image,
// and for a swapchain image the acquire semaphore, which the submit waits on at
// that same stage.
constexpr ImageState kDiscarded{VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_TRANSFER_BIT, 0};
constexpr ImageState kTransferTarget{VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
constexpr ImageState kTransferSource{VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT};
constexpr ImageState kPresentable{VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0};

void imageBarrier(VkCommandBuffer commands, VkImage image, ImageState from, ImageState to) {
    const VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = from.access,
        .dstAccessMask = to.access,
        .oldLayout = from.layout,
        .newLayout = to.layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = kColorRange,
    };
    vkCmdPipelineBarrier(commands, from.stage, to.stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// A frame has to fit in a single 2D image on this device to be displayed.
VkExtent2D checkedFrameExtent(const VulkanContext& context, std::uint32_t width,
                              std::uint32_t height) {
    const std::uint32_t limit = context.limits().maxImageDimension2D;
    if (width == 0 || height == 0 || width > limit || height > limit) {
        throw std::runtime_error(std::format(
            "a {}x{} frame cannot be displayed, the device limit is {} pixels per side", width,
            height, limit));
    }
    return {width, height};
}

// The largest rectangle with the frame's aspect ratio that fits in the target,
// centred, as the two corner offsets a blit expects.
std::array<VkOffset3D, 2> letterbox(VkExtent2D frame, VkExtent2D target) {
    const double scale = std::min(static_cast<double>(target.width) / frame.width,
                                  static_cast<double>(target.height) / frame.height);
    const auto fit = [scale](std::uint32_t frameSide, std::uint32_t targetSide) {
        const auto scaled = static_cast<std::int32_t>(std::lround(frameSide * scale));
        return std::clamp(scaled, 1, static_cast<std::int32_t>(targetSide));
    };
    const std::int32_t width = fit(frame.width, target.width);
    const std::int32_t height = fit(frame.height, target.height);
    const std::int32_t x = (static_cast<std::int32_t>(target.width) - width) / 2;
    const std::int32_t y = (static_cast<std::int32_t>(target.height) - height) / 2;
    return {{{x, y, 0}, {x + width, y + height, 1}}};
}

} // namespace

Viewer::Viewer(const VulkanContext& context, const Window& window,
               std::uint32_t frameWidth, std::uint32_t frameHeight)
    : context_(context),
      window_(window),
      frameExtent_(checkedFrameExtent(context, frameWidth, frameHeight)),
      windowExtent_(window.framebufferSize()),
      swapchain_(context, windowExtent_),
      staging_(context, VkDeviceSize{frameWidth} * frameHeight * 4, Buffer::Kind::Upload),
      commands_(context.allocateCommandBuffer()) {
    createDisplayImage();

    const VkDevice device = context_.device();
    const VkSemaphoreCreateInfo semaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCheck(vkCreateSemaphore(device, &semaphoreInfo, nullptr, imageAvailable_.put(device)),
            "vkCreateSemaphore");

    // The fence starts signalled, so the first frame has nothing to wait for.
    const VkFenceCreateInfo fenceInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    vkCheck(vkCreateFence(device, &fenceInfo, nullptr, inFlight_.put(device)), "vkCreateFence");
}

Viewer::~Viewer() {
    // The last frame may still be executing or waiting to be presented.
    vkDeviceWaitIdle(context_.device());
}

void Viewer::createDisplayImage() {
    const VkDevice device = context_.device();

    const VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {frameExtent_.width, frameExtent_.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    vkCheck(vkCreateImage(device, &imageInfo, nullptr, displayImage_.put(device)),
            "vkCreateImage");

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, displayImage_.get(), &requirements);

    const VkMemoryAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = context_.findMemoryType(requirements.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };
    vkCheck(vkAllocateMemory(device, &allocInfo, nullptr, displayMemory_.put(device)),
            "vkAllocateMemory");
    vkCheck(vkBindImageMemory(device, displayImage_.get(), displayMemory_.get(), 0),
            "vkBindImageMemory");
}

void Viewer::rebuildSwapchain() {
    const VkExtent2D size = window_.framebufferSize();
    if (size.width == 0 || size.height == 0) {
        return;  // minimised, nothing to present to
    }
    swapchain_.recreate(size);
    windowExtent_ = size;
}

void Viewer::showFrame(std::span<const std::uint8_t> rgba) {
    if (rgba.size() != staging_.size()) {
        throw std::invalid_argument("frame pixel buffer does not match the display image");
    }

    // Follow the window size directly. Not every platform reports a resize through
    // an out of date swapchain.
    const VkExtent2D size = window_.framebufferSize();
    if (size.width == 0 || size.height == 0) {
        return;
    }
    if (size.width != windowExtent_.width || size.height != windowExtent_.height) {
        rebuildSwapchain();
    }

    // Wait for the previous frame before touching the staging buffer it reads.
    const VkDevice device = context_.device();
    const VkFence inFlight = inFlight_.get();
    constexpr std::uint64_t noTimeout = std::numeric_limits<std::uint64_t>::max();
    vkCheck(vkWaitForFences(device, 1, &inFlight, VK_TRUE, noTimeout), "vkWaitForFences");
    staging_.write(rgba);

    std::uint32_t imageIndex = 0;
    const VkSemaphore imageAvailable = imageAvailable_.get();
    const VkResult acquired = swapchain_.acquire(imageAvailable, imageIndex);
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
        rebuildSwapchain();
        return;
    }
    if (acquired != VK_SUBOPTIMAL_KHR) {
        vkCheck(acquired, "vkAcquireNextImageKHR");
    }

    recordFrame(imageIndex);

    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkSemaphore presentReady = swapchain_.presentSemaphore(imageIndex);
    const VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &imageAvailable,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &commands_,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &presentReady,
    };
    vkCheck(vkResetFences(device, 1, &inFlight), "vkResetFences");
    vkCheck(vkQueueSubmit(context_.computeQueue(), 1, &submitInfo, inFlight), "vkQueueSubmit");

    const VkSwapchainKHR swapchain = swapchain_.handle();
    const VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &presentReady,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &imageIndex,
    };
    const VkResult presented = vkQueuePresentKHR(context_.presentQueue(), &presentInfo);
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) {
        rebuildSwapchain();
    } else {
        vkCheck(presented, "vkQueuePresentKHR");
    }
}

void Viewer::recordFrame(std::uint32_t imageIndex) const {
    const VkImage displayImage = displayImage_.get();
    const VkImage swapImage = swapchain_.image(imageIndex);

    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkCheck(vkBeginCommandBuffer(commands_, &beginInfo), "vkBeginCommandBuffer");

    // Upload the new frame into the display image, then make it the blit source.
    imageBarrier(commands_, displayImage, kDiscarded, kTransferTarget);
    const VkBufferImageCopy upload{
        .imageSubresource = kColorLayer,
        .imageExtent = {frameExtent_.width, frameExtent_.height, 1},
    };
    vkCmdCopyBufferToImage(commands_, staging_.handle(), displayImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &upload);
    imageBarrier(commands_, displayImage, kTransferTarget, kTransferSource);

    // Clear the swapchain image to black, which is what shows around the frame
    // when the window has a different aspect ratio. The clear and the blit both
    // write the image, so a barrier in the same layout puts them in order.
    imageBarrier(commands_, swapImage, kDiscarded, kTransferTarget);
    const VkClearColorValue black{.float32 = {0.0f, 0.0f, 0.0f, 1.0f}};
    vkCmdClearColorImage(commands_, swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1,
                         &kColorRange);
    imageBarrier(commands_, swapImage, kTransferTarget, kTransferTarget);

    // Blit, scaling the frame to fit the window.
    const std::array<VkOffset3D, 2> target = letterbox(frameExtent_, swapchain_.extent());
    const VkImageBlit blit{
        .srcSubresource = kColorLayer,
        .srcOffsets = {{0, 0, 0},
                       {static_cast<std::int32_t>(frameExtent_.width),
                        static_cast<std::int32_t>(frameExtent_.height), 1}},
        .dstSubresource = kColorLayer,
        .dstOffsets = {target[0], target[1]},
    };
    vkCmdBlitImage(commands_, displayImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

    // Hand the swapchain image to the presentation engine.
    imageBarrier(commands_, swapImage, kTransferTarget, kPresentable);

    vkCheck(vkEndCommandBuffer(commands_), "vkEndCommandBuffer");
}

} // namespace vkp
