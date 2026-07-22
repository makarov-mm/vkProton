#include "Viewer.hpp"

#include <cstring>

#include "Swapchain.hpp"
#include "Window.hpp"
#include "../core/VulkanContext.hpp"
#include "../util/Log.hpp"

namespace vkp {

namespace {
void transitionImage(VkCommandBuffer cmd, VkImage image,
                     VkImageLayout oldLayout, VkImageLayout newLayout,
                     VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                     VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}
} // namespace

Viewer::Viewer(const VulkanContext& context, const Window& window,
               uint32_t frameWidth, uint32_t frameHeight)
    : context_(context), window_(window), frameWidth_(frameWidth), frameHeight_(frameHeight) {
    uint32_t w = frameWidth_;
    uint32_t h = frameHeight_;
    window_.framebufferSize(w, h);
    if (w == 0 || h == 0) {
        w = frameWidth_;
        h = frameHeight_;
    }
    swapchain_ = std::make_unique<Swapchain>(context_, w, h);
    createDisplayImage();
    createStaging();
    createSyncAndCommands();
}

Viewer::~Viewer() {
    VkDevice device = context_.device();
    vkDeviceWaitIdle(device);

    if (inFlight_ != VK_NULL_HANDLE) vkDestroyFence(device, inFlight_, nullptr);
    if (imageAvailable_ != VK_NULL_HANDLE) vkDestroySemaphore(device, imageAvailable_, nullptr);
    if (renderFinished_ != VK_NULL_HANDLE) vkDestroySemaphore(device, renderFinished_, nullptr);
    if (commandBuffer_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device, context_.commandPool(), 1, &commandBuffer_);
    }
    if (stagingMapped_ != nullptr) vkUnmapMemory(device, stagingMemory_);
    if (staging_ != VK_NULL_HANDLE) vkDestroyBuffer(device, staging_, nullptr);
    if (stagingMemory_ != VK_NULL_HANDLE) vkFreeMemory(device, stagingMemory_, nullptr);
    if (displayImage_ != VK_NULL_HANDLE) vkDestroyImage(device, displayImage_, nullptr);
    if (displayMemory_ != VK_NULL_HANDLE) vkFreeMemory(device, displayMemory_, nullptr);

    swapchain_.reset();
}

void Viewer::createDisplayImage() {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R8G8B8A8_UNORM;
    info.extent = {frameWidth_, frameHeight_, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(context_.device(), &info, nullptr, &displayImage_));

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(context_.device(), displayImage_, &requirements);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = context_.findMemoryType(requirements.memoryTypeBits,
                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(context_.device(), &alloc, nullptr, &displayMemory_));
    VK_CHECK(vkBindImageMemory(context_.device(), displayImage_, displayMemory_, 0));
}

void Viewer::createStaging() {
    stagingBytes_ = static_cast<VkDeviceSize>(frameWidth_) * frameHeight_ * 4;

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = stagingBytes_;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(context_.device(), &info, nullptr, &staging_));

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(context_.device(), staging_, &requirements);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = context_.findMemoryType(
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(context_.device(), &alloc, nullptr, &stagingMemory_));
    VK_CHECK(vkBindBufferMemory(context_.device(), staging_, stagingMemory_, 0));
    VK_CHECK(vkMapMemory(context_.device(), stagingMemory_, 0, stagingBytes_, 0, &stagingMapped_));
}

void Viewer::createSyncAndCommands() {
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = context_.commandPool();
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(context_.device(), &alloc, &commandBuffer_));

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VK_CHECK(vkCreateSemaphore(context_.device(), &semInfo, nullptr, &imageAvailable_));
    VK_CHECK(vkCreateSemaphore(context_.device(), &semInfo, nullptr, &renderFinished_));

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(context_.device(), &fenceInfo, nullptr, &inFlight_));
}

void Viewer::recreateSwapchain() {
    uint32_t w = 0;
    uint32_t h = 0;
    window_.framebufferSize(w, h);
    if (w == 0 || h == 0) {
        return; // minimised, nothing to present to
    }
    swapchain_->recreate(w, h);
}

void Viewer::showFrame(const std::vector<uint8_t>& rgba) {
    if (rgba.size() < stagingBytes_) {
        logError("frame pixel buffer is smaller than the display image");
        return;
    }

    vkWaitForFences(context_.device(), 1, &inFlight_, VK_TRUE, UINT64_MAX);

    std::memcpy(stagingMapped_, rgba.data(), static_cast<size_t>(stagingBytes_));

    uint32_t imageIndex = 0;
    VkResult acquireResult = swapchain_->acquire(imageAvailable_, imageIndex);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        logError("failed to acquire a swapchain image");
        return;
    }

    vkResetFences(context_.device(), 1, &inFlight_);
    recordAndSubmit(imageIndex);

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &renderFinished_;
    VkSwapchainKHR chain = swapchain_->handle();
    present.swapchainCount = 1;
    present.pSwapchains = &chain;
    present.pImageIndices = &imageIndex;

    VkResult presentResult = vkQueuePresentKHR(context_.presentQueue(), &present);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        recreateSwapchain();
    }
}

void Viewer::recordAndSubmit(uint32_t imageIndex) {
    VkImage swapImage = swapchain_->image(imageIndex);
    VkExtent2D extent = swapchain_->extent();

    VK_CHECK(vkResetCommandBuffer(commandBuffer_, 0));
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer_, &begin));

    // Upload the new frame into the display image.
    transitionImage(commandBuffer_, displayImage_,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy copy{};
    copy.bufferOffset = 0;
    copy.bufferRowLength = 0;
    copy.bufferImageHeight = 0;
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageOffset = {0, 0, 0};
    copy.imageExtent = {frameWidth_, frameHeight_, 1};
    vkCmdCopyBufferToImage(commandBuffer_, staging_, displayImage_,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    // Prepare display image as blit source and the swapchain image as blit dest.
    transitionImage(commandBuffer_, displayImage_,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    transitionImage(commandBuffer_, swapImage,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Blit, scaling the frame to the window size.
    VkImageBlit blit{};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel = 0;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[0] = {0, 0, 0};
    blit.srcOffsets[1] = {static_cast<int32_t>(frameWidth_), static_cast<int32_t>(frameHeight_), 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.dstOffsets[0] = {0, 0, 0};
    blit.dstOffsets[1] = {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
    vkCmdBlitImage(commandBuffer_,
                   displayImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blit, VK_FILTER_LINEAR);

    // Hand the swapchain image to the presentation engine.
    transitionImage(commandBuffer_, swapImage,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    VK_CHECK(vkEndCommandBuffer(commandBuffer_));

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &imageAvailable_;
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffer_;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &renderFinished_;
    VK_CHECK(vkQueueSubmit(context_.computeQueue(), 1, &submit, inFlight_));
}

} // namespace vkp
