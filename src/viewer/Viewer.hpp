#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

namespace vkp {

class VulkanContext;
class Window;
class Swapchain;

// Presents pipeline output to a window. Each frame arrives as a ready made RGBA
// image the size of a telescope frame. The viewer uploads it to a device image
// and blits that into the acquired swapchain image, scaling to the window. There
// is no render pass: the whole path is copies and a blit, which is the smallest
// correct way to get compute results on screen.
class Viewer {
public:
    Viewer(const VulkanContext& context, const Window& window,
           uint32_t frameWidth, uint32_t frameHeight);
    ~Viewer();

    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;

    // Uploads and presents one frame. The pixel buffer must hold
    // frameWidth * frameHeight * 4 bytes in RGBA order.
    void showFrame(const std::vector<uint8_t>& rgba);

private:
    void createDisplayImage();
    void createStaging();
    void createSyncAndCommands();
    void recordAndSubmit(uint32_t imageIndex);
    void recreateSwapchain();

    const VulkanContext& context_;
    const Window& window_;
    uint32_t frameWidth_;
    uint32_t frameHeight_;

    std::unique_ptr<Swapchain> swapchain_;

    VkImage displayImage_ = VK_NULL_HANDLE;
    VkDeviceMemory displayMemory_ = VK_NULL_HANDLE;

    VkBuffer staging_ = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory_ = VK_NULL_HANDLE;
    void* stagingMapped_ = nullptr;
    VkDeviceSize stagingBytes_ = 0;

    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    VkSemaphore imageAvailable_ = VK_NULL_HANDLE;
    VkSemaphore renderFinished_ = VK_NULL_HANDLE;
    VkFence inFlight_ = VK_NULL_HANDLE;
};

} // namespace vkp
