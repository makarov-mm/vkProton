#pragma once

#include <cstdint>
#include <span>

#include <vulkan/vulkan.h>

#include "Swapchain.hpp"
#include "../core/Buffer.hpp"
#include "../core/Handle.hpp"

namespace vkp {

class VulkanContext;
class Window;

// Presents pipeline output to a window. Each frame arrives as a ready made RGBA
// image the size of a telescope frame. The viewer uploads it to a device image
// and blits that into the acquired swapchain image, scaled to fit the window with
// its aspect ratio kept. There is no render pass: the whole path is copies and a
// blit, which is the smallest correct way to get compute results on screen.
class Viewer {
public:
    Viewer(const VulkanContext& context, const Window& window,
           std::uint32_t frameWidth, std::uint32_t frameHeight);
    ~Viewer();

    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;

    // Uploads and presents one frame. The pixel buffer must hold
    // frameWidth * frameHeight * 4 bytes in RGBA order. Frames are dropped while
    // the window is minimised.
    void showFrame(std::span<const std::uint8_t> rgba);

private:
    void createDisplayImage();
    void rebuildSwapchain();
    void recordFrame(std::uint32_t imageIndex) const;

    const VulkanContext& context_;
    const Window& window_;
    VkExtent2D frameExtent_;
    // Framebuffer size the swapchain was last built for.
    VkExtent2D windowExtent_;

    Swapchain swapchain_;
    Buffer staging_;
    MemoryHandle displayMemory_;
    ImageHandle displayImage_;

    VkCommandBuffer commands_;
    SemaphoreHandle imageAvailable_;
    FenceHandle inFlight_;
};

} // namespace vkp
