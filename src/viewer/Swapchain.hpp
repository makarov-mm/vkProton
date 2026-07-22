#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

namespace vkp {

class VulkanContext;

// Wraps a swapchain and its images. The viewer blits its display image into an
// acquired swapchain image, so the images are created with transfer destination
// usage rather than colour attachment usage. There are no render passes or
// framebuffers here, which keeps the presentation path small.
class Swapchain {
public:
    Swapchain(const VulkanContext& context, uint32_t width, uint32_t height);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    // Rebuilds the swapchain at a new size, for example after a window resize.
    void recreate(uint32_t width, uint32_t height);

    // Acquires the next image. Returns the raw result so the caller can react to
    // VK_ERROR_OUT_OF_DATE_KHR and VK_SUBOPTIMAL_KHR by recreating.
    VkResult acquire(VkSemaphore signalSemaphore, uint32_t& imageIndex);

    VkSwapchainKHR handle() const { return swapchain_; }
    VkFormat format() const { return format_; }
    VkExtent2D extent() const { return extent_; }
    uint32_t imageCount() const { return static_cast<uint32_t>(images_.size()); }
    VkImage image(uint32_t index) const { return images_[index]; }

private:
    void create(uint32_t width, uint32_t height);
    void destroy();

    const VulkanContext& context_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
};

} // namespace vkp
