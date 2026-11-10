#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "../core/Handle.hpp"

namespace vkp {

class VulkanContext;

// Wraps a swapchain, its images and one present semaphore per image. The viewer
// blits its display image into an acquired swapchain image, so the images are
// created with transfer destination usage rather than colour attachment usage.
// There are no render passes or framebuffers here, which keeps the presentation
// path small.
class Swapchain {
public:
    // The size is only used on platforms where the surface does not dictate it.
    Swapchain(const VulkanContext& context, VkExtent2D size);

    // Rebuilds the swapchain at a new size, for example after a window resize. The
    // current one is kept if the window turns out to have no area at the moment.
    void recreate(VkExtent2D size);

    // Acquires the next image. Returns the raw result so the caller can react to
    // VK_ERROR_OUT_OF_DATE_KHR and VK_SUBOPTIMAL_KHR by recreating.
    [[nodiscard]] VkResult acquire(VkSemaphore signalSemaphore, std::uint32_t& imageIndex) const;

    [[nodiscard]] VkSwapchainKHR handle() const noexcept { return swapchain_.get(); }
    [[nodiscard]] VkExtent2D extent() const noexcept { return extent_; }
    [[nodiscard]] VkImage image(std::uint32_t index) const { return images_.at(index); }

    // Signalled when the blit into the given image is done and it can be presented.
    // Each image has its own semaphore, because the presentation engine may still
    // hold the one from the previous present when the next frame is submitted.
    [[nodiscard]] VkSemaphore presentSemaphore(std::uint32_t index) const {
        return presentSemaphores_.at(index).get();
    }

private:
    // Returns false, leaving everything as it was, when the surface has no area.
    bool create(VkExtent2D size);

    const VulkanContext& context_;
    // Declared before the swapchain, so they outlive it on destruction.
    std::vector<SemaphoreHandle> presentSemaphores_;
    SwapchainHandle swapchain_;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
};

} // namespace vkp
