#include "Swapchain.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "../core/VulkanContext.hpp"
#include "../util/Log.hpp"

namespace vkp {

namespace {
VkSurfaceFormatKHR chooseFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats.front();
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
    // FIFO is always available and gives a tear free, steady cadence, which suits
    // a viewer that advances at telescope frame rates rather than game frame rates.
    for (const auto& m : modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
            return m;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}
} // namespace

Swapchain::Swapchain(const VulkanContext& context, uint32_t width, uint32_t height)
    : context_(context) {
    create(width, height);
}

Swapchain::~Swapchain() {
    destroy();
}

void Swapchain::destroy() {
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(context_.device(), swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
    images_.clear();
}

void Swapchain::recreate(uint32_t width, uint32_t height) {
    vkDeviceWaitIdle(context_.device());
    destroy();
    create(width, height);
}

void Swapchain::create(uint32_t width, uint32_t height) {
    VkPhysicalDevice gpu = context_.physicalDevice();
    VkSurfaceKHR surface = context_.surface();

    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &caps));

    uint32_t formatCount = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &formatCount, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &formatCount, formats.data()));

    uint32_t modeCount = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &modeCount, nullptr));
    std::vector<VkPresentModeKHR> modes(modeCount);
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &modeCount, modes.data()));

    VkSurfaceFormatKHR surfaceFormat = chooseFormat(formats);
    format_ = surfaceFormat.format;

    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        extent_ = caps.currentExtent;
    } else {
        extent_.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent_.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }

    // The viewer blits into the swapchain image, so transfer destination usage is
    // required. It is part of the guaranteed supported set, but check to be safe.
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if ((caps.supportedUsageFlags & usage) != usage) {
        logError("swapchain does not support transfer destination usage");
        std::abort();
    }

    VkSwapchainCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = surface;
    info.minImageCount = imageCount;
    info.imageFormat = format_;
    info.imageColorSpace = surfaceFormat.colorSpace;
    info.imageExtent = extent_;
    info.imageArrayLayers = 1;
    info.imageUsage = usage;
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = choosePresentMode(modes);
    info.clipped = VK_TRUE;
    info.oldSwapchain = VK_NULL_HANDLE;

    const uint32_t computeFamily = context_.computeQueueFamily();
    const uint32_t presentFamily = context_.presentQueueFamily();
    uint32_t families[2] = {computeFamily, presentFamily};
    if (computeFamily != presentFamily) {
        info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = 2;
        info.pQueueFamilyIndices = families;
    } else {
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VK_CHECK(vkCreateSwapchainKHR(context_.device(), &info, nullptr, &swapchain_));

    uint32_t actualCount = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(context_.device(), swapchain_, &actualCount, nullptr));
    images_.resize(actualCount);
    VK_CHECK(vkGetSwapchainImagesKHR(context_.device(), swapchain_, &actualCount, images_.data()));
}

VkResult Swapchain::acquire(VkSemaphore signalSemaphore, uint32_t& imageIndex) {
    return vkAcquireNextImageKHR(context_.device(), swapchain_,
                                 std::numeric_limits<uint64_t>::max(),
                                 signalSemaphore, VK_NULL_HANDLE, &imageIndex);
}

} // namespace vkp
