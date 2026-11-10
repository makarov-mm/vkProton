#include "Swapchain.hpp"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include "../core/VulkanContext.hpp"
#include "../core/VulkanUtil.hpp"

namespace vkp {

namespace {

VkSurfaceFormatKHR chooseFormat(std::span<const VkSurfaceFormatKHR> formats) {
    if (formats.empty()) {
        throw std::runtime_error("the surface reports no formats");
    }
    // The frames arrive display ready, so a plain 8 bit UNORM format passes them
    // through unchanged. An sRGB format would apply a second transfer curve.
    const auto wanted = std::ranges::find_if(formats, [](const VkSurfaceFormatKHR& candidate) {
        return (candidate.format == VK_FORMAT_B8G8R8A8_UNORM ||
                candidate.format == VK_FORMAT_R8G8B8A8_UNORM) &&
               candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    });
    return wanted != formats.end() ? *wanted : formats.front();
}

VkPresentModeKHR choosePresentMode(std::span<const VkPresentModeKHR> modes) {
    // The viewer paces itself at telescope frame rates, so mailbox is preferred:
    // presenting then never blocks the loop on the display's refresh. FIFO is the
    // fallback every implementation provides.
    return std::ranges::contains(modes, VK_PRESENT_MODE_MAILBOX_KHR) ? VK_PRESENT_MODE_MAILBOX_KHR
                                                                     : VK_PRESENT_MODE_FIFO_KHR;
}

VkCompositeAlphaFlagBitsKHR chooseCompositeAlpha(VkCompositeAlphaFlagsKHR supported) {
    // Opaque is what a viewer wants. The rest only matter where it is not offered.
    for (const VkCompositeAlphaFlagBitsKHR mode :
         {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
          VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR}) {
        if ((supported & mode) != 0) {
            return mode;
        }
    }
    return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
}

} // namespace

Swapchain::Swapchain(const VulkanContext& context, VkExtent2D size) : context_(context) {
    if (!create(size)) {
        throw std::runtime_error("the window has no drawable area to present to");
    }
}

void Swapchain::recreate(VkExtent2D size) {
    vkCheck(vkDeviceWaitIdle(context_.device()), "vkDeviceWaitIdle");
    create(size);
}

bool Swapchain::create(VkExtent2D size) {
    const VkPhysicalDevice gpu = context_.physicalDevice();
    const VkSurfaceKHR surface = context_.surface();
    const VkDevice device = context_.device();

    VkSurfaceCapabilitiesKHR caps{};
    vkCheck(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &caps),
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    const auto formats = enumerate<VkSurfaceFormatKHR>(
        [&](std::uint32_t* count, VkSurfaceFormatKHR* items) {
            vkCheck(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, count, items),
                    "vkGetPhysicalDeviceSurfaceFormatsKHR");
        });
    const auto modes = enumerate<VkPresentModeKHR>(
        [&](std::uint32_t* count, VkPresentModeKHR* items) {
            vkCheck(vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, count, items),
                    "vkGetPhysicalDeviceSurfacePresentModesKHR");
        });

    // The viewer blits into the swapchain image, which needs transfer destination
    // usage. Only colour attachment usage is guaranteed, so check for it.
    constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if ((caps.supportedUsageFlags & usage) != usage) {
        throw std::runtime_error("the swapchain does not support transfer destination usage");
    }

    // Most platforms dictate the extent. Where they leave it open, follow the window.
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == std::numeric_limits<std::uint32_t>::max()) {
        extent.width = std::clamp(size.width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height =
            std::clamp(size.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    if (extent.width == 0 || extent.height == 0) {
        // The window was minimised in the meantime. The current swapchain stays.
        return false;
    }

    std::uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) {
        imageCount = std::min(imageCount, caps.maxImageCount);
    }

    const VkSurfaceFormatKHR surfaceFormat = chooseFormat(formats);
    const std::array<std::uint32_t, 2> families{context_.computeQueueFamily(),
                                                context_.presentQueueFamily()};
    const bool shared = families[0] != families[1];

    const VkSwapchainCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = usage,
        .imageSharingMode = shared ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = shared ? 2u : 0u,
        .pQueueFamilyIndices = shared ? families.data() : nullptr,
        .preTransform = caps.currentTransform,
        .compositeAlpha = chooseCompositeAlpha(caps.supportedCompositeAlpha),
        .presentMode = choosePresentMode(modes),
        .clipped = VK_TRUE,
        // Naming the retired swapchain lets the driver hand its resources over.
        .oldSwapchain = swapchain_.get(),
    };
    SwapchainHandle replacement;
    vkCheck(vkCreateSwapchainKHR(device, &info, nullptr, replacement.put(device)),
            "vkCreateSwapchainKHR");
    swapchain_ = std::move(replacement);
    extent_ = extent;

    images_ = enumerate<VkImage>([&](std::uint32_t* count, VkImage* items) {
        vkCheck(vkGetSwapchainImagesKHR(device, swapchain_.get(), count, items),
                "vkGetSwapchainImagesKHR");
    });

    const VkSemaphoreCreateInfo semaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    presentSemaphores_.clear();
    presentSemaphores_.resize(images_.size());
    for (SemaphoreHandle& semaphore : presentSemaphores_) {
        vkCheck(vkCreateSemaphore(device, &semaphoreInfo, nullptr, semaphore.put(device)),
                "vkCreateSemaphore");
    }
    return true;
}

VkResult Swapchain::acquire(VkSemaphore signalSemaphore, std::uint32_t& imageIndex) const {
    return vkAcquireNextImageKHR(context_.device(), swapchain_.get(),
                                 std::numeric_limits<std::uint64_t>::max(), signalSemaphore,
                                 VK_NULL_HANDLE, &imageIndex);
}

} // namespace vkp
