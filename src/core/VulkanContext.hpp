#pragma once

#include <concepts>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

#include "Handle.hpp"

namespace vkp {

// Optional settings for the context. A default constructed config gives a headless
// context. The viewer fills in the presentation fields so the same device can also
// drive a swapchain.
struct ContextConfig {
    bool enableValidation = false;
    // Instance extensions the surface needs, supplied by the window backend.
    std::vector<const char*> instanceExtensions;
    // Wraps the window in a surface for the given instance. The window backend owns
    // the platform handle. Setting this turns presentation on.
    std::function<VkSurfaceKHR(VkInstance)> createSurface;
};

// Owns the Vulkan instance, the selected device and a compute queue. By default
// this is a headless setup: no surface, no swapchain, no graphics queue, which is
// all a data pipeline needs. With presentation on it also owns the surface, enables
// the swapchain extension and picks queues that can blit and present, so the viewer
// can display results from the same device.
class VulkanContext {
public:
    explicit VulkanContext(const ContextConfig& config = {});

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    [[nodiscard]] VkDevice device() const noexcept { return device_.get(); }
    [[nodiscard]] VkPhysicalDevice physicalDevice() const noexcept { return physicalDevice_; }
    [[nodiscard]] const VkPhysicalDeviceLimits& limits() const noexcept {
        return properties_.limits;
    }

    // The queue all work is submitted to. In a presenting context its family also
    // supports graphics, because the viewer blits on it.
    [[nodiscard]] VkQueue computeQueue() const noexcept { return computeQueue_; }
    [[nodiscard]] std::uint32_t computeQueueFamily() const noexcept {
        return queueFamilies_.compute;
    }

    // Presentation state. Only meaningful when the context was created with a surface.
    [[nodiscard]] VkSurfaceKHR surface() const noexcept { return surface_.get(); }
    [[nodiscard]] VkQueue presentQueue() const noexcept { return presentQueue_; }
    [[nodiscard]] std::uint32_t presentQueueFamily() const noexcept {
        return queueFamilies_.present;
    }

    // Finds a memory type allowed by typeBits that has all the required property
    // flags, preferring one that also has the preferred flags. Throws if none exists,
    // because a pipeline cannot proceed without it.
    [[nodiscard]] std::uint32_t findMemoryType(std::uint32_t typeBits,
                                               VkMemoryPropertyFlags required,
                                               VkMemoryPropertyFlags preferred = 0) const;

    // Allocates a primary command buffer from the context's pool. It lives as long
    // as the context does.
    [[nodiscard]] VkCommandBuffer allocateCommandBuffer() const;

    // Records commands through the given callable, submits them to the compute queue
    // and waits for completion.
    template <std::invocable<VkCommandBuffer> Recorder>
    void submitAndWait(Recorder&& record) const {
        const VkCommandBuffer commands = beginImmediate();
        std::invoke(std::forward<Recorder>(record), commands);
        submitImmediate(commands);
    }

private:
    struct QueueFamilies {
        std::uint32_t compute = 0;
        std::uint32_t present = 0;
    };

    void createInstance(const ContextConfig& config);
    void selectPhysicalDevice();
    void createDevice();
    void createCommandPool();

    // Queue families this context would use on the given device, or nothing if the
    // device cannot provide them.
    [[nodiscard]] std::optional<QueueFamilies> findQueueFamilies(VkPhysicalDevice device) const;
    [[nodiscard]] bool presenting() const noexcept { return surface_.get() != VK_NULL_HANDLE; }

    [[nodiscard]] VkCommandBuffer beginImmediate() const;
    void submitImmediate(VkCommandBuffer commands) const;

    // Declaration order is creation order, so destruction runs the right way round:
    // command pool, device, surface, instance.
    InstanceHandle instance_;
    SurfaceHandle surface_;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties_{};
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    QueueFamilies queueFamilies_;
    DeviceHandle device_;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    CommandPoolHandle commandPool_;
    VkCommandBuffer immediateCommands_ = VK_NULL_HANDLE;
};

} // namespace vkp
