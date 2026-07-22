#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <vulkan/vulkan.h>

namespace vkp {

// Optional configuration for the context. Headless callers can ignore all of it
// and construct with the convenience constructor. The viewer fills in the
// present fields so the same device can also drive a swapchain.
struct ContextConfig {
    bool enableValidation = false;
    bool enablePresent = false;
    // Instance extensions the surface needs, supplied by the window backend.
    std::vector<const char*> instanceExtensions;
    // Creates a surface from the instance. The window backend owns the platform
    // handle, this just wraps it. Only called when enablePresent is true.
    std::function<VkSurfaceKHR(VkInstance)> surfaceFactory;
};

// Owns the Vulkan instance, the selected device and a compute queue. By default
// this is a headless setup: no surface, no swapchain, no graphics queue, which is
// all a data pipeline needs. When present is enabled it also creates a surface,
// enables the swapchain extension and picks a present capable queue, so the
// viewer can display results from the same device.
class VulkanContext {
public:
    explicit VulkanContext(bool enableValidation);
    explicit VulkanContext(const ContextConfig& config);
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    VkInstance instance() const { return instance_; }
    VkDevice device() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    VkQueue computeQueue() const { return computeQueue_; }
    uint32_t computeQueueFamily() const { return computeQueueFamily_; }
    VkCommandPool commandPool() const { return commandPool_; }

    bool hasPresent() const { return hasPresent_; }
    VkSurfaceKHR surface() const { return surface_; }
    VkQueue presentQueue() const { return presentQueue_; }
    uint32_t presentQueueFamily() const { return presentQueueFamily_; }

    const char* deviceName() const { return deviceProperties_.deviceName; }

    // Finds a memory type index that satisfies the given type bits and property
    // flags. Aborts if none is found, because a pipeline cannot proceed without it.
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags) const;

    // Allocates and begins a primary command buffer, runs the recorder, then
    // submits it to the compute queue and waits for completion.
    void runCommands(const std::function<void(VkCommandBuffer)>& recorder) const;

private:
    void createInstance();
    void selectPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();
    bool familySupportsPresent(VkPhysicalDevice device, uint32_t family) const;

    ContextConfig config_;
    bool hasPresent_ = false;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties deviceProperties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    uint32_t computeQueueFamily_ = 0;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    uint32_t presentQueueFamily_ = 0;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
};

} // namespace vkp
