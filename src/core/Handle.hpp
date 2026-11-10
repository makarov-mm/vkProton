#pragma once

#include <utility>

#include <vulkan/vulkan.h>

namespace vkp {

// Move only owner of one Vulkan object. Parent is the object it is destroyed
// through, the device for most types and the instance for a surface, and Destroy
// is the matching vkDestroy or vkFree function. Because every handle cleans up
// after itself, the classes that hold them need no destructors and stay leak free
// when a constructor throws half way through.
template <typename Parent, typename T, auto Destroy>
class Handle {
public:
    Handle() = default;
    Handle(Parent parent, T handle) noexcept : parent_(parent), handle_(handle) {}
    ~Handle() { reset(); }

    Handle(Handle&& other) noexcept
        : parent_(other.parent_), handle_(std::exchange(other.handle_, T{})) {}

    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) {
            reset();
            parent_ = other.parent_;
            handle_ = std::exchange(other.handle_, T{});
        }
        return *this;
    }

    [[nodiscard]] T get() const noexcept { return handle_; }

    // Destroys the current object, if any, and returns the slot a vkCreate call
    // writes the new one into.
    [[nodiscard]] T* put(Parent parent) noexcept {
        reset();
        parent_ = parent;
        return &handle_;
    }

    void reset() noexcept {
        if (handle_ != T{}) {
            Destroy(parent_, handle_, nullptr);
            handle_ = T{};
        }
    }

private:
    Parent parent_{};
    T handle_{};
};

namespace detail {

// The instance and the device have no parent object. An empty tag gives their
// destroy functions the same shape as all the others.
struct NoParent {};

inline void destroyInstance(NoParent, VkInstance instance, const VkAllocationCallbacks* allocator) {
    vkDestroyInstance(instance, allocator);
}

inline void destroyDevice(NoParent, VkDevice device, const VkAllocationCallbacks* allocator) {
    vkDestroyDevice(device, allocator);
}

} // namespace detail

using InstanceHandle = Handle<detail::NoParent, VkInstance, detail::destroyInstance>;
using DeviceHandle = Handle<detail::NoParent, VkDevice, detail::destroyDevice>;
using SurfaceHandle = Handle<VkInstance, VkSurfaceKHR, vkDestroySurfaceKHR>;

template <typename T, auto Destroy>
using DeviceChild = Handle<VkDevice, T, Destroy>;

using BufferHandle = DeviceChild<VkBuffer, vkDestroyBuffer>;
using MemoryHandle = DeviceChild<VkDeviceMemory, vkFreeMemory>;
using ImageHandle = DeviceChild<VkImage, vkDestroyImage>;
using CommandPoolHandle = DeviceChild<VkCommandPool, vkDestroyCommandPool>;
using ShaderModuleHandle = DeviceChild<VkShaderModule, vkDestroyShaderModule>;
using DescriptorSetLayoutHandle = DeviceChild<VkDescriptorSetLayout, vkDestroyDescriptorSetLayout>;
using DescriptorPoolHandle = DeviceChild<VkDescriptorPool, vkDestroyDescriptorPool>;
using PipelineLayoutHandle = DeviceChild<VkPipelineLayout, vkDestroyPipelineLayout>;
using PipelineHandle = DeviceChild<VkPipeline, vkDestroyPipeline>;
using SemaphoreHandle = DeviceChild<VkSemaphore, vkDestroySemaphore>;
using FenceHandle = DeviceChild<VkFence, vkDestroyFence>;
using SwapchainHandle = DeviceChild<VkSwapchainKHR, vkDestroySwapchainKHR>;

} // namespace vkp
