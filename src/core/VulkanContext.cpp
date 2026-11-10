#include "VulkanContext.hpp"

#include <algorithm>
#include <stdexcept>
#include <string_view>

#include "VulkanUtil.hpp"
#include "../util/Log.hpp"

namespace vkp {

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

bool validationLayerAvailable() {
    const auto layers = enumerate<VkLayerProperties>(
        [](std::uint32_t* count, VkLayerProperties* items) {
            vkCheck(vkEnumerateInstanceLayerProperties(count, items),
                    "vkEnumerateInstanceLayerProperties");
        });
    return std::ranges::any_of(layers, [](const VkLayerProperties& layer) {
        return std::string_view(layer.layerName) == kValidationLayer;
    });
}

// Higher is better: dedicated hardware first, a CPU rasteriser only as a last resort.
int deviceRank(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
        default: return 0;
    }
}

} // namespace

VulkanContext::VulkanContext(const ContextConfig& config) {
    createInstance(config);
    if (config.createSurface) {
        surface_ = SurfaceHandle(instance_.get(), config.createSurface(instance_.get()));
        if (!presenting()) {
            throw std::runtime_error("the window backend did not provide a surface");
        }
    }
    selectPhysicalDevice();
    createDevice();
    createCommandPool();
    // The name is a fixed size character array. Viewing it as a string stops at the
    // terminator on every standard library.
    logInfo("using device: {}", std::string_view(properties_.deviceName));
}

void VulkanContext::createInstance(const ContextConfig& config) {
    // Only Vulkan 1.0 core functionality is used, plus the swapchain extension in
    // the viewer, so no newer version is asked for.
    const VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vkPhoton",
        .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .pEngineName = "vkPhoton",
        .engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };

    std::vector<const char*> layers;
    if (config.enableValidation) {
        if (validationLayerAvailable()) {
            layers.push_back(kValidationLayer);
            logInfo("validation layer enabled");
        } else {
            logInfo("validation layer requested but not available, continuing without it");
        }
    }

    const VkInstanceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<std::uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<std::uint32_t>(config.instanceExtensions.size()),
        .ppEnabledExtensionNames = config.instanceExtensions.data(),
    };
    vkCheck(vkCreateInstance(&createInfo, nullptr, instance_.put({})), "vkCreateInstance");
}

std::optional<VulkanContext::QueueFamilies> VulkanContext::findQueueFamilies(
    VkPhysicalDevice device) const {
    const auto families = enumerate<VkQueueFamilyProperties>(
        [device](std::uint32_t* count, VkQueueFamilyProperties* items) {
            vkGetPhysicalDeviceQueueFamilyProperties(device, count, items);
        });

    // The pipeline needs compute. The viewer also blits into the swapchain on the
    // same queue, and vkCmdBlitImage is only allowed on a graphics capable family.
    VkQueueFlags required = VK_QUEUE_COMPUTE_BIT;
    if (presenting()) {
        required |= VK_QUEUE_GRAPHICS_BIT;
    }

    const auto canPresent = [&](std::uint32_t family) {
        VkBool32 supported = VK_FALSE;
        vkCheck(vkGetPhysicalDeviceSurfaceSupportKHR(device, family, surface_.get(), &supported),
                "vkGetPhysicalDeviceSurfaceSupportKHR");
        return supported == VK_TRUE;
    };

    std::optional<std::uint32_t> compute;
    std::optional<std::uint32_t> present;
    for (std::uint32_t family = 0; family < families.size(); ++family) {
        const bool usable = (families[family].queueFlags & required) == required;
        const bool presents = presenting() && canPresent(family);
        if (usable && presents) {
            // One queue serves both roles, which is the common case.
            return QueueFamilies{.compute = family, .present = family};
        }
        if (usable && !compute) {
            compute = family;
        }
        if (presents && !present) {
            present = family;
        }
    }

    if (!compute || (presenting() && !present)) {
        return std::nullopt;
    }
    return QueueFamilies{.compute = *compute, .present = present.value_or(*compute)};
}

void VulkanContext::selectPhysicalDevice() {
    const auto devices = enumerate<VkPhysicalDevice>(
        [this](std::uint32_t* count, VkPhysicalDevice* items) {
            vkCheck(vkEnumeratePhysicalDevices(instance_.get(), count, items),
                    "vkEnumeratePhysicalDevices");
        });
    if (devices.empty()) {
        throw std::runtime_error("no Vulkan capable device found");
    }

    int bestRank = -1;
    for (const VkPhysicalDevice candidate : devices) {
        const auto families = findQueueFamilies(candidate);
        if (!families) {
            continue;
        }
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (const int rank = deviceRank(properties.deviceType); rank > bestRank) {
            bestRank = rank;
            physicalDevice_ = candidate;
            properties_ = properties;
            queueFamilies_ = *families;
        }
    }

    if (physicalDevice_ == VK_NULL_HANDLE) {
        throw std::runtime_error(presenting()
            ? "no device can both run the pipeline and present to the window"
            : "no device with a compute queue found");
    }
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties_);
}

void VulkanContext::createDevice() {
    std::vector<std::uint32_t> families{queueFamilies_.compute};
    if (presenting() && queueFamilies_.present != queueFamilies_.compute) {
        families.push_back(queueFamilies_.present);
    }

    const float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    for (const std::uint32_t family : families) {
        queueInfos.push_back({
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = family,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        });
    }

    std::vector<const char*> extensions;
    if (presenting()) {
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    const VkDeviceCreateInfo deviceInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = static_cast<std::uint32_t>(queueInfos.size()),
        .pQueueCreateInfos = queueInfos.data(),
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
    vkCheck(vkCreateDevice(physicalDevice_, &deviceInfo, nullptr, device_.put({})),
            "vkCreateDevice");

    vkGetDeviceQueue(device_.get(), queueFamilies_.compute, 0, &computeQueue_);
    if (presenting()) {
        vkGetDeviceQueue(device_.get(), queueFamilies_.present, 0, &presentQueue_);
    }
}

void VulkanContext::createCommandPool() {
    const VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queueFamilies_.compute,
    };
    vkCheck(vkCreateCommandPool(device_.get(), &poolInfo, nullptr,
                                commandPool_.put(device_.get())),
            "vkCreateCommandPool");
    immediateCommands_ = allocateCommandBuffer();
}

std::uint32_t VulkanContext::findMemoryType(std::uint32_t typeBits,
                                            VkMemoryPropertyFlags required,
                                            VkMemoryPropertyFlags preferred) const {
    std::optional<std::uint32_t> fallback;
    for (std::uint32_t i = 0; i < memoryProperties_.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = memoryProperties_.memoryTypes[i].propertyFlags;
        if ((typeBits & (1u << i)) == 0 || (flags & required) != required) {
            continue;
        }
        if ((flags & preferred) == preferred) {
            return i;
        }
        if (!fallback) {
            fallback = i;
        }
    }
    if (!fallback) {
        throw std::runtime_error("no suitable memory type found");
    }
    return *fallback;
}

VkCommandBuffer VulkanContext::allocateCommandBuffer() const {
    const VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commandPool_.get(),
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer commands = VK_NULL_HANDLE;
    vkCheck(vkAllocateCommandBuffers(device_.get(), &allocInfo, &commands),
            "vkAllocateCommandBuffers");
    return commands;
}

VkCommandBuffer VulkanContext::beginImmediate() const {
    // The pool resets command buffers individually, so beginning the same buffer
    // again discards what the previous submission recorded.
    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkCheck(vkBeginCommandBuffer(immediateCommands_, &beginInfo), "vkBeginCommandBuffer");
    return immediateCommands_;
}

void VulkanContext::submitImmediate(VkCommandBuffer commands) const {
    vkCheck(vkEndCommandBuffer(commands), "vkEndCommandBuffer");

    const VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commands,
    };
    vkCheck(vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit");
    vkCheck(vkQueueWaitIdle(computeQueue_), "vkQueueWaitIdle");
}

} // namespace vkp
