#include "VulkanContext.hpp"

#include <cstring>
#include <set>
#include <vector>

#include "../util/Log.hpp"

namespace vkp {

namespace {
const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

bool validationLayerAvailable() {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& layer : layers) {
        if (std::strcmp(layer.layerName, kValidationLayer) == 0) {
            return true;
        }
    }
    return false;
}
} // namespace

VulkanContext::VulkanContext(bool enableValidation) : VulkanContext(ContextConfig{enableValidation, false, {}, {}}) {}

VulkanContext::VulkanContext(const ContextConfig& config) : config_(config) {
    createInstance();
    if (config_.enablePresent && config_.surfaceFactory) {
        surface_ = config_.surfaceFactory(instance_);
        hasPresent_ = surface_ != VK_NULL_HANDLE;
    }
    selectPhysicalDevice();
    createLogicalDevice();
    createCommandPool();
    logInfo(std::string("using device: ") + deviceName());
}

VulkanContext::~VulkanContext() {
    if (commandPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, commandPool_, nullptr);
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

void VulkanContext::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "vkPhoton";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "vkPhoton";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    std::vector<const char*> layers;
    if (config_.enableValidation && validationLayerAvailable()) {
        layers.push_back(kValidationLayer);
        logInfo("validation layer enabled");
    } else if (config_.enableValidation) {
        logInfo("validation layer requested but not available, continuing without it");
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(config_.instanceExtensions.size());
    createInfo.ppEnabledExtensionNames =
        config_.instanceExtensions.empty() ? nullptr : config_.instanceExtensions.data();

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance_));
}

bool VulkanContext::familySupportsPresent(VkPhysicalDevice device, uint32_t family) const {
    if (surface_ == VK_NULL_HANDLE) {
        return false;
    }
    VkBool32 supported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, family, surface_, &supported);
    return supported == VK_TRUE;
}

void VulkanContext::selectPhysicalDevice() {
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
    if (count == 0) {
        logError("no Vulkan capable device found");
        std::abort();
    }
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));

    // A device is usable if it has a compute queue, and when presenting, a queue
    // that can present to our surface. Discrete GPUs are preferred.
    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    for (VkPhysicalDevice candidate : devices) {
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());

        bool hasCompute = false;
        bool hasPresent = false;
        for (uint32_t i = 0; i < familyCount; ++i) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                hasCompute = true;
            }
            if (hasPresent_ && familySupportsPresent(candidate, i)) {
                hasPresent = true;
            }
        }
        if (!hasCompute || (hasPresent_ && !hasPresent)) {
            continue;
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physicalDevice_ = candidate;
            deviceProperties_ = props;
            return;
        }
        if (fallback == VK_NULL_HANDLE) {
            fallback = candidate;
        }
    }

    if (fallback == VK_NULL_HANDLE) {
        logError("no device satisfies the required queues");
        std::abort();
    }
    physicalDevice_ = fallback;
    vkGetPhysicalDeviceProperties(physicalDevice_, &deviceProperties_);
}

void VulkanContext::createLogicalDevice() {
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &familyCount, families.data());

    // Choose a compute family. Prefer one that can also present, so a single
    // queue serves both roles on the common hardware where that is possible.
    bool computeFound = false;
    for (uint32_t i = 0; i < familyCount; ++i) {
        const bool compute = (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        if (compute && (!hasPresent_ || familySupportsPresent(physicalDevice_, i))) {
            computeQueueFamily_ = i;
            computeFound = true;
            break;
        }
    }
    if (!computeFound) {
        for (uint32_t i = 0; i < familyCount; ++i) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                computeQueueFamily_ = i;
                computeFound = true;
                break;
            }
        }
    }
    if (!computeFound) {
        logError("compute queue family disappeared during device creation");
        std::abort();
    }

    presentQueueFamily_ = computeQueueFamily_;
    if (hasPresent_ && !familySupportsPresent(physicalDevice_, computeQueueFamily_)) {
        for (uint32_t i = 0; i < familyCount; ++i) {
            if (familySupportsPresent(physicalDevice_, i)) {
                presentQueueFamily_ = i;
                break;
            }
        }
    }

    std::set<uint32_t> uniqueFamilies{computeQueueFamily_};
    if (hasPresent_) {
        uniqueFamilies.insert(presentQueueFamily_);
    }

    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    for (uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;
        queueInfos.push_back(queueInfo);
    }

    std::vector<const char*> deviceExtensions;
    if (hasPresent_) {
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.empty() ? nullptr : deviceExtensions.data();

    VK_CHECK(vkCreateDevice(physicalDevice_, &deviceInfo, nullptr, &device_));
    vkGetDeviceQueue(device_, computeQueueFamily_, 0, &computeQueue_);
    if (hasPresent_) {
        vkGetDeviceQueue(device_, presentQueueFamily_, 0, &presentQueue_);
    }
}

void VulkanContext::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = computeQueueFamily_;
    VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_));
}

uint32_t VulkanContext::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags) const {
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        const bool typeAllowed = (typeBits & (1u << i)) != 0;
        const bool hasFlags = (memProps.memoryTypes[i].propertyFlags & flags) == flags;
        if (typeAllowed && hasFlags) {
            return i;
        }
    }
    logError("no suitable memory type found");
    std::abort();
}

void VulkanContext::runCommands(const std::function<void(VkCommandBuffer)>& recorder) const {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device_, &allocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    recorder(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VK_CHECK(vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(computeQueue_));

    vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
}

} // namespace vkp
