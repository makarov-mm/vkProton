#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#include <vulkan/vulkan.h>

namespace vkp {

inline void logInfo(const std::string& message) {
    std::cout << "[vkPhoton] " << message << "\n";
}

inline void logError(const std::string& message) {
    std::cerr << "[vkPhoton] error: " << message << "\n";
}

inline const char* vkResultName(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        default: return "VK_ERROR_UNKNOWN";
    }
}

} // namespace vkp

// Aborts with a readable message when a Vulkan call does not return VK_SUCCESS.
#define VK_CHECK(call)                                                          \
    do {                                                                       \
        VkResult vkCheckResult = (call);                                       \
        if (vkCheckResult != VK_SUCCESS) {                                     \
            vkp::logError(std::string(#call) + " returned " +                  \
                          vkp::vkResultName(vkCheckResult));                   \
            std::abort();                                                      \
        }                                                                      \
    } while (false)
