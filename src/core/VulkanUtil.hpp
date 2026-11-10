#pragma once

#include <concepts>
#include <cstdint>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

namespace vkp {

// Symbolic name of a result code, for error messages.
[[nodiscard]] std::string_view resultName(VkResult result) noexcept;

// Throws std::runtime_error naming the operation and the result code.
[[noreturn]] void throwVulkanError(VkResult result, std::string_view operation);

// Throws unless a Vulkan call returned VK_SUCCESS.
inline void vkCheck(VkResult result, std::string_view operation) {
    if (result != VK_SUCCESS) {
        throwVulkanError(result, operation);
    }
}

// Wraps Vulkan's two call enumeration idiom: ask for the count, then fill an array
// of that size. The query receives (uint32_t* count, T* items) and makes the call.
template <typename T, std::invocable<std::uint32_t*, T*> Query>
[[nodiscard]] std::vector<T> enumerate(Query&& query) {
    std::uint32_t count = 0;
    query(&count, static_cast<T*>(nullptr));
    std::vector<T> items(count);
    if (count > 0) {
        query(&count, items.data());
        items.resize(count);
    }
    return items;
}

} // namespace vkp
