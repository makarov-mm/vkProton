#pragma once

#include <cstddef>

#include <vulkan/vulkan.h>

namespace vkp {

class VulkanContext;

// A single storage buffer plus its backing memory. Two flavours are used:
//   host visible, for staging uploads and reading results back
//   device local, for the buffers the compute shaders actually touch
// One allocation per buffer is not how a production engine manages memory, but
// at this scale it keeps ownership obvious and the code readable.
class Buffer {
public:
    Buffer(const VulkanContext& context,
           VkDeviceSize size,
           VkBufferUsageFlags usage,
           VkMemoryPropertyFlags memoryFlags);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    VkBuffer handle() const { return buffer_; }
    VkDeviceSize size() const { return size_; }

    // Valid only for host visible buffers. Returns a persistent mapping.
    void* map();
    void unmap();

    // Copies raw bytes into a host visible buffer through the persistent mapping.
    void upload(const void* data, VkDeviceSize bytes);

    // Copies raw bytes out of a host visible buffer into a caller provided block.
    void download(void* data, VkDeviceSize bytes);

private:
    const VulkanContext& context_;
    VkDeviceSize size_ = 0;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    void* mapped_ = nullptr;
    bool hostVisible_ = false;
};

} // namespace vkp
