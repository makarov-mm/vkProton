#include "Buffer.hpp"

#include <cstring>

#include "VulkanContext.hpp"
#include "../util/Log.hpp"

namespace vkp {

Buffer::Buffer(const VulkanContext& context,
               VkDeviceSize size,
               VkBufferUsageFlags usage,
               VkMemoryPropertyFlags memoryFlags)
    : context_(context), size_(size) {
    hostVisible_ = (memoryFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size_;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(context_.device(), &bufferInfo, nullptr, &buffer_));

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(context_.device(), buffer_, &requirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = requirements.size;
    allocInfo.memoryTypeIndex = context_.findMemoryType(requirements.memoryTypeBits, memoryFlags);
    VK_CHECK(vkAllocateMemory(context_.device(), &allocInfo, nullptr, &memory_));

    VK_CHECK(vkBindBufferMemory(context_.device(), buffer_, memory_, 0));
}

Buffer::~Buffer() {
    if (mapped_ != nullptr) {
        vkUnmapMemory(context_.device(), memory_);
        mapped_ = nullptr;
    }
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(context_.device(), buffer_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(context_.device(), memory_, nullptr);
    }
}

void* Buffer::map() {
    if (!hostVisible_) {
        logError("map called on a device local buffer");
        std::abort();
    }
    if (mapped_ == nullptr) {
        VK_CHECK(vkMapMemory(context_.device(), memory_, 0, size_, 0, &mapped_));
    }
    return mapped_;
}

void Buffer::unmap() {
    if (mapped_ != nullptr) {
        vkUnmapMemory(context_.device(), memory_);
        mapped_ = nullptr;
    }
}

void Buffer::upload(const void* data, VkDeviceSize bytes) {
    void* dst = map();
    std::memcpy(dst, data, static_cast<size_t>(bytes));
}

void Buffer::download(void* data, VkDeviceSize bytes) {
    void* src = map();
    std::memcpy(data, src, static_cast<size_t>(bytes));
}

} // namespace vkp
