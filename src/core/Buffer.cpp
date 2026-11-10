#include "Buffer.hpp"

#include <cstring>
#include <stdexcept>

#include "VulkanContext.hpp"
#include "VulkanUtil.hpp"

namespace vkp {

namespace {

struct Traits {
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags requiredMemory;
    VkMemoryPropertyFlags preferredMemory;
};

Traits traitsFor(Buffer::Kind kind) {
    constexpr VkMemoryPropertyFlags hostVisible =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    switch (kind) {
        case Buffer::Kind::Upload:
            return {VK_BUFFER_USAGE_TRANSFER_SRC_BIT, hostVisible, hostVisible};
        case Buffer::Kind::Readback:
            // Cached memory keeps host reads fast. Without it the mapping can be
            // write combined, which is fine to fill but very slow to read back.
            return {VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostVisible,
                    hostVisible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT};
        case Buffer::Kind::Storage:
            break;
    }
    return {VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
}

} // namespace

Buffer::Buffer(const VulkanContext& context, VkDeviceSize size, Kind kind) : size_(size) {
    const VkDevice device = context.device();
    const Traits traits = traitsFor(kind);

    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size_,
        .usage = traits.usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    vkCheck(vkCreateBuffer(device, &bufferInfo, nullptr, buffer_.put(device)), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer_.get(), &requirements);

    const VkMemoryAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits,
                                                  traits.requiredMemory, traits.preferredMemory),
    };
    vkCheck(vkAllocateMemory(device, &allocInfo, nullptr, memory_.put(device)),
            "vkAllocateMemory");
    vkCheck(vkBindBufferMemory(device, buffer_.get(), memory_.get(), 0), "vkBindBufferMemory");

    if (kind != Kind::Storage) {
        vkCheck(vkMapMemory(device, memory_.get(), 0, VK_WHOLE_SIZE, 0, &mapped_), "vkMapMemory");
    }
}

void Buffer::writeBytes(std::span<const std::byte> source) {
    if (mapped_ == nullptr) {
        throw std::logic_error("write called on a buffer that is not host visible");
    }
    if (source.size() > size_) {
        throw std::length_error("write exceeds the buffer size");
    }
    std::memcpy(mapped_, source.data(), source.size());
}

const void* Buffer::mappedBytes(std::size_t byteCount) const {
    if (mapped_ == nullptr) {
        throw std::logic_error("view called on a buffer that is not host visible");
    }
    if (byteCount > size_) {
        throw std::length_error("view exceeds the buffer size");
    }
    return mapped_;
}

} // namespace vkp
