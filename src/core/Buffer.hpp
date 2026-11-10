#pragma once

#include <cstddef>
#include <ranges>
#include <span>

#include <vulkan/vulkan.h>

#include "Handle.hpp"

namespace vkp {

class VulkanContext;

// A single buffer plus its backing memory. One allocation per buffer is not how a
// production engine manages memory, but at this scale it keeps ownership obvious
// and the code readable.
class Buffer {
public:
    enum class Kind {
        Storage,   // device local, bound to the compute shaders and usable in copies
        Upload,    // host visible staging memory: the CPU writes, the GPU copies out
        Readback,  // host visible result memory: the GPU copies in, the CPU reads
    };

    Buffer(const VulkanContext& context, VkDeviceSize size, Kind kind);

    [[nodiscard]] VkBuffer handle() const noexcept { return buffer_.get(); }
    [[nodiscard]] VkDeviceSize size() const noexcept { return size_; }

    // Copies a contiguous range of trivially copyable values to the start of a host
    // visible buffer.
    template <std::ranges::contiguous_range Range>
    void write(const Range& source) {
        writeBytes(std::as_bytes(std::span(source)));
    }

    // Views the first count values of a host visible buffer. Host visible buffers
    // stay mapped for their whole life, so the view remains valid and always shows
    // what the GPU copied there last.
    template <typename T>
    [[nodiscard]] std::span<const T> view(std::size_t count) const {
        return {static_cast<const T*>(mappedBytes(count * sizeof(T))), count};
    }

private:
    void writeBytes(std::span<const std::byte> source);
    [[nodiscard]] const void* mappedBytes(std::size_t byteCount) const;

    VkDeviceSize size_;
    // The memory is declared first so it outlives the buffer bound to it.
    MemoryHandle memory_;
    BufferHandle buffer_;
    void* mapped_ = nullptr;
};

} // namespace vkp
