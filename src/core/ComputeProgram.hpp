#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>

#include "Handle.hpp"

namespace vkp {

class VulkanContext;
class Buffer;

// Wraps everything needed to run one compute shader over a set of storage buffers:
// a descriptor set layout of storage buffer bindings, a pipeline layout carrying a
// push constant range, the compute pipeline itself and its descriptor sets. Each
// stage of the pipeline owns one of these.
class ComputeProgram {
public:
    // setCount descriptor sets are allocated, so a stage that alternates between
    // buffers can keep one set per arrangement and never rewrite them.
    ComputeProgram(const VulkanContext& context,
                   const std::filesystem::path& spirvPath,
                   std::uint32_t bindingCount,
                   std::uint32_t pushConstantSize,
                   std::uint32_t setCount = 1);

    // Points one descriptor set at the given buffers, one per binding in order.
    void bindBuffers(std::uint32_t setIndex, std::initializer_list<const Buffer*> buffers);

    // Records a bind and a 2D dispatch into an already open command buffer.
    void dispatch(VkCommandBuffer commands,
                  std::uint32_t setIndex,
                  std::uint32_t groupsX,
                  std::uint32_t groupsY,
                  std::span<const std::byte> pushConstants) const;

private:
    VkDevice device_;
    std::uint32_t bindingCount_;
    std::uint32_t pushConstantSize_;

    DescriptorSetLayoutHandle setLayout_;
    PipelineLayoutHandle pipelineLayout_;
    PipelineHandle pipeline_;
    DescriptorPoolHandle descriptorPool_;
    std::vector<VkDescriptorSet> descriptorSets_;  // freed together with the pool
};

} // namespace vkp
