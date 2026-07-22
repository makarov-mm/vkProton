#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace vkp {

class VulkanContext;
class Buffer;

// Wraps everything needed to run one compute shader over a set of storage
// buffers: the shader module, a descriptor set layout of storage buffer
// bindings, a pipeline layout carrying a push constant range, the compute
// pipeline itself and a single descriptor set. Each stage of the pipeline owns
// one of these.
class ComputeProgram {
public:
    ComputeProgram(const VulkanContext& context,
                   const std::string& spirvPath,
                   uint32_t bindingCount,
                   uint32_t pushConstantSize);
    ~ComputeProgram();

    ComputeProgram(const ComputeProgram&) = delete;
    ComputeProgram& operator=(const ComputeProgram&) = delete;

    // Points the descriptor set at the given buffers, one per binding in order.
    void bindBuffers(const std::vector<Buffer*>& buffers);

    // Records a bind and a 2D dispatch into an already open command buffer.
    void recordDispatch(VkCommandBuffer cmd,
                        uint32_t groupsX,
                        uint32_t groupsY,
                        const void* pushConstants) const;

private:
    VkShaderModule loadShaderModule(const std::string& spirvPath) const;

    const VulkanContext& context_;
    uint32_t bindingCount_ = 0;
    uint32_t pushConstantSize_ = 0;

    VkShaderModule shaderModule_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
};

} // namespace vkp
