#include "ComputeProgram.hpp"

#include <format>
#include <fstream>
#include <stdexcept>

#include "Buffer.hpp"
#include "VulkanContext.hpp"
#include "VulkanUtil.hpp"

namespace vkp {

namespace {

constexpr std::uint32_t kSpirvMagic = 0x07230203;

// Reads a compiled shader as the stream of 32 bit words Vulkan expects.
std::vector<std::uint32_t> readSpirv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error(std::format("cannot open shader file {}", path.string()));
    }

    constexpr auto wordSize = static_cast<std::streamoff>(sizeof(std::uint32_t));
    const std::streamoff byteCount = file.tellg();
    std::vector<std::uint32_t> words;
    if (byteCount > 0 && byteCount % wordSize == 0) {
        words.resize(static_cast<std::size_t>(byteCount / wordSize));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(words.data()), byteCount);
    }

    if (words.empty() || !file || words.front() != kSpirvMagic) {
        throw std::runtime_error(std::format("{} is not a SPIR-V module", path.string()));
    }
    return words;
}

ShaderModuleHandle loadShader(VkDevice device, const std::filesystem::path& path) {
    const std::vector<std::uint32_t> code = readSpirv(path);
    const VkShaderModuleCreateInfo shaderInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size() * sizeof(std::uint32_t),
        .pCode = code.data(),
    };
    ShaderModuleHandle shader;
    vkCheck(vkCreateShaderModule(device, &shaderInfo, nullptr, shader.put(device)),
            "vkCreateShaderModule");
    return shader;
}

} // namespace

ComputeProgram::ComputeProgram(const VulkanContext& context,
                               const std::filesystem::path& spirvPath,
                               std::uint32_t bindingCount,
                               std::uint32_t pushConstantSize,
                               std::uint32_t setCount)
    : device_(context.device()), bindingCount_(bindingCount), pushConstantSize_(pushConstantSize) {
    // Descriptor set layout: one storage buffer per binding, all in the compute stage.
    std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount_);
    for (std::uint32_t i = 0; i < bindingCount_; ++i) {
        bindings[i] = {
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = bindingCount_,
        .pBindings = bindings.data(),
    };
    vkCheck(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, setLayout_.put(device_)),
            "vkCreateDescriptorSetLayout");

    const VkDescriptorSetLayout setLayout = setLayout_.get();
    const VkPushConstantRange pushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = pushConstantSize_,
    };
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &setLayout,
        .pushConstantRangeCount = pushConstantSize_ > 0 ? 1u : 0u,
        .pPushConstantRanges = &pushRange,
    };
    vkCheck(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr,
                                   pipelineLayout_.put(device_)),
            "vkCreatePipelineLayout");

    // The shader module is only needed while the pipeline is being built.
    const ShaderModuleHandle shader = loadShader(device_, spirvPath);
    const VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader.get(),
            .pName = "main",
        },
        .layout = pipelineLayout_.get(),
    };
    vkCheck(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                     pipeline_.put(device_)),
            "vkCreateComputePipelines");

    // Descriptor sets come from a pool sized to this program alone.
    const VkDescriptorPoolSize poolSize{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = bindingCount_ * setCount,
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = setCount,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    vkCheck(vkCreateDescriptorPool(device_, &poolInfo, nullptr, descriptorPool_.put(device_)),
            "vkCreateDescriptorPool");

    const std::vector<VkDescriptorSetLayout> setLayouts(setCount, setLayout);
    const VkDescriptorSetAllocateInfo setInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool_.get(),
        .descriptorSetCount = setCount,
        .pSetLayouts = setLayouts.data(),
    };
    descriptorSets_.resize(setCount);
    vkCheck(vkAllocateDescriptorSets(device_, &setInfo, descriptorSets_.data()),
            "vkAllocateDescriptorSets");
}

void ComputeProgram::bindBuffers(std::uint32_t setIndex,
                                 std::initializer_list<const Buffer*> buffers) {
    if (buffers.size() != bindingCount_) {
        throw std::invalid_argument("bindBuffers called with the wrong buffer count");
    }

    std::vector<VkDescriptorBufferInfo> infos;
    infos.reserve(buffers.size());
    for (const Buffer* buffer : buffers) {
        infos.push_back({.buffer = buffer->handle(), .offset = 0, .range = VK_WHOLE_SIZE});
    }

    std::vector<VkWriteDescriptorSet> writes(bindingCount_);
    for (std::uint32_t i = 0; i < bindingCount_; ++i) {
        writes[i] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSets_.at(setIndex),
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[i],
        };
    }
    vkUpdateDescriptorSets(device_, bindingCount_, writes.data(), 0, nullptr);
}

void ComputeProgram::dispatch(VkCommandBuffer commands,
                              std::uint32_t setIndex,
                              std::uint32_t groupsX,
                              std::uint32_t groupsY,
                              std::span<const std::byte> pushConstants) const {
    if (pushConstants.size() != pushConstantSize_) {
        throw std::invalid_argument("push constant data does not match the declared size");
    }

    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.get());
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_.get(), 0, 1,
                            &descriptorSets_.at(setIndex), 0, nullptr);
    if (pushConstantSize_ > 0) {
        vkCmdPushConstants(commands, pipelineLayout_.get(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           pushConstantSize_, pushConstants.data());
    }
    vkCmdDispatch(commands, groupsX, groupsY, 1);
}

} // namespace vkp
