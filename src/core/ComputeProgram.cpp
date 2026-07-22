#include "ComputeProgram.hpp"

#include <fstream>
#include <vector>

#include "Buffer.hpp"
#include "VulkanContext.hpp"
#include "../util/Log.hpp"

namespace vkp {

ComputeProgram::ComputeProgram(const VulkanContext& context,
                               const std::string& spirvPath,
                               uint32_t bindingCount,
                               uint32_t pushConstantSize)
    : context_(context), bindingCount_(bindingCount), pushConstantSize_(pushConstantSize) {
    shaderModule_ = loadShaderModule(spirvPath);

    // Descriptor set layout: one storage buffer per binding, all in the compute stage.
    std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount_);
    for (uint32_t i = 0; i < bindingCount_; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindingCount_;
    layoutInfo.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(context_.device(), &layoutInfo, nullptr, &setLayout_));

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = pushConstantSize_;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &setLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = pushConstantSize_ > 0 ? 1 : 0;
    pipelineLayoutInfo.pPushConstantRanges = pushConstantSize_ > 0 ? &pushRange : nullptr;
    VK_CHECK(vkCreatePipelineLayout(context_.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout_));

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule_;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout_;
    VK_CHECK(vkCreateComputePipelines(context_.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_));

    // One descriptor set drawn from a pool sized to this program alone.
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = bindingCount_;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(context_.device(), &poolInfo, nullptr, &descriptorPool_));

    VkDescriptorSetAllocateInfo setAlloc{};
    setAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAlloc.descriptorPool = descriptorPool_;
    setAlloc.descriptorSetCount = 1;
    setAlloc.pSetLayouts = &setLayout_;
    VK_CHECK(vkAllocateDescriptorSets(context_.device(), &setAlloc, &descriptorSet_));
}

ComputeProgram::~ComputeProgram() {
    VkDevice device = context_.device();
    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool_, nullptr);
    }
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipeline_, nullptr);
    }
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipelineLayout_, nullptr);
    }
    if (setLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, setLayout_, nullptr);
    }
    if (shaderModule_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, shaderModule_, nullptr);
    }
}

VkShaderModule ComputeProgram::loadShaderModule(const std::string& spirvPath) const {
    std::ifstream file(spirvPath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        logError("cannot open shader file: " + spirvPath);
        std::abort();
    }
    size_t byteCount = static_cast<size_t>(file.tellg());
    std::vector<char> code(byteCount);
    file.seekg(0);
    file.read(code.data(), static_cast<std::streamsize>(byteCount));
    file.close();

    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = byteCount;
    moduleInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(context_.device(), &moduleInfo, nullptr, &module));
    return module;
}

void ComputeProgram::bindBuffers(const std::vector<Buffer*>& buffers) {
    if (buffers.size() != bindingCount_) {
        logError("bindBuffers called with the wrong buffer count");
        std::abort();
    }

    std::vector<VkDescriptorBufferInfo> infos(bindingCount_);
    std::vector<VkWriteDescriptorSet> writes(bindingCount_);
    for (uint32_t i = 0; i < bindingCount_; ++i) {
        infos[i].buffer = buffers[i]->handle();
        infos[i].offset = 0;
        infos[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptorSet_;
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(context_.device(), bindingCount_, writes.data(), 0, nullptr);
}

void ComputeProgram::recordDispatch(VkCommandBuffer cmd,
                                    uint32_t groupsX,
                                    uint32_t groupsY,
                                    const void* pushConstants) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
                            0, 1, &descriptorSet_, 0, nullptr);
    if (pushConstantSize_ > 0 && pushConstants != nullptr) {
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, pushConstantSize_, pushConstants);
    }
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
}

} // namespace vkp
