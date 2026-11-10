#pragma once

#include <cstdint>

namespace vkp {

// Layout must match the Detection struct declared in shaders/common.glsl.
// std430 packs these four 4 byte members with no padding, so a plain struct
// with 32 bit types maps one to one onto the GPU buffer.
struct Detection {
    std::uint32_t x;
    std::uint32_t y;
    float diffValue;
    float snr;
};

static_assert(sizeof(Detection) == 16, "Detection must match the std430 layout in common.glsl");

} // namespace vkp
