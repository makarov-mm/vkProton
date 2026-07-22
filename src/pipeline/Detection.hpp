#pragma once

#include <cstdint>

namespace vkp {

// Layout must match the Detection struct declared in detect.comp.
// std430 packs these four 4 byte members with no padding, so a plain struct
// with 32 bit types maps one to one onto the GPU buffer.
struct Detection {
    uint32_t x;
    uint32_t y;
    float diffValue;
    float snr;
};

} // namespace vkp
