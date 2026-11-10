#pragma once

#include <cstdint>

namespace vkp {

// Shared by all four compute stages. The shaders get the same block from
// shaders/common.glsl, so a single 32 byte range is bound once and reused. Keeping
// one struct avoids drift between stages and stays well under the 128 byte
// guaranteed limit.
//
// Noise is estimated on the difference image over an annulus: a ring that skips
// the central core so a compact transient never corrupts the estimate of its own
// local noise. The shape ratio is a cheap point spread function test that rejects
// single pixel spikes.
struct PushConstants {
    std::uint32_t width;
    std::uint32_t height;
    std::int32_t annulusOuter;    // outer radius of the noise annulus in pixels
    std::int32_t annulusInner;    // inner radius that is excluded from the annulus
    float detectSigma;            // detection threshold in units of local difference noise
    float shapeRatio;             // minimum neighbour to centre ratio, zero disables the test
    std::uint32_t maxDetections;  // capacity of the detection list
    std::uint32_t hasPrev;        // 1 when a previous frame is available for differencing
};

static_assert(sizeof(PushConstants) == 32, "PushConstants must match the block in common.glsl");

} // namespace vkp
