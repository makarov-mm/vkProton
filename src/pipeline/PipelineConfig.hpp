#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "Detection.hpp"

namespace vkp {

// Parameters shared by the GPU pipeline and its CPU reference.
struct PipelineConfig {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    int annulusOuter = 7;       // outer radius of the noise annulus in pixels
    int annulusInner = 4;       // inner radius excluded from the annulus
    float detectSigma = 6.0f;   // detection threshold in units of local difference noise
    float shapeRatio = 0.35f;   // minimum neighbour to centre ratio, zero disables the test
    std::uint32_t maxDetections = 8192;
};

struct FrameResult {
    // Detections in row major order, capped at config.maxDetections.
    std::vector<Detection> detections;
    // Total number of detections found, which can exceed the cap.
    std::uint32_t reportedCount = 0;
    // The calibrated pixels of this frame, kept for visualisation. Empty when the
    // caller did not ask for them. The view is only valid until the next frame is
    // processed.
    std::span<const float> calibrated;
};

} // namespace vkp
