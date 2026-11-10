// Declarations shared by all four compute stages. Every stage pulls this file in with #include,
// so the work group size, the push constant block and the detection record are written once and
// cannot drift apart. The C++ side mirrors them in src/pipeline/PushConstants.hpp and
// src/pipeline/Detection.hpp.

// Frames are processed in 16x16 pixel tiles. kTileSize in TransientPipeline.cpp must match.
layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform Push {
    uint  width;
    uint  height;
    int   annulusOuter;   // outer radius of the noise annulus in pixels
    int   annulusInner;   // inner radius that is excluded from the annulus
    float detectSigma;    // detection threshold in units of local difference noise
    float shapeRatio;     // minimum neighbour to centre ratio, zero disables the test
    uint  maxDetections;  // capacity of the detection list
    uint  hasPrev;        // 1 when a previous frame is available for differencing
} pc;

struct Detection {
    uint  x;
    uint  y;
    float diffValue;
    float snr;
};
