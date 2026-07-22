#pragma once

#include <cstdint>
#include <vector>

namespace vkp {

// A single injected transient with its ground truth position and brightness.
// The pipeline never sees this. It exists so the demo can report how many real
// events the detector recovered.
struct TransientTruth {
    uint32_t x;
    uint32_t y;
    float flux;
};

// Produces a stream of synthetic telescope frames. Each frame is a static star
// field observed through a fixed flat field and bias pattern, with photon and
// read noise added, and occasionally a transient point source injected. The
// generator also exposes the master bias and inverse flat field so the pipeline
// has something real to calibrate against.
//
// Everything is deterministic given the seed, so a run is fully reproducible.
class FrameSource {
public:
    FrameSource(uint32_t width, uint32_t height, uint32_t seed);

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    size_t pixelCount() const { return static_cast<size_t>(width_) * height_; }

    // Master calibration frames, constant across the run.
    const std::vector<float>& biasField() const { return bias_; }
    const std::vector<float>& flatInvField() const { return flatInv_; }

    // Fills raw with the sensor readout for the given frame index and records
    // any transients injected into that frame.
    void generateRaw(uint32_t frameIndex, std::vector<float>& raw);

    const std::vector<TransientTruth>& lastTransients() const { return lastTransients_; }

private:
    void buildStars(uint32_t seed);
    void buildCalibration(uint32_t seed);
    void addGaussian(std::vector<float>& image, float cx, float cy, float flux, float sigma) const;

    struct Star {
        float x;
        float y;
        float flux;
    };

    uint32_t width_;
    uint32_t height_;
    uint32_t seed_;

    float skyLevel_ = 100.0f;
    float biasLevel_ = 500.0f;
    float readNoise_ = 5.0f;
    float psfSigma_ = 1.3f;

    std::vector<Star> stars_;
    std::vector<float> bias_;
    std::vector<float> flat_;
    std::vector<float> flatInv_;
    std::vector<float> trueSignal_;
    std::vector<TransientTruth> lastTransients_;
};

} // namespace vkp
