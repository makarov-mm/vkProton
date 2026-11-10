#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace vkp {

// A single injected transient with its ground truth position and brightness.
// The pipeline never sees this. It exists so the demo can report how many real
// events the detector recovered.
struct TransientTruth {
    std::uint32_t x;
    std::uint32_t y;
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
    // Smallest frame side the generator supports. Stars and transients keep a few
    // pixels clear of the edges, so anything smaller leaves no room to place them.
    static constexpr std::uint32_t kMinSide = 16;

    FrameSource(std::uint32_t width, std::uint32_t height, std::uint32_t seed);

    // Master calibration frames, constant across the run.
    [[nodiscard]] std::span<const float> biasField() const noexcept { return bias_; }
    [[nodiscard]] std::span<const float> flatInvField() const noexcept { return flatInv_; }

    // Fills raw with the sensor readout for the given frame index and records
    // any transients injected into that frame.
    void generateRaw(std::uint32_t frameIndex, std::vector<float>& raw);

    [[nodiscard]] std::span<const TransientTruth> lastTransients() const noexcept {
        return lastTransients_;
    }

private:
    void buildCalibration();
    void buildStars();
    // Adds a point source with the detector's point spread function to an image.
    void addGaussian(std::span<float> image, float cx, float cy, float flux) const;

    static constexpr float kSkyLevel = 100.0f;
    static constexpr float kBiasLevel = 500.0f;
    static constexpr float kReadNoise = 5.0f;
    static constexpr float kPsfSigma = 1.3f;

    std::uint32_t width_;
    std::uint32_t height_;
    std::uint32_t seed_;

    std::vector<float> bias_;
    std::vector<float> flat_;
    std::vector<float> flatInv_;
    std::vector<float> trueSignal_;
    std::vector<TransientTruth> lastTransients_;
};

} // namespace vkp
