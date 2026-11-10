#include "FrameSource.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <stdexcept>

namespace vkp {

namespace {

// Cheap deterministic per frame seed so noise and transients are reproducible
// without carrying any state between calls.
std::uint32_t mixSeed(std::uint32_t base, std::uint32_t frame) {
    std::uint32_t h = base * 2654435761u;
    h ^= (frame + 0x9e3779b9u + (h << 6) + (h >> 2));
    return h;
}

} // namespace

FrameSource::FrameSource(std::uint32_t width, std::uint32_t height, std::uint32_t seed)
    : width_(width), height_(height), seed_(seed) {
    if (width_ < kMinSide || height_ < kMinSide) {
        throw std::invalid_argument("frame is too small for the synthetic sky");
    }
    const std::size_t pixelCount = static_cast<std::size_t>(width_) * height_;
    bias_.resize(pixelCount);
    flat_.resize(pixelCount);
    flatInv_.resize(pixelCount);
    trueSignal_.resize(pixelCount);
    buildCalibration();
    buildStars();
}

void FrameSource::buildCalibration() {
    std::mt19937 rng(seed_ ^ 0x1234abcdu);
    std::uniform_real_distribution<float> jitter(-2.0f, 2.0f);

    // Bias: a constant offset with a mild diagonal gradient and per pixel jitter,
    // the kind of fixed pattern a real detector shows.
    // Flat: a smooth vignette that dims the corners, values around 0.75 to 1.05.
    for (std::uint32_t y = 0; y < height_; ++y) {
        for (std::uint32_t x = 0; x < width_; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * width_ + x;

            const float gx = static_cast<float>(x) / static_cast<float>(width_);
            const float gy = static_cast<float>(y) / static_cast<float>(height_);
            bias_[i] = kBiasLevel + 20.0f * (gx + gy) + jitter(rng);

            const float dx = gx - 0.5f;
            const float dy = gy - 0.5f;
            const float radius2 = dx * dx + dy * dy;
            flat_[i] = std::max(1.05f - 0.6f * radius2, 0.5f);
            flatInv_[i] = 1.0f / flat_[i];
        }
    }
}

void FrameSource::buildStars() {
    std::mt19937 rng(seed_ ^ 0x55aa55aau);
    std::uniform_real_distribution<float> posX(4.0f, static_cast<float>(width_) - 5.0f);
    std::uniform_real_distribution<float> posY(4.0f, static_cast<float>(height_) - 5.0f);
    std::uniform_real_distribution<float> flux(400.0f, 6000.0f);

    // The static true signal is the sky plus every star. It never changes, so
    // difference imaging cancels it perfectly apart from noise. Roughly one star
    // per 800 pixels gives a believable but not crowded field.
    std::ranges::fill(trueSignal_, kSkyLevel);
    const std::size_t starCount = std::max<std::size_t>(20, trueSignal_.size() / 800);
    for (std::size_t s = 0; s < starCount; ++s) {
        const float x = posX(rng);
        const float y = posY(rng);
        const float starFlux = flux(rng);
        addGaussian(trueSignal_, x, y, starFlux);
    }
}

void FrameSource::addGaussian(std::span<float> image, float cx, float cy, float flux) const {
    const int radius = static_cast<int>(std::ceil(kPsfSigma * 3.5f));
    const float norm = flux / (2.0f * std::numbers::pi_v<float> * kPsfSigma * kPsfSigma);
    const float inv2s2 = 1.0f / (2.0f * kPsfSigma * kPsfSigma);

    // Only the part of the footprint that lies inside the frame is visited.
    const int ix = static_cast<int>(std::lround(cx));
    const int iy = static_cast<int>(std::lround(cy));
    const int x0 = std::max(ix - radius, 0);
    const int x1 = std::min(ix + radius, static_cast<int>(width_) - 1);
    const int y0 = std::max(iy - radius, 0);
    const int y1 = std::min(iy + radius, static_cast<int>(height_) - 1);

    for (int y = y0; y <= y1; ++y) {
        const float fy = static_cast<float>(y) - cy;
        float* const row = image.data() + static_cast<std::size_t>(y) * width_;
        for (int x = x0; x <= x1; ++x) {
            const float fx = static_cast<float>(x) - cx;
            row[x] += norm * std::exp(-(fx * fx + fy * fy) * inv2s2);
        }
    }
}

void FrameSource::generateRaw(std::uint32_t frameIndex, std::vector<float>& raw) {
    raw.assign(trueSignal_.begin(), trueSignal_.end());
    lastTransients_.clear();

    std::mt19937 rng(mixSeed(seed_, frameIndex));

    // Inject a transient on roughly one frame in three, at a fresh location.
    // A moving object would also work, but a flash is the cleanest thing to see
    // in a single difference image.
    std::uniform_int_distribution<int> chance(0, 2);
    if (frameIndex > 0 && chance(rng) == 0) {
        std::uniform_int_distribution<int> posX(6, static_cast<int>(width_) - 7);
        std::uniform_int_distribution<int> posY(6, static_cast<int>(height_) - 7);
        std::uniform_real_distribution<float> flux(3000.0f, 9000.0f);
        const int x = posX(rng);
        const int y = posY(rng);
        const float transientFlux = flux(rng);
        addGaussian(raw, static_cast<float>(x), static_cast<float>(y), transientFlux);
        lastTransients_.push_back({.x = static_cast<std::uint32_t>(x),
                                   .y = static_cast<std::uint32_t>(y),
                                   .flux = transientFlux});
    }

    // Apply the detector response and add noise, turning the true signal into a
    // raw sensor readout: raw = trueSignal * flat + bias + noise. Shot noise and
    // read noise add in quadrature.
    std::normal_distribution<float> unit(0.0f, 1.0f);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        const float signal = raw[i];
        const float shot = std::sqrt(std::max(signal, 0.0f));
        const float noiseSigma = std::sqrt(shot * shot + kReadNoise * kReadNoise);
        raw[i] = signal * flat_[i] + bias_[i] + noiseSigma * unit(rng);
    }
}

} // namespace vkp
