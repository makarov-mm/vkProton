#include "FrameSource.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace vkp {

namespace {
// Cheap deterministic per frame seed so noise and transients are reproducible
// without carrying any state between calls.
uint32_t mixSeed(uint32_t base, uint32_t frame) {
    uint32_t h = base * 2654435761u;
    h ^= (frame + 0x9e3779b9u + (h << 6) + (h >> 2));
    return h;
}
} // namespace

FrameSource::FrameSource(uint32_t width, uint32_t height, uint32_t seed)
    : width_(width), height_(height), seed_(seed) {
    bias_.resize(pixelCount());
    flat_.resize(pixelCount());
    flatInv_.resize(pixelCount());
    trueSignal_.resize(pixelCount());
    buildCalibration(seed_);
    buildStars(seed_);
}

void FrameSource::buildCalibration(uint32_t seed) {
    std::mt19937 rng(seed ^ 0x1234abcdu);
    std::uniform_real_distribution<float> jitter(-2.0f, 2.0f);

    // Bias: a constant offset with a mild diagonal gradient and per pixel jitter,
    // the kind of fixed pattern a real detector shows.
    // Flat: a smooth vignette that dims the corners, values around 0.85 to 1.05.
    for (uint32_t y = 0; y < height_; ++y) {
        for (uint32_t x = 0; x < width_; ++x) {
            size_t i = static_cast<size_t>(y) * width_ + x;

            float gx = static_cast<float>(x) / static_cast<float>(width_);
            float gy = static_cast<float>(y) / static_cast<float>(height_);
            bias_[i] = biasLevel_ + 20.0f * (gx + gy) + jitter(rng);

            float dx = gx - 0.5f;
            float dy = gy - 0.5f;
            float radius2 = dx * dx + dy * dy;
            float flat = 1.05f - 0.6f * radius2;
            flat_[i] = std::max(flat, 0.5f);
            flatInv_[i] = 1.0f / flat_[i];
        }
    }
}

void FrameSource::buildStars(uint32_t seed) {
    std::mt19937 rng(seed ^ 0x55aa55aau);
    std::uniform_real_distribution<float> posX(4.0f, static_cast<float>(width_) - 5.0f);
    std::uniform_real_distribution<float> posY(4.0f, static_cast<float>(height_) - 5.0f);
    std::uniform_real_distribution<float> flux(400.0f, 6000.0f);

    // Roughly one star per 800 pixels, which gives a believable but not crowded field.
    size_t count = std::max<size_t>(20, pixelCount() / 800);
    stars_.reserve(count);
    for (size_t s = 0; s < count; ++s) {
        stars_.push_back(Star{posX(rng), posY(rng), flux(rng)});
    }

    // Precompute the static true signal: sky plus every star. This never changes,
    // so difference imaging cancels it perfectly apart from noise.
    std::fill(trueSignal_.begin(), trueSignal_.end(), skyLevel_);
    for (const Star& star : stars_) {
        addGaussian(trueSignal_, star.x, star.y, star.flux, psfSigma_);
    }
}

void FrameSource::addGaussian(std::vector<float>& image, float cx, float cy, float flux, float sigma) const {
    int radius = static_cast<int>(std::ceil(sigma * 3.5f));
    float norm = flux / (2.0f * 3.14159265f * sigma * sigma);
    float inv2s2 = 1.0f / (2.0f * sigma * sigma);

    int ix = static_cast<int>(std::lround(cx));
    int iy = static_cast<int>(std::lround(cy));
    for (int dy = -radius; dy <= radius; ++dy) {
        int y = iy + dy;
        if (y < 0 || y >= static_cast<int>(height_)) {
            continue;
        }
        for (int dx = -radius; dx <= radius; ++dx) {
            int x = ix + dx;
            if (x < 0 || x >= static_cast<int>(width_)) {
                continue;
            }
            float fx = static_cast<float>(x) - cx;
            float fy = static_cast<float>(y) - cy;
            float value = norm * std::exp(-(fx * fx + fy * fy) * inv2s2);
            image[static_cast<size_t>(y) * width_ + x] += value;
        }
    }
}

void FrameSource::generateRaw(uint32_t frameIndex, std::vector<float>& raw) {
    raw.assign(trueSignal_.begin(), trueSignal_.end());
    lastTransients_.clear();

    uint32_t frameSeed = mixSeed(seed_, frameIndex);
    std::mt19937 rng(frameSeed);

    // Inject a transient on roughly one frame in three, at a fresh location.
    // A moving object would also work, but a flash is the cleanest thing to see
    // in a single difference image.
    std::uniform_int_distribution<int> chance(0, 2);
    if (frameIndex > 0 && chance(rng) == 0) {
        std::uniform_int_distribution<int> px(6, static_cast<int>(width_) - 7);
        std::uniform_int_distribution<int> py(6, static_cast<int>(height_) - 7);
        std::uniform_real_distribution<float> tflux(3000.0f, 9000.0f);
        int tx = px(rng);
        int ty = py(rng);
        float f = tflux(rng);
        addGaussian(raw, static_cast<float>(tx), static_cast<float>(ty), f, psfSigma_);
        lastTransients_.push_back(TransientTruth{static_cast<uint32_t>(tx),
                                                 static_cast<uint32_t>(ty), f});
    }

    // Apply the detector response and add noise, turning the true signal into a
    // raw sensor readout: raw = trueSignal * flat + bias + noise.
    std::normal_distribution<float> unit(0.0f, 1.0f);
    for (size_t i = 0; i < raw.size(); ++i) {
        float signal = raw[i];
        float shot = std::sqrt(std::max(signal, 0.0f));
        float noiseSigma = std::sqrt(shot * shot + readNoise_ * readNoise_);
        float noisy = signal * flat_[i] + bias_[i] + noiseSigma * unit(rng);
        raw[i] = noisy;
    }
}

} // namespace vkp
