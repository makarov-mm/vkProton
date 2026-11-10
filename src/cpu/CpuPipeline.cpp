#include "CpuPipeline.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vkp {

CpuPipeline::CpuPipeline(const PipelineConfig& config)
    : config_(config),
      width_(static_cast<int>(config.width)),
      height_(static_cast<int>(config.height)),
      pixelCount_(static_cast<std::size_t>(config.width) * config.height),
      calibrated_{std::vector<float>(pixelCount_), std::vector<float>(pixelCount_)},
      diff_(pixelCount_),
      diffMean_(pixelCount_),
      diffStd_(pixelCount_) {}

void CpuPipeline::uploadCalibration(std::span<const float> bias, std::span<const float> flatInv) {
    if (bias.size() != pixelCount_ || flatInv.size() != pixelCount_) {
        throw std::invalid_argument("calibration frames do not match the configured frame size");
    }
    bias_.assign(bias.begin(), bias.end());
    flatInv_.assign(flatInv.begin(), flatInv.end());
}

FrameResult CpuPipeline::processFrame(std::span<const float> raw) {
    if (raw.size() != pixelCount_ || bias_.size() != pixelCount_) {
        throw std::invalid_argument("raw frame or calibration does not match the frame size");
    }

    const std::uint32_t current = frameCounter_ % 2;
    const bool hasPrev = frameCounter_ > 0;

    calibrate(raw, calibrated_[current]);

    // Without a previous frame there is nothing to difference against, so the
    // first frame of a sequence cannot produce detections.
    FrameResult result;
    if (hasPrev) {
        difference(calibrated_[current], calibrated_[1 - current]);
        background();
        result = detect();
    }
    result.calibrated = calibrated_[current];

    ++frameCounter_;
    return result;
}

// Stage 1: calibration.
void CpuPipeline::calibrate(std::span<const float> raw, std::span<float> calibrated) const {
    for (std::size_t i = 0; i < pixelCount_; ++i) {
        calibrated[i] = std::max((raw[i] - bias_[i]) * flatInv_[i], 0.0f);
    }
}

// Stage 2: difference imaging.
void CpuPipeline::difference(std::span<const float> calibrated, std::span<const float> previous) {
    for (std::size_t i = 0; i < pixelCount_; ++i) {
        diff_[i] = calibrated[i] - previous[i];
    }
}

// Stage 3: local noise on the difference image over an annulus, with one sigma
// clipping pass.
void CpuPipeline::background() {
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            RingStats stats = ringStats(x, y, false, {.mean = 0.0f, .sigma = 1.0f});
            stats = ringStats(x, y, true, stats);

            const std::size_t i = pixelIndex(x, y);
            diffMean_[i] = stats.mean;
            diffStd_[i] = std::max(stats.sigma, 1e-3f);
        }
    }
}

// Stage 4: detection on the difference image.
FrameResult CpuPipeline::detect() const {
    FrameResult result;
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const std::size_t i = pixelIndex(x, y);
            const float center = diff_[i];
            if (center <= 0.0f) {
                continue;
            }

            const float significance = (center - diffMean_[i]) / diffStd_[i];
            if (significance < config_.detectSigma) {
                continue;
            }

            if (!isLocalMaximum(x, y, center)) {
                continue;
            }

            // Point spread function shape test on the four direct neighbours.
            if (config_.shapeRatio > 0.0f) {
                const float up = diffAt(x, y - 1);
                const float down = diffAt(x, y + 1);
                const float left = diffAt(x - 1, y);
                const float right = diffAt(x + 1, y);
                const float neighbourMean = 0.25f * (up + down + left + right);
                if (neighbourMean < config_.shapeRatio * center) {
                    continue;
                }
            }

            if (result.detections.size() < config_.maxDetections) {
                result.detections.push_back({.x = static_cast<std::uint32_t>(x),
                                             .y = static_cast<std::uint32_t>(y),
                                             .diffValue = center,
                                             .snr = significance});
            }
            ++result.reportedCount;
        }
    }
    return result;
}

// Mean and standard deviation of the annulus samples around (x, y). With clip set,
// only samples within three sigma of the given estimate are counted. With fewer
// than two samples left the estimate is returned as is.
CpuPipeline::RingStats CpuPipeline::ringStats(int x, int y, bool clip, RingStats estimate) const {
    const int rOuter = config_.annulusOuter;
    const int outer2 = rOuter * rOuter;
    const int inner2 = config_.annulusInner * config_.annulusInner;
    const float lo = estimate.mean - 3.0f * estimate.sigma;
    const float hi = estimate.mean + 3.0f * estimate.sigma;

    float sum = 0.0f;
    float sumSq = 0.0f;
    float count = 0.0f;
    for (int dy = -rOuter; dy <= rOuter; ++dy) {
        for (int dx = -rOuter; dx <= rOuter; ++dx) {
            const int rr = dx * dx + dy * dy;
            if (rr <= inner2 || rr > outer2) {
                continue;
            }
            const float v = diffAt(x + dx, y + dy);
            if (!clip || (v >= lo && v <= hi)) {
                sum += v;
                sumSq += v * v;
                count += 1.0f;
            }
        }
    }

    if (count <= 1.0f) {
        return estimate;
    }
    const float mean = sum / count;
    return {.mean = mean, .sigma = std::sqrt(std::max(sumSq / count - mean * mean, 0.0f))};
}

// Local maximum test over the 3x3 neighbourhood of the difference image.
bool CpuPipeline::isLocalMaximum(int x, int y, float center) const {
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if ((dx != 0 || dy != 0) && diffAt(x + dx, y + dy) > center) {
                return false;
            }
        }
    }
    return true;
}

// Difference value at (x, y) with the coordinates clamped to the frame.
float CpuPipeline::diffAt(int x, int y) const {
    return diff_[pixelIndex(std::clamp(x, 0, width_ - 1), std::clamp(y, 0, height_ - 1))];
}

std::size_t CpuPipeline::pixelIndex(int x, int y) const noexcept {
    return static_cast<std::size_t>(y) * config_.width + static_cast<std::size_t>(x);
}

} // namespace vkp
