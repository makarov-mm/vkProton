#include "CpuPipeline.hpp"

#include <algorithm>
#include <cmath>

namespace vkp {

CpuPipeline::CpuPipeline(const PipelineConfig& config) : config_(config) {
    pixelCount_ = static_cast<size_t>(config_.width) * config_.height;
    calibrated_.resize(pixelCount_);
    prevCalibrated_.assign(pixelCount_, 0.0f);
    diff_.assign(pixelCount_, 0.0f);
    diffMean_.resize(pixelCount_);
    diffStd_.resize(pixelCount_);
}

void CpuPipeline::uploadCalibration(const std::vector<float>& bias,
                                    const std::vector<float>& flatInv) {
    bias_ = bias;
    flatInv_ = flatInv;
}

FrameResult CpuPipeline::processFrame(const std::vector<float>& raw) {
    const int w = static_cast<int>(config_.width);
    const int h = static_cast<int>(config_.height);
    const int rOuter = config_.annulusOuter;
    const int rInner = config_.annulusInner;
    const int outer2 = rOuter * rOuter;
    const int inner2 = rInner * rInner;
    const bool hasPrev = frameCounter_ > 0;

    // Stage 1: calibration.
    for (size_t i = 0; i < pixelCount_; ++i) {
        calibrated_[i] = std::max((raw[i] - bias_[i]) * flatInv_[i], 0.0f);
    }

    // Stage 2: difference imaging.
    for (size_t i = 0; i < pixelCount_; ++i) {
        diff_[i] = hasPrev ? (calibrated_[i] - prevCalibrated_[i]) : 0.0f;
    }

    // Stage 3: local noise on the difference image over an annulus with one
    // sigma clipping pass.
    if (hasPrev) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float sum = 0.0f;
                float sumSq = 0.0f;
                float count = 0.0f;
                for (int dy = -rOuter; dy <= rOuter; ++dy) {
                    for (int dx = -rOuter; dx <= rOuter; ++dx) {
                        int rr = dx * dx + dy * dy;
                        if (rr <= inner2 || rr > outer2) {
                            continue;
                        }
                        int nx = std::clamp(x + dx, 0, w - 1);
                        int ny = std::clamp(y + dy, 0, h - 1);
                        float v = diff_[static_cast<size_t>(ny) * w + nx];
                        sum += v;
                        sumSq += v * v;
                        count += 1.0f;
                    }
                }
                float mean = 0.0f;
                float sigma = 1.0f;
                if (count > 1.0f) {
                    mean = sum / count;
                    sigma = std::sqrt(std::max(sumSq / count - mean * mean, 0.0f));
                }

                float lo = mean - 3.0f * sigma;
                float hi = mean + 3.0f * sigma;
                float csum = 0.0f;
                float csumSq = 0.0f;
                float ccount = 0.0f;
                for (int dy = -rOuter; dy <= rOuter; ++dy) {
                    for (int dx = -rOuter; dx <= rOuter; ++dx) {
                        int rr = dx * dx + dy * dy;
                        if (rr <= inner2 || rr > outer2) {
                            continue;
                        }
                        int nx = std::clamp(x + dx, 0, w - 1);
                        int ny = std::clamp(y + dy, 0, h - 1);
                        float v = diff_[static_cast<size_t>(ny) * w + nx];
                        if (v >= lo && v <= hi) {
                            csum += v;
                            csumSq += v * v;
                            ccount += 1.0f;
                        }
                    }
                }
                if (ccount > 1.0f) {
                    mean = csum / ccount;
                    sigma = std::sqrt(std::max(csumSq / ccount - mean * mean, 0.0f));
                }
                size_t i = static_cast<size_t>(y) * w + x;
                diffMean_[i] = mean;
                diffStd_[i] = std::max(sigma, 1e-3f);
            }
        }
    }

    // Stage 4: detection on the difference image.
    FrameResult result;
    if (hasPrev) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t i = static_cast<size_t>(y) * w + x;
                float center = diff_[i];
                if (center <= 0.0f) {
                    continue;
                }
                float significance = (center - diffMean_[i]) / diffStd_[i];
                if (significance < config_.detectSigma) {
                    continue;
                }

                bool isMax = true;
                for (int dy = -1; dy <= 1 && isMax; ++dy) {
                    int ny = std::clamp(y + dy, 0, h - 1);
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        int nx = std::clamp(x + dx, 0, w - 1);
                        if (diff_[static_cast<size_t>(ny) * w + nx] > center) {
                            isMax = false;
                            break;
                        }
                    }
                }
                if (!isMax) {
                    continue;
                }

                float up = diff_[static_cast<size_t>(std::clamp(y - 1, 0, h - 1)) * w + x];
                float down = diff_[static_cast<size_t>(std::clamp(y + 1, 0, h - 1)) * w + x];
                float left = diff_[i - (x > 0 ? 1 : 0)];
                float right = diff_[i + (x < w - 1 ? 1 : 0)];
                float neighbourMean = 0.25f * (up + down + left + right);
                if (neighbourMean < config_.shapeRatio * center) {
                    continue;
                }

                if (result.detections.size() < config_.maxDetections) {
                    Detection d;
                    d.x = static_cast<uint32_t>(x);
                    d.y = static_cast<uint32_t>(y);
                    d.diffValue = center;
                    d.snr = significance;
                    result.detections.push_back(d);
                }
                ++result.reportedCount;
            }
        }
    }

    prevCalibrated_ = calibrated_;
    ++frameCounter_;
    return result;
}

} // namespace vkp
