#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <span>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#include "app/Options.hpp"
#include "core/VulkanContext.hpp"
#include "cpu/CpuPipeline.hpp"
#include "io/CatalogWriter.hpp"
#include "io/ImageWriter.hpp"
#include "pipeline/FrameSource.hpp"
#include "pipeline/TransientPipeline.hpp"
#include "util/Log.hpp"
#include "util/Timer.hpp"
#include "viewer/Viewer.hpp"
#include "viewer/Window.hpp"

namespace {

vkp::PipelineConfig makeConfig(const vkp::Options& options) {
    return {
        .width = options.width,
        .height = options.height,
        .annulusOuter = options.annulusOuter,
        .annulusInner = options.annulusInner,
        .detectSigma = options.detectSigma,
        .shapeRatio = options.shapeRatio,
    };
}

// Counts how many injected transients have a detection within a small radius.
std::uint32_t countRecovered(std::span<const vkp::TransientTruth> truth,
                             std::span<const vkp::Detection> detections) {
    constexpr int tolerance = 3;
    const auto recovered = std::ranges::count_if(truth, [&](const vkp::TransientTruth& t) {
        return std::ranges::any_of(detections, [&](const vkp::Detection& d) {
            const int dx = static_cast<int>(d.x) - static_cast<int>(t.x);
            const int dy = static_cast<int>(d.y) - static_cast<int>(t.y);
            return dx * dx + dy * dy <= tolerance * tolerance;
        });
    });
    return static_cast<std::uint32_t>(recovered);
}

void createOutputDirectory(const std::filesystem::path& directory) {
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        throw std::runtime_error(std::format("cannot create output directory {}: {}",
                                             directory.string(), error.message()));
    }
}

// Initial window size for a frame: the frame's own size, scaled so its longer side
// fits a typical desktop and tiny frames are still large enough to look at. The
// viewer letterboxes, so the lower bound on the shorter side only adds black bars.
VkExtent2D initialWindowSize(std::uint32_t frameWidth, std::uint32_t frameHeight) {
    constexpr double minLongSide = 256.0;
    constexpr double maxLongSide = 1000.0;
    constexpr long minSide = 64;

    const double longSide = std::max(frameWidth, frameHeight);
    const double scale = std::clamp(1.0, minLongSide / longSide, maxLongSide / longSide);
    const auto side = [=](std::uint32_t frameSide) {
        return static_cast<std::uint32_t>(std::max(std::lround(frameSide * scale), minSide));
    };
    return {side(frameWidth), side(frameHeight)};
}

// Headless mode. Streams the synthetic sky through the GPU pipeline, writes the
// catalogue and a few annotated frames, then reports recall against the injected
// truth and throughput against the CPU reference.
void runHeadless(const vkp::Options& options) {
    // The CPU baseline only has to establish a reference rate, so it runs on the
    // first few frames. The speedup compares the two pipelines on exactly those.
    constexpr std::uint32_t maxBenchmarkFrames = 6;

    const vkp::PipelineConfig config = makeConfig(options);
    const double megapixels = static_cast<double>(config.width) * config.height / 1.0e6;
    const std::uint32_t benchmarkFrames = std::min(options.frames, maxBenchmarkFrames);

    vkp::logInfo("frame size {}x{}, {} frames", config.width, config.height, options.frames);

    // A bad output path is the cheapest thing to find out about, so check it first.
    createOutputDirectory(options.outputDir);

    const vkp::VulkanContext context({.enableValidation = options.validation});
    vkp::FrameSource source(config.width, config.height, options.seed);
    vkp::TransientPipeline pipeline(context, config);
    pipeline.uploadCalibration(source.biasField(), source.flatInvField());

    vkp::CatalogWriter catalog(options.outputDir / "transients.csv");
    std::vector<float> raw;
    std::uint32_t totalTruth = 0;
    std::uint32_t totalRecovered = 0;
    std::uint32_t imagesWritten = 0;
    // Pipeline time only: upload, the four compute stages and readback. Generating
    // the synthetic sky and writing files are not part of what is being measured.
    double gpuSeconds = 0.0;
    double gpuBenchmarkSeconds = 0.0;

    for (std::uint32_t frame = 0; frame < options.frames; ++frame) {
        source.generateRaw(frame, raw);
        const bool wantImage = options.writeImages && imagesWritten < options.maxImages;

        const vkp::Timer timer;
        const vkp::FrameResult result = pipeline.processFrame(raw, wantImage);
        const double seconds = timer.elapsedSeconds();
        gpuSeconds += seconds;
        if (frame < benchmarkFrames) {
            gpuBenchmarkSeconds += seconds;
        }

        catalog.writeFrame(frame, result.detections);
        if (wantImage) {
            vkp::writeAnnotatedPpm(options.outputDir / std::format("frame_{:04}.ppm", frame),
                                   result.calibrated, config.width, config.height,
                                   result.detections);
            ++imagesWritten;
        }

        const std::span<const vkp::TransientTruth> truth = source.lastTransients();
        const std::uint32_t recovered = countRecovered(truth, result.detections);
        totalTruth += static_cast<std::uint32_t>(truth.size());
        totalRecovered += recovered;

        vkp::logInfo("frame {:3}  detections {:4}  injected {}  recovered {}", frame,
                     result.reportedCount, truth.size(), recovered);
    }
    catalog.close();

    const double gpuFps = options.frames / gpuSeconds;
    const double recall = totalTruth > 0 ? 100.0 * totalRecovered / totalTruth : 0.0;
    vkp::logInfo("");
    vkp::logInfo("GPU summary");
    vkp::logInfo("  processed {} frames in {:.3f} s  ({:.1f} fps, {:.1f} Mpix/s)", options.frames,
                 gpuSeconds, gpuFps, gpuFps * megapixels);
    vkp::logInfo("  recovered {} of {} injected transients ({:.1f} percent recall)",
                 totalRecovered, totalTruth, recall);

    if (options.runBenchmark) {
        vkp::CpuPipeline cpu(config);
        cpu.uploadCalibration(source.biasField(), source.flatInvField());

        double cpuSeconds = 0.0;
        for (std::uint32_t frame = 0; frame < benchmarkFrames; ++frame) {
            source.generateRaw(frame, raw);
            const vkp::Timer timer;
            cpu.processFrame(raw);
            cpuSeconds += timer.elapsedSeconds();
        }
        const double cpuFps = benchmarkFrames / cpuSeconds;

        vkp::logInfo("");
        vkp::logInfo("CPU baseline (single thread reference)");
        vkp::logInfo("  {} frames in {:.3f} s  ({:.2f} fps, {:.1f} Mpix/s)", benchmarkFrames,
                     cpuSeconds, cpuFps, cpuFps * megapixels);
        vkp::logInfo("  GPU speedup over CPU baseline: {:.1f}x", cpuSeconds / gpuBenchmarkSeconds);
    }

    vkp::logInfo("");
    vkp::logInfo("wrote {} annotated frames and a catalogue to {}/", imagesWritten,
                 options.outputDir.string());
}

// Live viewer mode. Streams the synthetic sky through the pipeline and presents
// each annotated frame in a window, looping over the sequence until the window is
// closed. The pipeline history is reset on each wrap so difference imaging starts
// clean, exactly as it would at the top of a fresh observing run.
void runViewer(const vkp::Options& options) {
    const vkp::PipelineConfig config = makeConfig(options);
    const VkExtent2D windowSize = initialWindowSize(config.width, config.height);
    vkp::Window window(windowSize.width, windowSize.height, "vkPhoton");

    // The window outlives the context, so the surface goes before the window does.
    const vkp::VulkanContext context({
        .enableValidation = options.validation,
        .instanceExtensions = window.requiredInstanceExtensions(),
        .createSurface = [&window](VkInstance instance) { return window.createSurface(instance); },
    });

    vkp::FrameSource source(config.width, config.height, options.seed);
    vkp::TransientPipeline pipeline(context, config);
    pipeline.uploadCalibration(source.biasField(), source.flatInvField());
    vkp::Viewer viewer(context, window, config.width, config.height);

    vkp::logInfo("viewer running, close the window or press Escape to quit");

    using Clock = std::chrono::steady_clock;
    const auto frameInterval = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(options.viewFps > 0 ? 1.0 / options.viewFps : 0.0));

    std::vector<float> raw;
    std::uint32_t frame = 0;
    for (;;) {
        window.pollEvents();
        if (window.shouldClose()) {
            break;
        }
        const Clock::time_point frameStart = Clock::now();

        source.generateRaw(frame, raw);
        const vkp::FrameResult result = pipeline.processFrame(raw, true);
        viewer.showFrame(vkp::buildRgba(result.calibrated, config.width, config.height,
                                        result.detections));

        if (++frame >= options.frames) {
            frame = 0;
            pipeline.resetHistory();
        }
        std::this_thread::sleep_until(frameStart + frameInterval);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::span<const char* const> arguments(argv, static_cast<std::size_t>(argc));
        const auto options = vkp::parseOptions(arguments.subspan(arguments.empty() ? 0 : 1));
        if (!options) {
            vkp::logError("{}", options.error());
            vkp::logError("run with --help for the list of options");
            return EXIT_FAILURE;
        }

        if (options->showHelp) {
            vkp::printUsage();
        } else if (options->view) {
            runViewer(*options);
        } else {
            runHeadless(*options);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        vkp::logError("{}", error.what());
        return EXIT_FAILURE;
    }
}
