#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "cpu/CpuPipeline.hpp"
#include "core/VulkanContext.hpp"
#include "io/CatalogWriter.hpp"
#include "io/ImageWriter.hpp"
#include "pipeline/FrameSource.hpp"
#include "pipeline/TransientPipeline.hpp"
#include "util/Log.hpp"
#include "util/Timer.hpp"
#include "viewer/Viewer.hpp"
#include "viewer/Window.hpp"

namespace {

struct Options {
    uint32_t width = 1024;
    uint32_t height = 1024;
    uint32_t frames = 30;
    uint32_t seed = 1;
    uint32_t maxImages = 8;
    int annulusOuter = 7;
    int annulusInner = 4;
    float detectSigma = 6.0f;
    float shapeRatio = 0.35f;
    bool writeImages = true;
    bool runBenchmark = true;
    bool validation = false;
    bool view = false;
    uint32_t viewFps = 8;
    std::string outputDir = "output";
};

void printUsage() {
    std::cout <<
        "vkPhoton - Vulkan compute pipeline for streaming transient detection\n\n"
        "Options:\n"
        "  --width N          frame width in pixels (default 1024)\n"
        "  --height N         frame height in pixels (default 1024)\n"
        "  --frames N         number of frames to stream (default 30)\n"
        "  --seed N           random seed for the synthetic sky (default 1)\n"
        "  --detect-sigma F   detection threshold in local noise units (default 6.0)\n"
        "  --annulus-outer N  outer radius of the noise annulus (default 7)\n"
        "  --annulus-inner N  inner radius excluded from the annulus (default 4)\n"
        "  --shape-ratio F    minimum neighbour to centre ratio, 0 disables (default 0.35)\n"
        "  --max-images N     cap on annotated frames written (default 8)\n"
        "  --output DIR       output directory (default output)\n"
        "  --view             open a live window and stream the pipeline into it\n"
        "  --view-fps N       target frames per second in the viewer (default 8)\n"
        "  --no-images        skip writing annotated frames\n"
        "  --no-benchmark     skip the CPU baseline comparison\n"
        "  --validation       enable the Vulkan validation layer\n"
        "  --help             show this message\n";
}

bool parseArgs(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](uint32_t& out) {
            if (i + 1 < argc) {
                out = static_cast<uint32_t>(std::stoul(argv[++i]));
            }
        };
        auto nextFloat = [&](float& out) {
            if (i + 1 < argc) {
                out = std::stof(argv[++i]);
            }
        };
        if (arg == "--width") next(options.width);
        else if (arg == "--height") next(options.height);
        else if (arg == "--frames") next(options.frames);
        else if (arg == "--seed") next(options.seed);
        else if (arg == "--detect-sigma") nextFloat(options.detectSigma);
        else if (arg == "--annulus-outer") { uint32_t v = 7; next(v); options.annulusOuter = static_cast<int>(v); }
        else if (arg == "--annulus-inner") { uint32_t v = 4; next(v); options.annulusInner = static_cast<int>(v); }
        else if (arg == "--shape-ratio") nextFloat(options.shapeRatio);
        else if (arg == "--max-images") next(options.maxImages);
        else if (arg == "--output") { if (i + 1 < argc) options.outputDir = argv[++i]; }
        else if (arg == "--view") options.view = true;
        else if (arg == "--view-fps") next(options.viewFps);
        else if (arg == "--no-images") options.writeImages = false;
        else if (arg == "--no-benchmark") options.runBenchmark = false;
        else if (arg == "--validation") options.validation = true;
        else if (arg == "--help") { printUsage(); return false; }
        else { vkp::logError("unknown argument: " + arg); printUsage(); return false; }
    }
    return true;
}

// Counts how many injected transients have a detection within a small radius.
uint32_t countRecovered(const std::vector<vkp::TransientTruth>& truth,
                        const std::vector<vkp::Detection>& detections) {
    const int tolerance = 3;
    uint32_t recovered = 0;
    for (const auto& t : truth) {
        for (const auto& d : detections) {
            int dx = static_cast<int>(d.x) - static_cast<int>(t.x);
            int dy = static_cast<int>(d.y) - static_cast<int>(t.y);
            if (dx * dx + dy * dy <= tolerance * tolerance) {
                ++recovered;
                break;
            }
        }
    }
    return recovered;
}

std::string frameImagePath(const std::string& dir, uint32_t frame) {
    std::ostringstream out;
    out << dir << "/frame_" << std::setfill('0') << std::setw(4) << frame << ".ppm";
    return out.str();
}

vkp::PipelineConfig makeConfig(const Options& options) {
    vkp::PipelineConfig config;
    config.width = options.width;
    config.height = options.height;
    config.annulusOuter = options.annulusOuter;
    config.annulusInner = options.annulusInner;
    config.detectSigma = options.detectSigma;
    config.shapeRatio = options.shapeRatio;
    return config;
}

// Live viewer mode. Streams the synthetic sky through the pipeline and presents
// each annotated frame in a window, looping over the sequence until the window is
// closed. The pipeline history is reset on each wrap so difference imaging starts
// clean, exactly as it would at the top of a fresh observing run.
int runViewer(const Options& options) {
    vkp::PipelineConfig config = makeConfig(options);

    uint32_t winW = std::min(config.width, 1000u);
    uint32_t winH = std::min(config.height, 1000u);
    vkp::Window window(winW, winH, "vkPhoton");

    vkp::ContextConfig ctxConfig;
    ctxConfig.enableValidation = options.validation;
    ctxConfig.enablePresent = true;
    ctxConfig.instanceExtensions = window.requiredInstanceExtensions();
    ctxConfig.surfaceFactory = [&window](VkInstance instance) {
        return window.createSurface(instance);
    };

    vkp::VulkanContext context(ctxConfig);
    if (!context.hasPresent()) {
        vkp::logError("the selected device cannot present to a window");
        return 1;
    }

    vkp::FrameSource source(config.width, config.height, options.seed);
    vkp::TransientPipeline pipeline(context, config);
    pipeline.uploadCalibration(source.biasField(), source.flatInvField());
    vkp::Viewer viewer(context, window, config.width, config.height);

    vkp::logInfo("viewer running, close the window or press Escape to quit");

    const auto frameInterval = std::chrono::milliseconds(
        options.viewFps > 0 ? 1000 / options.viewFps : 0);

    std::vector<float> raw;
    uint32_t frame = 0;
    while (!window.shouldClose()) {
        window.pollEvents();
        auto tick = std::chrono::steady_clock::now();

        source.generateRaw(frame, raw);
        vkp::FrameResult result = pipeline.processFrame(raw);
        std::vector<uint8_t> rgba = vkp::ImageWriter::buildRgba(
            pipeline.lastCalibrated(), config.width, config.height, result.detections);
        viewer.showFrame(rgba);

        ++frame;
        if (frame >= options.frames) {
            frame = 0;
            pipeline.resetHistory();
        }

        std::this_thread::sleep_until(tick + frameInterval);
    }

    vkDeviceWaitIdle(context.device());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseArgs(argc, argv, options)) {
        return 0;
    }

    if (options.view) {
        return runViewer(options);
    }

    vkp::PipelineConfig config = makeConfig(options);

    const double megapixels = static_cast<double>(config.width) * config.height / 1.0e6;

    vkp::logInfo("frame size " + std::to_string(config.width) + "x" +
                 std::to_string(config.height) + ", " + std::to_string(options.frames) + " frames");

    vkp::VulkanContext context(options.validation);
    vkp::FrameSource source(config.width, config.height, options.seed);
    vkp::TransientPipeline pipeline(context, config);
    pipeline.uploadCalibration(source.biasField(), source.flatInvField());

    vkp::CatalogWriter catalog(options.outputDir + "/transients.csv");
    if (!catalog.isOpen()) {
        vkp::logError("could not open catalogue file, does the output directory exist?");
        vkp::logInfo("create it first, for example: mkdir -p " + options.outputDir);
        return 1;
    }

    std::vector<float> raw;
    uint32_t totalReported = 0;
    uint32_t totalTruth = 0;
    uint32_t totalRecovered = 0;
    uint32_t imagesWritten = 0;

    vkp::Timer gpuTimer;
    for (uint32_t frame = 0; frame < options.frames; ++frame) {
        source.generateRaw(frame, raw);
        vkp::FrameResult result = pipeline.processFrame(raw);

        catalog.writeFrame(frame, result.detections);

        const auto& truth = source.lastTransients();
        uint32_t recovered = countRecovered(truth, result.detections);
        totalReported += result.reportedCount;
        totalTruth += static_cast<uint32_t>(truth.size());
        totalRecovered += recovered;

        if (options.writeImages && imagesWritten < options.maxImages) {
            std::string path = frameImagePath(options.outputDir, frame);
            if (vkp::ImageWriter::writeAnnotatedPpm(path, pipeline.lastCalibrated(),
                                                    config.width, config.height,
                                                    result.detections)) {
                ++imagesWritten;
            }
        }

        std::ostringstream line;
        line << "frame " << std::setw(3) << frame
             << "  detections " << std::setw(4) << result.reportedCount
             << "  injected " << truth.size()
             << "  recovered " << recovered;
        vkp::logInfo(line.str());
    }
    double gpuSeconds = gpuTimer.elapsedSeconds();

    vkp::logInfo("");
    vkp::logInfo("GPU summary");
    {
        double fps = options.frames / gpuSeconds;
        std::ostringstream out;
        out << "  processed " << options.frames << " frames in "
            << std::fixed << std::setprecision(3) << gpuSeconds << " s"
            << "  (" << std::setprecision(1) << fps << " fps, "
            << std::setprecision(1) << fps * megapixels << " Mpix/s)";
        vkp::logInfo(out.str());
    }
    {
        std::ostringstream out;
        double recall = totalTruth > 0 ? 100.0 * totalRecovered / totalTruth : 0.0;
        out << "  recovered " << totalRecovered << " of " << totalTruth
            << " injected transients (" << std::fixed << std::setprecision(1) << recall << " percent recall)";
        vkp::logInfo(out.str());
    }

    if (options.runBenchmark) {
        vkp::CpuPipeline cpu(config);
        cpu.uploadCalibration(source.biasField(), source.flatInvField());
        // Keep the CPU baseline short when the image is large, it is only there
        // to establish a reference rate.
        uint32_t benchFrames = options.frames < 6 ? options.frames : 6;

        vkp::Timer cpuTimer;
        for (uint32_t frame = 0; frame < benchFrames; ++frame) {
            source.generateRaw(frame, raw);
            cpu.processFrame(raw);
        }
        double cpuSeconds = cpuTimer.elapsedSeconds();
        double cpuFps = benchFrames / cpuSeconds;
        double gpuFps = options.frames / gpuSeconds;

        vkp::logInfo("");
        vkp::logInfo("CPU baseline (single thread reference)");
        std::ostringstream out;
        out << "  " << benchFrames << " frames in " << std::fixed << std::setprecision(3)
            << cpuSeconds << " s  (" << std::setprecision(2) << cpuFps << " fps, "
            << std::setprecision(1) << cpuFps * megapixels << " Mpix/s)";
        vkp::logInfo(out.str());

        std::ostringstream speedup;
        speedup << "  GPU speedup over CPU baseline: " << std::fixed << std::setprecision(1)
                << (gpuFps / cpuFps) << "x";
        vkp::logInfo(speedup.str());
    }

    vkp::logInfo("");
    vkp::logInfo("wrote " + std::to_string(imagesWritten) + " annotated frames and a catalogue to " +
                 options.outputDir + "/");
    return 0;
}
