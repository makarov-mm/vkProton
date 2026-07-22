# vkPhoton

A Vulkan compute pipeline that streams synthetic telescope frames and detects astronomical transients on the GPU. It is a small, self contained take on the kind of workload NVIDIA showed with cuPhoton: pull raw sensor data through a calibration and detection pipeline that lives entirely in device memory, and emit a science product at the other end. This one runs on an ordinary desktop GPU instead of a datacenter part, and it is built on Vulkan compute rather than CUDA.

The compute runs headless by default and writes its results to disk. A `--view` flag adds a live window that streams the annotated frames as they are processed, using the same device.

The whole thing is deterministic and needs no external data. It generates its own sky, injects its own transients, finds them, and reports how many it recovered.

## What it does

Each frame is a static star field seen through a fixed flat field and bias pattern, with photon and read noise added, and occasionally a transient point source dropped in at a random location. The pipeline never sees where the transients are. It has to find them, and at the end the driver compares the detections against the injected truth to report recall.

The GPU work is four compute stages, all operating on storage buffers with barriers between them:

1. **Calibration.** `calibrated = (raw - bias) * flatInv`. Removes the detector response, exactly as a real photometric pipeline does before measuring anything.
2. **Difference imaging.** Subtracts the previous calibrated frame from the current one. Static sources cancel, so what survives is whatever changed between exposures. You do not search the sky, you search the difference of the sky.
3. **Noise estimation.** For every pixel, it measures the mean and standard deviation of the difference image over an annulus, a ring that skips the central core. Skipping the core is the important part: a real transient sits at the centre, and if it were included it would inflate the estimate of its own local noise and hide itself. Measuring on the difference image also puts the noise where it belongs, since bright stars leave elevated photon noise exactly where false positives would otherwise appear, and the annulus raises the local threshold there automatically.
4. **Detection.** A pixel is reported when it is a positive excursion, its significance clears the threshold in units of local noise, it is a local maximum in its 3x3 neighbourhood, and its four direct neighbours are also elevated. That last test is a cheap point spread function check: a real source spreads over several pixels, a noise spike does not. Survivors are appended to a shared list through an atomic counter.

The result of each frame is an annotated image with detections marked, plus a row per detection in a CSV catalogue.

On the reference sky, the detector recovers essentially all injected transients with very few false positives. Precision is not perfect, and it is not meant to be: proper survey pipelines use point spread function matched subtraction to normalise the noise before differencing, which is the natural next step here.

## Output

Running the tool writes into the output directory:

- `frame_NNNN.ppm`, calibrated frames with an asinh stretch and red crosses on detections
- `transients.csv`, columns `frame,x,y,diff,snr`

PPM is used on purpose so the project has no image library dependency. Any viewer or `convert frame_0003.ppm frame_0003.png` will turn them into something more portable.

## Building

There are two ways to build: CMake for a cross platform build, and a native Visual Studio solution for Windows. Both build the same code. The only difference is how the viewer window is created, which is isolated behind one small class.

### CMake

Requirements:

- A C++17 compiler
- The Vulkan SDK, or on Linux the `libvulkan-dev` and `glslang-tools` packages
- CMake 3.16 or newer
- A GPU with a compute queue and up to date drivers

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

CMake compiles the four shaders to SPIR-V as part of the build and points the executable at them, so there is no manual shader step. The viewer window uses GLFW: if it is not already installed, CMake fetches and builds it automatically, so a fresh clone builds with no extra steps. This path works on both Linux and Windows.

### Visual Studio

The repository also ships a native solution in the modern XML format, `vkPhoton.slnx`, with `vkPhoton.vcxproj`. The only dependency is the Vulkan SDK: the viewer uses a native Win32 window here, so there is no GLFW or vcpkg to set up.

1. Install the Vulkan SDK from LunarG. Its installer sets the `VULKAN_SDK` environment variable, which the project reads for the headers and `vulkan-1.lib`.
2. Open `vkPhoton.slnx` in Visual Studio 2022.
3. Pick the Debug or Release x64 configuration and build.

A pre build step compiles the four shaders to SPIR-V into the output directory next to the executable, and the program finds them there at runtime.

## Running

```
mkdir output
./build/vkPhoton
```

Useful options:

```
--width N          frame width in pixels (default 1024)
--height N         frame height in pixels (default 1024)
--frames N         number of frames to stream (default 30)
--seed N           random seed for the synthetic sky (default 1)
--detect-sigma F   detection threshold in local noise units (default 6.0)
--annulus-outer N  outer radius of the noise annulus (default 7)
--annulus-inner N  inner radius excluded from the annulus (default 4)
--shape-ratio F    minimum neighbour to centre ratio, 0 disables (default 0.35)
--max-images N     cap on annotated frames written (default 8)
--output DIR       output directory (default output)
--view             open a live window and stream the pipeline into it
--view-fps N       target frames per second in the viewer (default 8)
--no-images        skip writing annotated frames
--no-benchmark     skip the CPU baseline comparison
--validation       enable the Vulkan validation layer
```

The run prints per frame detection counts, an overall recall against the injected transients, GPU throughput in frames and megapixels per second, and a comparison against a single threaded CPU implementation of the same pipeline.

### Live viewer

```
./build/vkPhoton --view
```

With `--view` the tool opens a window and streams the annotated frames into it in real time, looping over the sequence. Each frame is uploaded to a device image and blitted into the swapchain, scaled to the window, so the display path is a handful of copies with no render pass. Close the window or press Escape to quit. This uses the same device as the compute pipeline, so the frames you watch are the actual pipeline output, transients marked and all.

## Layout

```
shaders/           the four compute stages in GLSL
src/core/          Vulkan context, buffers, compute program wrapper
src/pipeline/      frame generator and the GPU pipeline orchestrator
src/cpu/           single threaded reference pipeline for the benchmark
src/io/            PPM image writer and CSV catalogue writer
src/util/          logging, timing and executable path lookup
src/viewer/        window, swapchain and the live presentation path
```

The CPU pipeline in `src/cpu` mirrors the four shader stages line for line. It is there as an honest benchmark baseline and as a way to cross check the GPU output on the same input.

The viewer in `src/viewer` is written so that only window creation is platform specific. That one job sits behind a `Window` class with two backends, GLFW and native Win32, and exactly one is compiled per build. The swapchain, the presentation path and the context changes that enable it are shared and identical on both.

## Notes on the design

The pipeline itself is headless: no graphics queue, just storage buffers and dispatches, which keeps the Vulkan surface small and the focus on the work. The viewer adds presentation on top of that same device without changing how the compute runs.

Frames are processed one at a time with a submit and wait per frame. That measures honest per frame latency including upload and readback, but it leaves throughput on the table. Batching several frames per submit and overlapping upload with compute is the obvious performance follow up. Other natural extensions are point spread function matched subtraction for cleaner differencing, and connected component grouping so an extended source becomes a single detection.

## Verification

The shaders are validated to SPIR-V by the build. The detection algorithm was checked through the CPU reference, which shares the exact math of the shaders, and recovers the injected transients with high recall and precision across several seeds on the synthetic sky. The full C++, including the swapchain and presentation path, compiles and links against the Vulkan headers, and the native Win32 backend was cross checked against the Windows headers as well, both with warnings turned up.
