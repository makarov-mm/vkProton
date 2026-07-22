cmake_minimum_required(VERSION 3.16)
project(vkPhoton LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Vulkan REQUIRED)

# GLFW drives the viewer window on the CMake build. Prefer a system or vcpkg
# package, and fall back to fetching and building it if none is installed, so a
# fresh clone still builds with no manual dependency step.
find_package(glfw3 CONFIG QUIET)
if(NOT glfw3_FOUND)
    message(STATUS "glfw3 not found locally, fetching it")
    include(FetchContent)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4)
    FetchContent_MakeAvailable(glfw)
endif()

# Locate a GLSL to SPIR-V compiler. Newer CMake exposes glslangValidator through
# FindVulkan, otherwise fall back to searching the path for either tool.
if(DEFINED Vulkan_GLSLANG_VALIDATOR_EXECUTABLE AND Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    set(GLSL_COMPILER "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}")
    set(GLSL_COMPILER_KIND "glslang")
else()
    find_program(GLSLANG_VALIDATOR glslangValidator)
    find_program(GLSLC glslc)
    if(GLSLANG_VALIDATOR)
        set(GLSL_COMPILER "${GLSLANG_VALIDATOR}")
        set(GLSL_COMPILER_KIND "glslang")
    elseif(GLSLC)
        set(GLSL_COMPILER "${GLSLC}")
        set(GLSL_COMPILER_KIND "glslc")
    else()
        message(FATAL_ERROR "No GLSL compiler found. Install the Vulkan SDK or glslang-tools.")
    endif()
endif()
message(STATUS "Using GLSL compiler: ${GLSL_COMPILER} (${GLSL_COMPILER_KIND})")

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

set(SHADERS
    calibrate.comp
    background.comp
    difference.comp
    detect.comp
)

set(SPIRV_OUTPUTS)
foreach(shader ${SHADERS})
    set(src "${CMAKE_SOURCE_DIR}/shaders/${shader}")
    set(dst "${SHADER_OUTPUT_DIR}/${shader}.spv")
    if(GLSL_COMPILER_KIND STREQUAL "glslang")
        add_custom_command(
            OUTPUT "${dst}"
            COMMAND "${GLSL_COMPILER}" -V "${src}" -o "${dst}"
            DEPENDS "${src}"
            COMMENT "Compiling ${shader} to SPIR-V"
            VERBATIM)
    else()
        add_custom_command(
            OUTPUT "${dst}"
            COMMAND "${GLSL_COMPILER}" -fshader-stage=comp "${src}" -o "${dst}"
            DEPENDS "${src}"
            COMMENT "Compiling ${shader} to SPIR-V"
            VERBATIM)
    endif()
    list(APPEND SPIRV_OUTPUTS "${dst}")
endforeach()

add_custom_target(shaders DEPENDS ${SPIRV_OUTPUTS})

set(SOURCES
    src/main.cpp
    src/core/VulkanContext.cpp
    src/core/Buffer.cpp
    src/core/ComputeProgram.cpp
    src/pipeline/FrameSource.cpp
    src/pipeline/TransientPipeline.cpp
    src/cpu/CpuPipeline.cpp
    src/io/ImageWriter.cpp
    src/io/CatalogWriter.cpp
    src/util/Timer.cpp
    src/util/Paths.cpp
    src/viewer/Swapchain.cpp
    src/viewer/Viewer.cpp
    src/viewer/Window_Glfw.cpp
)

add_executable(vkPhoton ${SOURCES})
add_dependencies(vkPhoton shaders)
target_link_libraries(vkPhoton PRIVATE Vulkan::Vulkan glfw)
target_compile_definitions(vkPhoton PRIVATE VKPHOTON_SHADER_DIR="${SHADER_OUTPUT_DIR}")

if(MSVC)
    target_compile_options(vkPhoton PRIVATE /W4)
else()
    target_compile_options(vkPhoton PRIVATE -Wall -Wextra)
endif()
