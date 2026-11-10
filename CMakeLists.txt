cmake_minimum_required(VERSION 3.25)
project(vkPhoton LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single configuration generators default to an optimised build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Vulkan REQUIRED)

# GLFW drives the viewer window on the CMake build. A system or vcpkg package is
# used when there is one, otherwise it is fetched and built, so a fresh clone
# still builds with no manual dependency step.
include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS 3.3 NAMES glfw3 CONFIG)
FetchContent_MakeAvailable(glfw)

# FindVulkan also locates the GLSL compilers, from the SDK or from the path.
if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    set(GLSL_COMPILE "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}" -V)
elseif(Vulkan_GLSLC_EXECUTABLE)
    set(GLSL_COMPILE "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute)
else()
    message(FATAL_ERROR "No GLSL compiler found. Install the Vulkan SDK or glslang-tools.")
endif()
list(GET GLSL_COMPILE 0 GLSL_COMPILER)
message(STATUS "Using GLSL compiler: ${GLSL_COMPILER}")

set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

set(SHADERS
    calibrate.comp
    difference.comp
    background.comp
    detect.comp
)

# Every stage includes common.glsl, so a change there rebuilds all of them.
set(SPIRV_OUTPUTS)
foreach(shader IN LISTS SHADERS)
    set(spirv "${SHADER_OUTPUT_DIR}/${shader}.spv")
    add_custom_command(
        OUTPUT "${spirv}"
        COMMAND ${GLSL_COMPILE} "${SHADER_SOURCE_DIR}/${shader}" -o "${spirv}"
        DEPENDS "${SHADER_SOURCE_DIR}/${shader}" "${SHADER_SOURCE_DIR}/common.glsl"
        COMMENT "Compiling ${shader} to SPIR-V"
        VERBATIM)
    list(APPEND SPIRV_OUTPUTS "${spirv}")
endforeach()
add_custom_target(shaders DEPENDS ${SPIRV_OUTPUTS})

add_executable(vkPhoton
    src/main.cpp
    src/app/Options.cpp
    src/core/Buffer.cpp
    src/core/ComputeProgram.cpp
    src/core/VulkanContext.cpp
    src/core/VulkanUtil.cpp
    src/cpu/CpuPipeline.cpp
    src/io/CatalogWriter.cpp
    src/io/ImageWriter.cpp
    src/pipeline/FrameSource.cpp
    src/pipeline/TransientPipeline.cpp
    src/util/Paths.cpp
    src/viewer/Swapchain.cpp
    src/viewer/Viewer.cpp
    src/viewer/Window_Glfw.cpp
)
add_dependencies(vkPhoton shaders)
target_link_libraries(vkPhoton PRIVATE Vulkan::Vulkan glfw)
target_compile_definitions(vkPhoton PRIVATE VKPHOTON_SHADER_DIR="${SHADER_OUTPUT_DIR}")

if(MSVC)
    target_compile_options(vkPhoton PRIVATE /W4 /permissive- /utf-8)
else()
    # Vulkan structs are filled with designated initialisers that rely on every
    # other member being zeroed, so leaving members out is intended, not an oversight.
    target_compile_options(vkPhoton PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-missing-field-initializers)
endif()
