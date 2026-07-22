#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace vkp {

// A minimal window abstraction for the live viewer. The platform backend lives
// entirely in the matching implementation file, so this header stays free of any
// windowing headers. The CMake build compiles the GLFW backend, the Visual Studio
// project compiles the native Win32 backend, and everything above this class is
// identical on both.
class Window {
public:
    Window(uint32_t width, uint32_t height, const std::string& title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Instance extensions the surface needs, handed to the context before it
    // creates the instance.
    std::vector<const char*> requiredInstanceExtensions() const;

    // Wraps the platform window in a Vulkan surface for the given instance.
    VkSurfaceKHR createSurface(VkInstance instance) const;

    void pollEvents();
    bool shouldClose() const;

    // Current framebuffer size in pixels, used to size the swapchain. Returns
    // zero for either dimension while the window is minimised.
    void framebufferSize(uint32_t& width, uint32_t& height) const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace vkp
