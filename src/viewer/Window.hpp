#pragma once

#include <cstdint>
#include <memory>
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
    // The size is that of the client area, in pixels. Throws if the window cannot
    // be created.
    Window(std::uint32_t width, std::uint32_t height, const std::string& title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Instance extensions the surface needs, handed to the context before it
    // creates the instance.
    [[nodiscard]] std::vector<const char*> requiredInstanceExtensions() const;

    // Wraps the platform window in a Vulkan surface for the given instance. The
    // caller owns the surface and must destroy it before the window goes away.
    [[nodiscard]] VkSurfaceKHR createSurface(VkInstance instance) const;

    void pollEvents();

    // True once the user has asked to quit, by closing the window or pressing Escape.
    [[nodiscard]] bool shouldClose() const;

    // Current framebuffer size in pixels, used to size the swapchain. Either
    // dimension is zero while the window is minimised.
    [[nodiscard]] VkExtent2D framebufferSize() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vkp
