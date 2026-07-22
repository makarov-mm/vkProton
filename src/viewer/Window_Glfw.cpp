// GLFW window backend. Used by the CMake build on Linux and Windows. The native
// Win32 backend in Window_Win32.cpp is the alternative used by the Visual Studio
// project, and only one of the two is ever compiled.

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>

#include "Window.hpp"
#include "../util/Log.hpp"

namespace vkp {

struct Window::Impl {
    GLFWwindow* handle = nullptr;
};

Window::Window(uint32_t width, uint32_t height, const std::string& title) : impl_(new Impl()) {
    if (glfwInit() != GLFW_TRUE) {
        logError("failed to initialise GLFW");
        std::abort();
    }
    if (glfwVulkanSupported() != GLFW_TRUE) {
        logError("GLFW reports no Vulkan support on this system");
        std::abort();
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    impl_->handle = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height),
                                     title.c_str(), nullptr, nullptr);
    if (impl_->handle == nullptr) {
        logError("failed to create a GLFW window");
        std::abort();
    }
}

Window::~Window() {
    if (impl_->handle != nullptr) {
        glfwDestroyWindow(impl_->handle);
    }
    glfwTerminate();
    delete impl_;
}

std::vector<const char*> Window::requiredInstanceExtensions() const {
    uint32_t count = 0;
    const char** names = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> extensions;
    extensions.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        extensions.push_back(names[i]);
    }
    return extensions;
}

VkSurfaceKHR Window::createSurface(VkInstance instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(instance, impl_->handle, nullptr, &surface) != VK_SUCCESS) {
        logError("failed to create a window surface through GLFW");
        std::abort();
    }
    return surface;
}

void Window::pollEvents() {
    glfwPollEvents();
}

bool Window::shouldClose() const {
    if (glfwWindowShouldClose(impl_->handle)) {
        return true;
    }
    // Escape closes the viewer, which is the expected behaviour for a tool window.
    return glfwGetKey(impl_->handle, GLFW_KEY_ESCAPE) == GLFW_PRESS;
}

void Window::framebufferSize(uint32_t& width, uint32_t& height) const {
    int w = 0;
    int h = 0;
    glfwGetFramebufferSize(impl_->handle, &w, &h);
    width = static_cast<uint32_t>(w);
    height = static_cast<uint32_t>(h);
}

} // namespace vkp
