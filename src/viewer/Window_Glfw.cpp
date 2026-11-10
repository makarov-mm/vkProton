// GLFW window backend. Used by the CMake build on Linux and Windows. The native
// Win32 backend in Window_Win32.cpp is the alternative used by the Visual Studio
// project, and only one of the two is ever compiled.

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdexcept>

#include "Window.hpp"
#include "../core/VulkanUtil.hpp"
#include "../util/Log.hpp"

namespace vkp {

struct Window::Impl {
    GLFWwindow* handle = nullptr;

    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    ~Impl() {
        if (handle != nullptr) {
            glfwDestroyWindow(handle);
        }
        // Safe to call even when initialisation never succeeded.
        glfwTerminate();
    }
};

Window::Window(std::uint32_t width, std::uint32_t height, const std::string& title)
    : impl_(std::make_unique<Impl>()) {
    glfwSetErrorCallback([](int, const char* description) { logError("GLFW: {}", description); });

    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("failed to initialise GLFW");
    }
    if (glfwVulkanSupported() != GLFW_TRUE) {
        throw std::runtime_error("GLFW reports no Vulkan support on this system");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    impl_->handle = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height),
                                     title.c_str(), nullptr, nullptr);
    if (impl_->handle == nullptr) {
        throw std::runtime_error("failed to create a GLFW window");
    }

    // Escape closes the viewer, which is the expected behaviour for a tool window.
    // The callback sees every key press. Polling the key state instead would miss
    // a tap that starts and ends between two frames.
    glfwSetKeyCallback(impl_->handle, [](GLFWwindow* window, int key, int, int action, int) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    });
}

Window::~Window() = default;

std::vector<const char*> Window::requiredInstanceExtensions() const {
    std::uint32_t count = 0;
    const char** names = glfwGetRequiredInstanceExtensions(&count);
    if (names == nullptr) {
        throw std::runtime_error("GLFW cannot create Vulkan surfaces on this system");
    }
    return std::vector<const char*>(names, names + count);
}

VkSurfaceKHR Window::createSurface(VkInstance instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    vkCheck(glfwCreateWindowSurface(instance, impl_->handle, nullptr, &surface),
            "glfwCreateWindowSurface");
    return surface;
}

void Window::pollEvents() {
    glfwPollEvents();
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(impl_->handle) == GLFW_TRUE;
}

VkExtent2D Window::framebufferSize() const {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(impl_->handle, &width, &height);
    return {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
}

} // namespace vkp
