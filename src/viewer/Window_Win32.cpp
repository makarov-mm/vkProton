// Native Win32 window backend. Used by the Visual Studio project so the only
// dependency is the Vulkan SDK. The CMake build uses the GLFW backend in
// Window_Glfw.cpp instead, and only one of the two is ever compiled. Everything
// above the Window class is identical on both paths.

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <windows.h>

#include <cstdlib>

#include "Window.hpp"
#include "../util/Log.hpp"

namespace vkp {

struct Window::Impl {
    HINSTANCE hinstance = nullptr;
    HWND hwnd = nullptr;
    bool shouldClose = false;
};

namespace {
const char* kWindowClass = "vkPhotonWindow";

LRESULT CALLBACK windowProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    // The window stores a pointer to its close flag in the user data slot. Using
    // the flag rather than the whole Impl keeps this free function clear of the
    // private nested type.
    auto* closeFlag = reinterpret_cast<bool*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    switch (message) {
        case WM_CLOSE:
        case WM_DESTROY:
            if (closeFlag != nullptr) {
                *closeFlag = true;
            }
            return 0;
        case WM_KEYDOWN:
            if (closeFlag != nullptr && wparam == VK_ESCAPE) {
                *closeFlag = true;
            }
            return 0;
        default:
            return DefWindowProc(hwnd, message, wparam, lparam);
    }
}
} // namespace

Window::Window(uint32_t width, uint32_t height, const std::string& title) : impl_(new Impl()) {
    impl_->hinstance = GetModuleHandle(nullptr);

    WNDCLASSEXA wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = windowProc;
    wc.hInstance = impl_->hinstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = kWindowClass;
    RegisterClassExA(&wc);

    RECT rect{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    impl_->hwnd = CreateWindowExA(
        0, kWindowClass, title.c_str(), WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rect.right - rect.left, rect.bottom - rect.top,
        nullptr, nullptr, impl_->hinstance, nullptr);

    if (impl_->hwnd == nullptr) {
        logError("failed to create a Win32 window");
        std::abort();
    }

    SetWindowLongPtr(impl_->hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&impl_->shouldClose));
    ShowWindow(impl_->hwnd, SW_SHOW);
    UpdateWindow(impl_->hwnd);
}

Window::~Window() {
    if (impl_->hwnd != nullptr) {
        DestroyWindow(impl_->hwnd);
    }
    UnregisterClassA(kWindowClass, impl_->hinstance);
    delete impl_;
}

std::vector<const char*> Window::requiredInstanceExtensions() const {
    return {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
}

VkSurfaceKHR Window::createSurface(VkInstance instance) const {
    VkWin32SurfaceCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    info.hinstance = impl_->hinstance;
    info.hwnd = impl_->hwnd;

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (vkCreateWin32SurfaceKHR(instance, &info, nullptr, &surface) != VK_SUCCESS) {
        logError("failed to create a Win32 Vulkan surface");
        std::abort();
    }
    return surface;
}

void Window::pollEvents() {
    MSG message;
    while (PeekMessage(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessage(&message);
    }
}

bool Window::shouldClose() const {
    return impl_->shouldClose;
}

void Window::framebufferSize(uint32_t& width, uint32_t& height) const {
    RECT rect{};
    GetClientRect(impl_->hwnd, &rect);
    width = static_cast<uint32_t>(rect.right - rect.left);
    height = static_cast<uint32_t>(rect.bottom - rect.top);
}

} // namespace vkp
