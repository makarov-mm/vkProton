// Native Win32 window backend. Used by the Visual Studio project so the only
// dependency is the Vulkan SDK. The CMake build uses the GLFW backend in
// Window_Glfw.cpp instead, and only one of the two is ever compiled. Everything
// above the Window class is identical on both paths.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "Window.hpp"
#include "../core/VulkanUtil.hpp"

namespace vkp {

namespace {

constexpr wchar_t kWindowClass[] = L"vkPhotonWindow";

// Titles arrive as UTF-8, the Unicode window functions take UTF-16.
std::wstring toWide(const std::string& text) {
    const int size = static_cast<int>(text.size());
    const int length = MultiByteToWideChar(CP_UTF8, 0, text.data(), size, nullptr, 0);
    std::wstring wide(static_cast<std::size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), size, wide.data(), length);
    return wide;
}

} // namespace

struct Window::Impl {
    HINSTANCE instance = GetModuleHandleW(nullptr);
    HWND hwnd = nullptr;
    bool classRegistered = false;
    bool closeRequested = false;

    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    ~Impl() {
        if (hwnd != nullptr) {
            DestroyWindow(hwnd);
        }
        if (classRegistered) {
            UnregisterClassW(kWindowClass, instance);
        }
    }

    static LRESULT CALLBACK windowProc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
        // The window keeps a pointer to its state in the user data slot. The slot
        // is still empty for the messages that arrive during creation.
        auto* const self = reinterpret_cast<Impl*>(GetWindowLongPtrW(window, GWLP_USERDATA));
        if (self != nullptr) {
            switch (message) {
                case WM_CLOSE:
                case WM_DESTROY:
                    // Closing only raises the flag. The window itself is destroyed
                    // together with the Window object.
                    self->closeRequested = true;
                    return 0;
                case WM_KEYDOWN:
                    if (wparam == VK_ESCAPE) {
                        self->closeRequested = true;
                    }
                    return 0;
                default:
                    break;
            }
        }
        return DefWindowProcW(window, message, wparam, lparam);
    }
};

Window::Window(std::uint32_t width, std::uint32_t height, const std::string& title)
    : impl_(std::make_unique<Impl>()) {
    const WNDCLASSEXW windowClass{
        .cbSize = sizeof(WNDCLASSEXW),
        .style = CS_HREDRAW | CS_VREDRAW,
        .lpfnWndProc = Impl::windowProc,
        .hInstance = impl_->instance,
        .hCursor = LoadCursor(nullptr, IDC_ARROW),
        .lpszClassName = kWindowClass,
    };
    if (RegisterClassExW(&windowClass) == 0) {
        throw std::runtime_error("failed to register the Win32 window class");
    }
    impl_->classRegistered = true;

    // The requested size is the client area, so grow the outer rectangle by the
    // frame and the caption.
    RECT rect{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    impl_->hwnd = CreateWindowExW(0, kWindowClass, toWide(title).c_str(), WS_OVERLAPPEDWINDOW,
                                  CW_USEDEFAULT, CW_USEDEFAULT,
                                  rect.right - rect.left, rect.bottom - rect.top,
                                  nullptr, nullptr, impl_->instance, nullptr);
    if (impl_->hwnd == nullptr) {
        throw std::runtime_error("failed to create a Win32 window");
    }

    SetWindowLongPtrW(impl_->hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(impl_.get()));
    ShowWindow(impl_->hwnd, SW_SHOW);
}

Window::~Window() = default;

std::vector<const char*> Window::requiredInstanceExtensions() const {
    return {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
}

VkSurfaceKHR Window::createSurface(VkInstance instance) const {
    const VkWin32SurfaceCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
        .hinstance = impl_->instance,
        .hwnd = impl_->hwnd,
    };
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    vkCheck(vkCreateWin32SurfaceKHR(instance, &info, nullptr, &surface),
            "vkCreateWin32SurfaceKHR");
    return surface;
}

void Window::pollEvents() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE) != 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

bool Window::shouldClose() const {
    return impl_->closeRequested;
}

VkExtent2D Window::framebufferSize() const {
    RECT rect{};
    GetClientRect(impl_->hwnd, &rect);
    return {static_cast<std::uint32_t>(rect.right - rect.left),
            static_cast<std::uint32_t>(rect.bottom - rect.top)};
}

} // namespace vkp
