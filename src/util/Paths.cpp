#include "Paths.hpp"

#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

namespace vkp {

namespace {
std::string stripFileName(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return std::string();
    }
    return path.substr(0, slash);
}
} // namespace

std::string executableDirectory() {
#if defined(_WIN32)
    char buffer[MAX_PATH] = {0};
    DWORD length = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (length == 0 || length == MAX_PATH) {
        return std::string();
    }
    return stripFileName(std::string(buffer, length));
#else
    char buffer[PATH_MAX] = {0};
    ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length <= 0) {
        return std::string();
    }
    buffer[length] = '\0';
    return stripFileName(std::string(buffer));
#endif
}

} // namespace vkp
