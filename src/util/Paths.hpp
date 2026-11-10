#pragma once

#include <filesystem>

namespace vkp {

// Absolute directory that contains the running executable, or an empty path if it
// cannot be determined. Used to find the compiled shaders next to the binary
// regardless of the working directory the program was launched from.
[[nodiscard]] std::filesystem::path executableDirectory();

} // namespace vkp
