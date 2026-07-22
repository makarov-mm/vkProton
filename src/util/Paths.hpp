#pragma once

#include <string>

namespace vkp {

// Absolute directory that contains the running executable, or an empty string if
// it cannot be determined. Used to find the compiled shaders next to the binary
// regardless of the working directory the program was launched from.
std::string executableDirectory();

} // namespace vkp
