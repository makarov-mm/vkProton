#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>

#include "../pipeline/Detection.hpp"

namespace vkp {

// Appends detections to a CSV catalogue, one row per event. This is the text
// product a real survey pipeline emits alongside the imagery: a machine readable
// list of what was found and where.
class CatalogWriter {
public:
    // Creates the file and writes the header row. Throws if it cannot be opened.
    explicit CatalogWriter(const std::filesystem::path& path);

    void writeFrame(std::uint32_t frameIndex, std::span<const Detection> detections);

    // Flushes the catalogue to disk and throws if any of it could not be written.
    void close();

private:
    std::filesystem::path path_;
    std::ofstream file_;
};

} // namespace vkp
