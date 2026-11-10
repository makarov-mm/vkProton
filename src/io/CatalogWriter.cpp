#include "CatalogWriter.hpp"

#include <format>
#include <ostream>
#include <print>
#include <stdexcept>

namespace vkp {

CatalogWriter::CatalogWriter(const std::filesystem::path& path)
    : path_(path), file_(path, std::ios::out | std::ios::trunc) {
    if (!file_) {
        throw std::runtime_error(std::format("cannot open catalogue file {}", path_.string()));
    }
    std::println(file_, "frame,x,y,diff,snr");
}

void CatalogWriter::writeFrame(std::uint32_t frameIndex, std::span<const Detection> detections) {
    for (const Detection& d : detections) {
        std::println(file_, "{},{},{},{:.6g},{:.6g}", frameIndex, d.x, d.y, d.diffValue, d.snr);
    }
}

void CatalogWriter::close() {
    file_.close();
    if (!file_) {
        throw std::runtime_error(std::format("failed to write catalogue file {}", path_.string()));
    }
}

} // namespace vkp
