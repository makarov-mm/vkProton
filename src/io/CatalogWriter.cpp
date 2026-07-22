#include "CatalogWriter.hpp"

namespace vkp {

CatalogWriter::CatalogWriter(const std::string& path) {
    file_.open(path, std::ios::out | std::ios::trunc);
    if (file_.is_open()) {
        file_ << "frame,x,y,diff,snr\n";
    }
}

CatalogWriter::~CatalogWriter() {
    if (file_.is_open()) {
        file_.close();
    }
}

void CatalogWriter::writeFrame(uint32_t frameIndex, const std::vector<Detection>& detections) {
    if (!file_.is_open()) {
        return;
    }
    for (const Detection& d : detections) {
        file_ << frameIndex << "," << d.x << "," << d.y << ","
              << d.diffValue << "," << d.snr << "\n";
    }
}

} // namespace vkp
