#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "../pipeline/Detection.hpp"

namespace vkp {

// Appends detections to a CSV catalogue, one row per event. This is the text
// product a real survey pipeline emits alongside the imagery: a machine readable
// list of what was found and where.
class CatalogWriter {
public:
    explicit CatalogWriter(const std::string& path);
    ~CatalogWriter();

    bool isOpen() const { return file_.is_open(); }

    void writeFrame(uint32_t frameIndex, const std::vector<Detection>& detections);

private:
    std::ofstream file_;
};

} // namespace vkp
