#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strataslam/common/result.hpp"
#include "strataslam/frontend/feature_extractor.hpp"

namespace strataslam {

/// 16-bit single-channel image (e.g. a TUM depth PNG, in units of the dataset's depth factor).
struct Image16 {
  int width = 0, height = 0;
  std::vector<uint16_t> data;  ///< row-major, width * height
};

/// Reads an 8-bit grayscale image from a PNG (colour converted by libpng with the weights 0.299,
/// 0.587, 0.114, as OpenCV's IMREAD_GRAYSCALE asks it to), a JPEG (converted by libjpeg) or a
/// binary 8-bit PGM (P5).
///
/// Lives in its own library, `strataslam::image_io` (libpng and libjpeg), and not in OpenCV's
/// imgcodecs, which on Ubuntu loads GDAL and poppler (GPL) into every process that links it
/// (ADR-0047).
Result<GrayImage> read_gray_image(const std::string& path);

/// Reads a 16-bit single-channel PNG (8-bit grayscale PNGs are widened).
Result<Image16> read_png16(const std::string& path);

}  // namespace strataslam
