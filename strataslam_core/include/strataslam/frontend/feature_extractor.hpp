#pragma once

#include <memory>
#include <string>
#include <vector>

#include "strataslam/sensors/measurements.hpp"

namespace strataslam {

/// Grayscale 8-bit image.
struct GrayImage {
  int width = 0, height = 0;
  std::vector<uint8_t> data;
};

/// Feature extraction is separate from tracking so that each can be tested alone.
class FeatureExtractor {
 public:
  virtual ~FeatureExtractor() = default;
  /// Keypoints with 256-bit binary descriptors, strongest first, at most `max_features`. With a
  /// `mask` (width*height bytes, non-zero = allowed) keypoints are only detected where it is set.
  virtual std::vector<RgbKeypoint> extract(const GrayImage& image,
                                           const std::vector<uint8_t>* mask = nullptr) = 0;
  virtual std::string name() const = 0;
};

/// ORB (OpenCV). XFeat through ONNX Runtime is optional (STRATASLAM_WITH_ONNX) and not built here.
std::unique_ptr<FeatureExtractor> make_orb_extractor(int max_features);

/// Mask of the pixels whose depth is within [depth_min, depth_max] (for `extract`): an RGB-D
/// tracker can only back-project keypoints that have a depth.
std::vector<uint8_t> valid_depth_mask(const DepthImage& depth, double depth_min, double depth_max);

}  // namespace strataslam
