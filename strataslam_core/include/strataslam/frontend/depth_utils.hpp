#pragma once

#include <optional>

#include <Eigen/Core>

#include "strataslam/sensors/measurements.hpp"

namespace strataslam {

struct BackprojectionConfig {
  double depth_min = 0.3;            ///< m
  double depth_max = 6.0;            ///< m
  double discontinuity_ratio = 0.05; ///< reject if a neighbour differs by more than this fraction of the depth
};

/// Depth (m) at the pixel nearest to (u, v); nullopt if outside the image, out of range, invalid,
/// or on a depth discontinuity (a valid 8-neighbour differing by more than the configured ratio).
std::optional<double> sample_depth(const DepthImage& depth, double u, double v, const BackprojectionConfig& cfg);

/// 3D point in the optical camera frame for a keypoint, with the same validity rules.
std::optional<Eigen::Vector3d> backproject(const DepthImage& depth, const CameraIntrinsics& K, double u, double v,
                                           const BackprojectionConfig& cfg);

/// Subsampled point cloud (every `stride` pixels) in the optical camera frame, for mapping.
std::vector<Eigen::Vector3f> depth_to_cloud(const DepthImage& depth, const CameraIntrinsics& K, int stride,
                                            const BackprojectionConfig& cfg);

}  // namespace strataslam
