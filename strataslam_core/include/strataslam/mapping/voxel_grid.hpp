#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "strataslam/io/point_cloud.hpp"

namespace strataslam {

/// Centroid voxel-grid downsampling (deterministic: output ordered by first appearance).
std::vector<Eigen::Vector3f> voxel_downsample(const std::vector<Eigen::Vector3f>& points, double voxel_size);

/// Same, averaging the colors of the points in each voxel. `colors` has one entry per point.
PointCloud voxel_downsample_colored(const std::vector<Eigen::Vector3f>& points,
                                    const std::vector<std::array<uint8_t, 3>>& colors, double voxel_size);

/// Transforms points by `T` (4x4 homogeneous, double).
std::vector<Eigen::Vector3f> transform_points(const std::vector<Eigen::Vector3f>& points,
                                              const Eigen::Matrix4d& T);

}  // namespace strataslam
