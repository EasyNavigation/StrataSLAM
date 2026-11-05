#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "strataslam/common/result.hpp"

namespace strataslam {

/// Minimal point cloud for I/O (no PCL dependency in the core).
struct PointCloud {
  std::vector<Eigen::Vector3f> points;
  std::vector<std::array<uint8_t, 3>> colors;  ///< empty, or one RGB triple per point

  bool operator==(const PointCloud& o) const { return points == o.points && colors == o.colors; }
};

/// Binary PCD v0.7 (fields x y z [rgb]).
Result<void> write_pcd(const std::string& path, const PointCloud& cloud);
Result<PointCloud> read_pcd(const std::string& path);

/// Binary little-endian PLY (vertex x y z [red green blue]).
Result<void> write_ply(const std::string& path, const PointCloud& cloud);
Result<PointCloud> read_ply(const std::string& path);

}  // namespace strataslam
