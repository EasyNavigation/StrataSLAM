#include "strataslam/mapping/voxel_grid.hpp"

#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace strataslam {

std::vector<Eigen::Vector3f> voxel_downsample(const std::vector<Eigen::Vector3f>& points,
                                              double voxel_size) {
  struct Cell {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    int n = 0;
    size_t order = 0;
  };
  std::unordered_map<uint64_t, Cell> cells;
  cells.reserve(points.size());
  const double inv = 1.0 / voxel_size;
  size_t order = 0;
  for (const auto& p : points) {
    const int64_t ix = static_cast<int64_t>(std::floor(p.x() * inv)) + (1 << 20);
    const int64_t iy = static_cast<int64_t>(std::floor(p.y() * inv)) + (1 << 20);
    const int64_t iz = static_cast<int64_t>(std::floor(p.z() * inv)) + (1 << 20);
    const uint64_t key = (static_cast<uint64_t>(ix) & 0x1fffff) |
                         ((static_cast<uint64_t>(iy) & 0x1fffff) << 21) |
                         ((static_cast<uint64_t>(iz) & 0x1fffff) << 42);
    auto it = cells.find(key);
    if (it == cells.end()) it = cells.emplace(key, Cell{Eigen::Vector3d::Zero(), 0, order++}).first;
    it->second.sum += p.cast<double>();
    ++it->second.n;
  }
  std::vector<Eigen::Vector3f> out(cells.size());
  for (const auto& kv : cells) out[kv.second.order] = (kv.second.sum / kv.second.n).cast<float>();
  return out;
}

PointCloud voxel_downsample_colored(const std::vector<Eigen::Vector3f>& points,
                                    const std::vector<std::array<uint8_t, 3>>& colors,
                                    double voxel_size) {
  struct Cell {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero(), col = Eigen::Vector3d::Zero();
    int n = 0;
    size_t order = 0;
  };
  std::unordered_map<uint64_t, Cell> cells;
  cells.reserve(points.size());
  const double inv = 1.0 / voxel_size;
  size_t order = 0;
  for (size_t i = 0; i < points.size(); ++i) {
    const auto& p = points[i];
    const int64_t ix = static_cast<int64_t>(std::floor(p.x() * inv)) + (1 << 20);
    const int64_t iy = static_cast<int64_t>(std::floor(p.y() * inv)) + (1 << 20);
    const int64_t iz = static_cast<int64_t>(std::floor(p.z() * inv)) + (1 << 20);
    const uint64_t key = (static_cast<uint64_t>(ix) & 0x1fffff) |
                         ((static_cast<uint64_t>(iy) & 0x1fffff) << 21) |
                         ((static_cast<uint64_t>(iz) & 0x1fffff) << 42);
    auto it = cells.find(key);
    if (it == cells.end()) {
      Cell c;
      c.order = order++;
      it = cells.emplace(key, c).first;
    }
    it->second.sum += p.cast<double>();
    it->second.col += Eigen::Vector3d(colors[i][0], colors[i][1], colors[i][2]);
    ++it->second.n;
  }
  PointCloud out;
  out.points.resize(cells.size());
  out.colors.resize(cells.size());
  for (const auto& kv : cells) {
    out.points[kv.second.order] = (kv.second.sum / kv.second.n).cast<float>();
    const Eigen::Vector3d c = kv.second.col / kv.second.n;
    out.colors[kv.second.order] = {static_cast<uint8_t>(std::lround(c.x())),
                                   static_cast<uint8_t>(std::lround(c.y())),
                                   static_cast<uint8_t>(std::lround(c.z()))};
  }
  return out;
}

std::vector<Eigen::Vector3f> transform_points(const std::vector<Eigen::Vector3f>& points,
                                              const Eigen::Matrix4d& T) {
  std::vector<Eigen::Vector3f> out(points.size());
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  const Eigen::Vector3d t = T.block<3, 1>(0, 3);
  for (size_t i = 0; i < points.size(); ++i)
    out[i] = (R * points[i].cast<double>() + t).cast<float>();
  return out;
}

}  // namespace strataslam
