// Writes a small deterministic colored cloud as PCD and PLY, for external-compatibility tests
// (the Python suite reads them back with Open3D). Usage: write_sample_clouds <out_dir>
#include <cstdio>
#include <string>

#include "strataslam/io/point_cloud.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
    return 2;
  }
  strataslam::PointCloud c;
  for (int i = 0; i < 100; ++i) {
    c.points.emplace_back(0.5f * i, -0.25f * i, 0.125f * i);
    c.colors.push_back({uint8_t(i), uint8_t(2 * i), uint8_t(255 - i)});
  }
  const std::string dir = argv[1];
  const auto a = strataslam::write_pcd(dir + "/sample.pcd", c);
  const auto b = strataslam::write_ply(dir + "/sample.ply", c);
  if (!a || !b) {
    std::fprintf(stderr, "write failed\n");
    return 1;
  }
  return 0;
}
