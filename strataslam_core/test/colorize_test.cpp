#include <gtest/gtest.h>

#include "strataslam/mapping/colorize.hpp"

using namespace strataslam;
using Eigen::Vector3d;

namespace {
CameraIntrinsics K() {
  CameraIntrinsics k;
  k.width = 64;
  k.height = 48;
  k.fx = k.fy = 50;
  k.cx = 31.5;
  k.cy = 23.5;
  return k;
}
/// Left half red, right half blue.
RgbImage two_color_image() {
  RgbImage im;
  im.width = 64;
  im.height = 48;
  im.data.resize(64 * 48 * 3);
  for (int v = 0; v < 48; ++v)
    for (int u = 0; u < 64; ++u) {
      const size_t i = (static_cast<size_t>(v) * 64 + u) * 3;
      im.data[i] = u < 32 ? 255 : 0;
      im.data[i + 1] = 0;
      im.data[i + 2] = u < 32 ? 0 : 255;
    }
  return im;
}
}  // namespace

TEST(Colorize, PointsTakeTheColorOfTheirPixel) {
  // Camera frame == lidar frame here: optical axes (x right, y down, z forward).
  const std::vector<Eigen::Vector3f> pts{{-0.5f, 0, 2.0f}, {0.5f, 0, 2.0f}, {0, 0, -1.0f}, {50.f, 0, 2.0f}, {0, 0, 40.f}};
  std::vector<bool> colored;
  const auto c = colorize_points(pts, gtsam::Pose3(), K(), two_color_image(), nullptr, 0.3, 20.0, {1, 2, 3}, &colored);
  EXPECT_EQ(c[0], (std::array<uint8_t, 3>{255, 0, 0}));  // left of the axis -> red
  EXPECT_EQ(c[1], (std::array<uint8_t, 3>{0, 0, 255}));  // right -> blue
  EXPECT_EQ(c[2], (std::array<uint8_t, 3>{1, 2, 3}));    // behind the camera
  EXPECT_EQ(c[3], (std::array<uint8_t, 3>{1, 2, 3}));    // outside the image
  EXPECT_EQ(c[4], (std::array<uint8_t, 3>{1, 2, 3}));    // beyond max depth
  EXPECT_TRUE(colored[0]);
  EXPECT_FALSE(colored[2]);
}

TEST(Colorize, ExtrinsicIsApplied) {
  // The lidar sits 1 m to the left of the camera: a point straight ahead of the lidar projects to the
  // right half of the image.
  const gtsam::Pose3 T_cam_lidar(gtsam::Rot3(), Vector3d(1.0, 0, 0));  // lidar x -> camera x + 1 m
  const std::vector<Eigen::Vector3f> pts{{0.f, 0.f, 2.f}};
  const auto c = colorize_points(pts, T_cam_lidar, K(), two_color_image());
  EXPECT_EQ(c[0], (std::array<uint8_t, 3>{0, 0, 255}));
}

TEST(Colorize, OccludedPointsAreNotColored) {
  DepthImage d;
  d.width = 64;
  d.height = 48;
  d.meters.assign(64 * 48, 1.0f);  // a wall at 1 m everywhere
  const std::vector<Eigen::Vector3f> pts{{-0.2f, 0, 0.9f}, {-0.2f, 0, 3.0f}};  // in front of / behind the wall
  std::vector<bool> colored;
  colorize_points(pts, gtsam::Pose3(), K(), two_color_image(), &d, 0.3, 20.0, {0, 0, 0}, &colored);
  EXPECT_TRUE(colored[0]);
  EXPECT_FALSE(colored[1]);
}

TEST(Colorize, MissingOrMismatchedImageLeavesEverythingUnknown) {
  std::vector<bool> colored;
  const auto c = colorize_points({{0, 0, 2.f}}, gtsam::Pose3(), K(), RgbImage{}, nullptr, 0.3, 20.0, {9, 9, 9}, &colored);
  EXPECT_EQ(c[0], (std::array<uint8_t, 3>{9, 9, 9}));
  EXPECT_FALSE(colored[0]);
}
