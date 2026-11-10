#include "strataslam/mapping/colorize.hpp"

#include <gtest/gtest.h>

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
  const std::vector<Eigen::Vector3f> pts{
      {-0.5f, 0, 2.0f}, {0.5f, 0, 2.0f}, {0, 0, -1.0f}, {50.f, 0, 2.0f}, {0, 0, 40.f}};
  std::vector<bool> colored;
  const auto c = colorize_points(pts, gtsam::Pose3(), K(), two_color_image(), nullptr, 0.3, 20.0,
                                 {1, 2, 3}, &colored);
  EXPECT_EQ(c[0], (std::array<uint8_t, 3>{255, 0, 0}));  // left of the axis -> red
  EXPECT_EQ(c[1], (std::array<uint8_t, 3>{0, 0, 255}));  // right -> blue
  EXPECT_EQ(c[2], (std::array<uint8_t, 3>{1, 2, 3}));    // behind the camera
  EXPECT_EQ(c[3], (std::array<uint8_t, 3>{1, 2, 3}));    // outside the image
  EXPECT_EQ(c[4], (std::array<uint8_t, 3>{1, 2, 3}));    // beyond max depth
  EXPECT_TRUE(colored[0]);
  EXPECT_FALSE(colored[2]);
}

TEST(Colorize, ExtrinsicIsApplied) {
  // The lidar sits 1 m to the left of the camera: a point straight ahead of the lidar projects to
  // the right half of the image.
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
  const std::vector<Eigen::Vector3f> pts{{-0.2f, 0, 0.9f},
                                         {-0.2f, 0, 3.0f}};  // in front of / behind the wall
  std::vector<bool> colored;
  colorize_points(pts, gtsam::Pose3(), K(), two_color_image(), &d, 0.3, 20.0, {0, 0, 0}, &colored);
  EXPECT_TRUE(colored[0]);
  EXPECT_FALSE(colored[1]);
}

TEST(Colorize, MissingOrMismatchedImageLeavesEverythingUnknown) {
  std::vector<bool> colored;
  const auto c = colorize_points({{0, 0, 2.f}}, gtsam::Pose3(), K(), RgbImage{}, nullptr, 0.3, 20.0,
                                 {9, 9, 9}, &colored);
  EXPECT_EQ(c[0], (std::array<uint8_t, 3>{9, 9, 9}));
  EXPECT_FALSE(colored[0]);
}

// The depth of an RGB-D frame as a map cloud (ADR-0068): one point every `stride` pixels with a
// valid depth, in the frame asked for, each with the colour of its own pixel.
TEST(Colorize, ADepthMapCloudCarriesTheColoursOfItsPixels) {
  RgbdFrame f;
  f.K = K();
  f.rgb = two_color_image();
  f.depth.width = 64;
  f.depth.height = 48;
  f.depth.meters.assign(64 * 48, 2.0f);                   // a wall 2 m in front
  for (int u = 0; u < 64; ++u) f.depth.meters[u] = 0.0f;  // an invalid first row
  BackprojectionConfig cfg;
  // The camera 0.3 m above the base, looking along its x: optical z forward -> base x.
  const gtsam::Pose3 T_base_cam(
      gtsam::Rot3(Eigen::Matrix3d((Eigen::Matrix3d() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished())),
      gtsam::Point3(0, 0, 0.3));
  const PointCloud c = depth_map_cloud(f, 8, cfg, T_base_cam);
  ASSERT_FALSE(c.points.empty());
  ASSERT_EQ(c.colors.size(), c.points.size());
  for (size_t i = 0; i < c.points.size(); ++i) {
    EXPECT_NEAR(c.points[i].x(), 2.0f, 1e-4);  // the wall, 2 m ahead of the base
    // Left of the image (red) is the base's +y.
    const bool left = c.points[i].y() > 0;
    EXPECT_EQ(c.colors[i][0], left ? 255 : 0) << c.points[i].transpose();
    EXPECT_EQ(c.colors[i][2], left ? 0 : 255);
  }
  // Without a colour image: the points, no colours.
  f.rgb = RgbImage();
  const PointCloud g = depth_map_cloud(f, 8, cfg, T_base_cam);
  EXPECT_EQ(g.points.size(), c.points.size());
  EXPECT_TRUE(g.colors.empty());
}

// A point its own image did not see takes its colour from the first other view that sees it on the
// surface that view measured (ADR-0069): not through something nearer, not where the view saw
// farther.
TEST(Colorize, OtherViewsColourOnlyThePointsOnTheSurfaceTheyMeasured) {
  const std::vector<Eigen::Vector3f> pts{{-0.5f, 0, 2.0f}};  // left of the image: red, 2 m ahead
  auto view = [](float depth_m) {
    RgbdFrame f;
    f.K = K();
    f.rgb = two_color_image();
    f.depth.width = 64;
    f.depth.height = 48;
    f.depth.meters.assign(64 * 48, depth_m);
    return f;
  };
  const RgbdFrame nearer = view(1.0f), farther = view(4.0f), on_it = view(2.05f);
  std::vector<std::array<uint8_t, 3>> rgb(1, {128, 128, 128});
  std::vector<bool> seen(1, false);
  // Hidden behind a surface at 1 m, then where that view saw 4 m (the point was not there for it).
  colorize_from_views(pts, {{gtsam::Pose3(), &nearer}, {gtsam::Pose3(), &farther}}, 6.0, 0.3, &rgb,
                      &seen);
  EXPECT_FALSE(seen[0]);
  EXPECT_EQ(rgb[0][0], 128);
  // A view whose depth is the point's: its pixel's colour.
  colorize_from_views(pts, {{gtsam::Pose3(), &nearer}, {gtsam::Pose3(), &on_it}}, 6.0, 0.3, &rgb,
                      &seen);
  EXPECT_TRUE(seen[0]);
  EXPECT_EQ(rgb[0][0], 255);
  // Already coloured: later views do not change it.
  const RgbdFrame shifted = view(2.0f);
  colorize_from_views(pts, {{gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0, 0)), &shifted}}, 6.0,
                      0.3, &rgb, &seen);
  EXPECT_EQ(rgb[0][0], 255);
}

// A frame kept at 1/2 (ADR-0069): a point takes from it the colour it takes from the full frame,
// and its depth for the surface test is the full frame's at that pixel.
TEST(Colorize, ADecimatedFrameColoursAsTheFullOne) {
  RgbdFrame f;
  f.K = K();
  f.rgb = two_color_image();
  f.depth.width = 64;
  f.depth.height = 48;
  f.depth.meters.assign(64 * 48, 2.0f);
  const RgbdFrame h = decimate_frame(f, 2);
  EXPECT_EQ(h.K.width, 32);
  EXPECT_EQ(h.rgb.data.size(), 32u * 24u * 3u);
  EXPECT_EQ(h.depth.meters.size(), 32u * 24u);
  const std::vector<Eigen::Vector3f> pts{{-0.5f, 0.1f, 2.0f}, {0.5f, -0.1f, 2.0f}};
  std::vector<bool> a, b;
  const auto full = colorize_points(pts, gtsam::Pose3(), f.K, f.rgb, &f.depth, 0.3, 20.0,
                                    {128, 128, 128}, &a, true);
  const auto half = colorize_points(pts, gtsam::Pose3(), h.K, h.rgb, &h.depth, 0.3, 20.0,
                                    {128, 128, 128}, &b, true);
  EXPECT_EQ(full, half);
  EXPECT_EQ(a, b);
  EXPECT_TRUE(a[0] && a[1]);
  EXPECT_EQ(decimate_frame(f, 1).rgb.data, f.rgb.data);
}
