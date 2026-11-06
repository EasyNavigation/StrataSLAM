#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>

#include "strataslam/frontend/lidar_frontend.hpp"
#include "strataslam/mapping/voxel_grid.hpp"
#include "strataslam/place_recognition/scan_context.hpp"

using namespace strataslam;
using Eigen::Vector3d;
using Eigen::Vector3f;

namespace {

struct Plane {
  Vector3d o, u, v;
  double su, sv;
};

/// Random samples on the inner faces of an axis-aligned room [lo, hi], plus extra box obstacles.
std::vector<Vector3f> sample_room(const Vector3d& lo, const Vector3d& hi, int n_per_m2,
                                  unsigned seed,
                                  const std::vector<std::pair<Vector3d, Vector3d>>& boxes = {}) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0, 1);
  std::vector<Vector3f> pts;
  const auto face = [&](const Vector3d& o, const Vector3d& a, const Vector3d& b) {
    const int n = static_cast<int>(a.norm() * b.norm() * n_per_m2);
    for (int i = 0; i < n; ++i) pts.push_back((o + u(rng) * a + u(rng) * b).cast<float>());
  };
  const Vector3d d = hi - lo;
  face(lo, Vector3d(d.x(), 0, 0), Vector3d(0, d.y(), 0));                                // floor
  face(Vector3d(lo.x(), lo.y(), hi.z()), Vector3d(d.x(), 0, 0), Vector3d(0, d.y(), 0));  // ceiling
  face(lo, Vector3d(d.x(), 0, 0), Vector3d(0, 0, d.z()));                                // y = lo
  face(Vector3d(lo.x(), hi.y(), lo.z()), Vector3d(d.x(), 0, 0), Vector3d(0, 0, d.z()));
  face(lo, Vector3d(0, d.y(), 0), Vector3d(0, 0, d.z()));  // x = lo
  face(Vector3d(hi.x(), lo.y(), lo.z()), Vector3d(0, d.y(), 0), Vector3d(0, 0, d.z()));
  for (const auto& [blo, bhi] : boxes) {
    const Vector3d e = bhi - blo;
    face(blo, Vector3d(e.x(), 0, 0), Vector3d(0, 0, e.z()));
    face(Vector3d(blo.x(), bhi.y(), blo.z()), Vector3d(e.x(), 0, 0), Vector3d(0, 0, e.z()));
    face(blo, Vector3d(0, e.y(), 0), Vector3d(0, 0, e.z()));
    face(Vector3d(bhi.x(), blo.y(), blo.z()), Vector3d(0, e.y(), 0), Vector3d(0, 0, e.z()));
    face(Vector3d(blo.x(), blo.y(), bhi.z()), Vector3d(e.x(), 0, 0), Vector3d(0, e.y(), 0));
  }
  return pts;
}

std::vector<Vector3f> room() {
  return sample_room({-4, -3, 0}, {4, 3, 3}, 60, 1,
                     {{{1, 0.5, 0}, {2, 1.5, 1.2}},
                      {{-3, -2, 0}, {-2.2, -1.2, 2.0}},
                      {{-1, 1.8, 0}, {0.2, 2.6, 0.7}}});
}

}  // namespace

TEST(LidarRegistration, RecoversKnownTransformInARoom) {
  const auto target = room();
  const gtsam::Pose3 T_true(gtsam::Rot3::RzRyRx(0.01, -0.01, 0.08), Vector3d(0.30, -0.20, 0.05));
  // Source = same surfaces seen from a displaced sensor (a different random sampling).
  auto src_world = sample_room({-4, -3, 0}, {4, 3, 3}, 60, 2,
                               {{{1, 0.5, 0}, {2, 1.5, 1.2}},
                                {{-3, -2, 0}, {-2.2, -1.2, 2.0}},
                                {{-1, 1.8, 0}, {0.2, 2.6, 0.7}}});
  const auto source =
      transform_points(src_world, T_true.inverse().matrix());  // source frame = displaced
  LidarFrontendConfig cfg;
  cfg.scan_voxel = 0.15;
  cfg.max_correspondence = 1.0;
  const gtsam::Pose3 init(gtsam::Rot3::RzRyRx(0, 0, 0.05),
                          Vector3d(0.15, -0.1, 0.0));  // off by ~0.2 m, 3 deg
  const auto out = LidarFrontend::register_clouds(target, source, init, cfg);
  EXPECT_TRUE(out.converged);
  const gtsam::Pose3 err = out.T_target_source.between(T_true);
  EXPECT_LT(err.translation().norm(), 0.01);
  EXPECT_LT(gtsam::Rot3::Logmap(err.rotation()).norm() * 180 / M_PI, 0.1);
  EXPECT_EQ(out.quality.state, TrackingState::Good);
  EXPECT_GT(out.quality.fitness, 0.7);
}

TEST(LidarRegistration, LongCorridorIsDegradedRoomIsGood) {
  LidarFrontendConfig cfg;
  cfg.scan_voxel = 0.15;
  cfg.max_correspondence = 1.0;
  const auto corridor = sample_room({-20, -1, 0}, {20, 1, 2.5}, 60, 3);
  const auto corridor2 = sample_room({-20, -1, 0}, {20, 1, 2.5}, 60, 4);
  const auto shifted =
      transform_points(corridor2, gtsam::Pose3(gtsam::Rot3(), Vector3d(-0.3, 0, 0)).matrix());
  const auto rc = LidarFrontend::register_clouds(corridor, shifted, gtsam::Pose3(), cfg);
  EXPECT_EQ(rc.quality.state, TrackingState::Degraded);
  EXPECT_LT(rc.quality.min_eigenvalue, cfg.degeneracy_threshold);

  const auto r = room();
  const auto r2 = sample_room({-4, -3, 0}, {4, 3, 3}, 60, 5,
                              {{{1, 0.5, 0}, {2, 1.5, 1.2}},
                               {{-3, -2, 0}, {-2.2, -1.2, 2.0}},
                               {{-1, 1.8, 0}, {0.2, 2.6, 0.7}}});
  const auto rr = LidarFrontend::register_clouds(r, r2, gtsam::Pose3(), cfg);
  EXPECT_EQ(rr.quality.state, TrackingState::Good);
  EXPECT_GT(rr.quality.min_eigenvalue, 5 * rc.quality.min_eigenvalue);
}

TEST(LidarRegistration, UnrelatedCloudsAreLost) {
  LidarFrontendConfig cfg;
  cfg.scan_voxel = 0.15;
  const auto a = room();
  const auto b =
      transform_points(sample_room({-40, -30, 0}, {40, 30, 8}, 1, 9), gtsam::Pose3().matrix());
  const auto out = LidarFrontend::register_clouds(a, b, gtsam::Pose3(), cfg);
  EXPECT_EQ(out.quality.state, TrackingState::Lost);
}

TEST(Voxel, DownsampleAveragesAndIsDeterministic) {
  std::vector<Vector3f> pts{{0.01f, 0, 0}, {0.03f, 0, 0}, {1.01f, 0, 0}, {1.02f, 0.01f, 0}};
  const auto out = voxel_downsample(pts, 0.5);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_NEAR(out[0].x(), 0.02f, 1e-6);
  EXPECT_NEAR(out[1].x(), 1.015f, 1e-6);
  EXPECT_EQ(voxel_downsample(pts, 0.5), out);
}

// ---------------- Scan Context ----------------

namespace {
std::vector<Vector3f> asymmetric_scene(unsigned seed) {
  // An off-centre corridor-like structure with distinct tall and short obstacles at different
  // bearings.
  auto pts = sample_room({-12, -6, -1.5}, {14, 9, 1.5}, 12, seed,
                         {{{4, 1, -1.5}, {5, 2, 1.0}},
                          {{-6, -3, -1.5}, {-4, -2, 0.0}},
                          {{-2, 5, -1.5}, {-1, 6, 1.4}},
                          {{8, -4, -1.5}, {10, -3.5, -0.2}}});
  return pts;
}
}  // namespace

TEST(ScanContext, RotatedScanMatchesWithRecoveredYaw) {
  ScanContextConfig cfg;
  cfg.max_range = 20;
  const auto pts = asymmetric_scene(1);
  const ScanContext a = make_scan_context(pts, cfg);
  for (double yaw : {0.0, 0.5, -1.3, 2.9}) {
    // The query is the same place seen with the sensor rotated by `yaw`: points rotate by -yaw.
    const auto rotated =
        transform_points(pts, gtsam::Pose3(gtsam::Rot3::Yaw(-yaw), Vector3d::Zero()).matrix());
    const ScanContext b = make_scan_context(rotated, cfg);
    const ScanContextMatch m = compare_scan_contexts(b, a);
    EXPECT_LT(m.distance, 0.08) << yaw;
    const double d = std::atan2(std::sin(m.yaw - yaw), std::cos(m.yaw - yaw));
    EXPECT_LT(std::abs(d), 2.0 * M_PI / cfg.sectors) << "yaw=" << yaw << " got " << m.yaw;
  }
}

TEST(ScanContext, DifferentPlacesAreFarAndRingKeyIsRotationInvariant) {
  ScanContextConfig cfg;
  cfg.max_range = 20;
  const ScanContext a = make_scan_context(asymmetric_scene(1), cfg);
  const ScanContext c = make_scan_context(sample_room({-3, -2, -1.5}, {3, 2, 1.5}, 40, 7), cfg);
  EXPECT_GT(compare_scan_contexts(c, a).distance, 0.2);
  const auto rot = transform_points(asymmetric_scene(1),
                                    gtsam::Pose3(gtsam::Rot3::Yaw(1.1), Vector3d::Zero()).matrix());
  const ScanContext b = make_scan_context(rot, cfg);
  EXPECT_LT((a.ring_key - b.ring_key).norm(), 0.15);
}

TEST(ScanContext, DatabaseFindsRevisitAndSkipsRecentEntries) {
  ScanContextConfig cfg;
  cfg.max_range = 20;
  cfg.exclude_recent = 5;
  ScanContextDatabase db(cfg);
  for (int i = 0; i < 12; ++i) {
    const unsigned seed = 100 + i;
    db.add(i, make_scan_context(sample_room({-3.0 - 0.7 * i, -2, -1.5}, {3, 2.0 + 0.4 * i, 1.5}, 20,
                                            seed, {{{0.5 * i, 0, -1.5}, {0.5 * i + 1, 1, 1.0}}}),
                                cfg));
  }
  // Query = place 3 re-observed with the sensor rotated by 0.7 rad.
  const auto pts3 = sample_room({-3.0 - 0.7 * 3, -2, -1.5}, {3, 2.0 + 0.4 * 3, 1.5}, 20, 555,
                                {{{1.5, 0, -1.5}, {2.5, 1, 1.0}}});
  const auto q = make_scan_context(
      transform_points(pts3, gtsam::Pose3(gtsam::Rot3::Yaw(-0.7), Vector3d::Zero()).matrix()), cfg);
  const auto res = db.query(q);
  ASSERT_FALSE(res.empty());
  EXPECT_EQ(res.front().id, 3u);
  EXPECT_NEAR(res.front().match.yaw, 0.7, 2.0 * M_PI / cfg.sectors);
  // The same query against a database containing only recent entries finds nothing.
  ScanContextDatabase small(cfg);
  for (int i = 0; i < 4; ++i) small.add(i, make_scan_context(pts3, cfg));
  EXPECT_TRUE(small.query(q).empty());
}

TEST(ScanContext, PayloadRoundTrip) {
  ScanContextConfig cfg;
  const ScanContext a = make_scan_context(asymmetric_scene(1), cfg);
  const auto p = ScanContextDatabase::to_payload(a, 1234, 40.f);
  const ScanContext b = ScanContextDatabase::from_payload(p);
  EXPECT_EQ(a.matrix, b.matrix);
  EXPECT_EQ(a.ring_key, b.ring_key);
}

// ---------------- LiDAR attitude bounded by the IMU's gravity (ADR-0058) ----------------

namespace {

/// Up in the base frame of the pose the frontend gives to a second scan of the room, taken from
/// where the first one was (a level base), when the IMU reports `up_base` (none: unbounded).
Vector3d up_after_second_scan(const std::optional<Vector3d>& up_base) {
  LidarFrontend fe{LidarFrontendConfig()};
  LidarScan scan;
  scan.points = room();
  SensorFrame f;
  f.lidar = &scan;
  scan.stamp = f.stamp = 1'000'000'000LL;
  EXPECT_TRUE(fe.process(f).valid);
  scan.stamp = f.stamp = 1'100'000'000LL;
  f.up_base = up_base;
  const FrontendResult r = fe.process(f);
  EXPECT_TRUE(r.valid);
  EXPECT_LT(r.T_odom_base.translation().norm(), 0.02);  // the leveling keeps the position
  return r.T_odom_base.rotation().matrix().transpose() * Vector3d::UnitZ();
}

double angle_deg(const Vector3d& a, const Vector3d& b) {
  return std::acos(std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0)) * 180.0 / M_PI;
}

}  // namespace

TEST(LidarFrontend, TheImusGravityBoundsTheAttitudeToOneDegree) {
  const Vector3d level = Vector3d::UnitZ();
  // Without the IMU: the scan's own attitude, level.
  EXPECT_LT(angle_deg(up_after_second_scan(std::nullopt), level), 0.1);
  // The IMU 0.5 deg away: within the bound, the LiDAR's attitude stands.
  const Vector3d close = gtsam::Rot3::Pitch(0.5 * M_PI / 180).matrix() * level;
  EXPECT_LT(angle_deg(up_after_second_scan(close), level), 0.1);
  // The IMU 5 deg away: the attitude is moved to 1 deg from it, towards where the LiDAR had it.
  const Vector3d tilted = gtsam::Rot3::Pitch(5.0 * M_PI / 180).matrix() * level;
  const Vector3d up = up_after_second_scan(tilted);
  EXPECT_NEAR(angle_deg(up, tilted), 1.0, 0.05);
  EXPECT_NEAR(angle_deg(up, level), 4.0, 0.1);
}
