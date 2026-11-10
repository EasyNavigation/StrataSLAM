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

// ---------------- Confined spaces: the rotation is the gyroscope's ----------------

namespace {

/// An elevator cabin, 1.6 x 1.4 x 2.4 m, around the sensor (a handrail on one wall).
std::vector<Vector3f> cabin() {
  return sample_room({-0.8, -0.7, -1.2}, {0.8, 0.7, 1.2}, 400, 3,
                     {{{0.7, -0.5, -0.3}, {0.8, 0.5, -0.25}}});
}

}  // namespace

TEST(LidarFrontend, InAConfinedSpaceTheRotationIsTheGyroscopesAndTheTranslationIsRegistered) {
  LidarFrontendConfig cfg;
  cfg.min_range = 0.3;
  LidarFrontend fe{cfg};
  LidarScan scan;
  scan.points = cabin();
  SensorFrame f;
  f.lidar = &scan;
  scan.stamp = f.stamp = 1'000'000'000LL;
  ASSERT_TRUE(fe.process(f).valid);
  // Turned by 25 deg and moved 10 cm: the gyroscope reports the turn 1 deg short.
  const gtsam::Pose3 moved(gtsam::Rot3::Yaw(25.0 * M_PI / 180), Vector3d(0.10, 0.0, 0.0));
  scan.points = transform_points(cabin(), moved.inverse().matrix());
  scan.stamp = f.stamp = 1'100'000'000LL;
  f.gyro_delta = gtsam::Rot3::Yaw(24.0 * M_PI / 180);
  const FrontendResult r = fe.process(f);
  ASSERT_TRUE(r.valid);
  EXPECT_TRUE(r.confined);
  EXPECT_TRUE(r.T_odom_base.translation().allFinite());
  EXPECT_NEAR(gtsam::Rot3::Logmap(r.T_odom_base.rotation()).z() * 180 / M_PI, 24.0, 0.05);
  EXPECT_LT((r.T_odom_base.translation() - Vector3d(0.10, 0, 0)).norm(), 0.05);
}

TEST(LidarFrontend, OutOfConfinedSpacesTheGyroscopeIsNotUsed) {
  LidarFrontend a{LidarFrontendConfig()}, b{LidarFrontendConfig()};
  LidarScan scan;
  scan.points = room();
  SensorFrame f;
  f.lidar = &scan;
  scan.stamp = f.stamp = 1'000'000'000LL;
  ASSERT_TRUE(a.process(f).valid);
  ASSERT_TRUE(b.process(f).valid);
  const gtsam::Pose3 moved(gtsam::Rot3::Yaw(0.1), Vector3d(0.2, 0.1, 0.0));
  scan.points = transform_points(room(), moved.inverse().matrix());
  scan.stamp = f.stamp = 1'100'000'000LL;
  const FrontendResult ra = a.process(f);
  f.gyro_delta = gtsam::Rot3::Yaw(0.3);  // wrong on purpose: not used in the open
  const FrontendResult rb = b.process(f);
  EXPECT_FALSE(rb.confined);
  EXPECT_TRUE(ra.T_odom_base.equals(rb.T_odom_base, 1e-12));
}

// ---------------- Degenerate registration: bounded by a physical acceleration ----------------

namespace {

/// Where the frontend puts a sensor that stood still for two scans and then shows up `jump`
/// metres along a corridor in the next one, 0.1 s later. Along the corridor only a small box
/// constrains the registration: it is `Degraded`, but sees the jump.
Vector3d after_a_jump(double jump, double max_acceleration, TrackingState* state) {
  LidarFrontendConfig cfg;
  cfg.max_acceleration = max_acceleration;
  LidarFrontend fe{cfg};
  const auto corridor =
      sample_room({-20, -1, -1.2}, {20, 1, 1.3}, 60, 7, {{{1.0, 0.6, -1.2}, {1.3, 1.0, -0.9}}});
  LidarScan scan;
  scan.points = corridor;
  SensorFrame f;
  f.lidar = &scan;
  for (int i = 0; i < 2; ++i) {
    scan.stamp = f.stamp = 1'000'000'000LL + i * 100'000'000LL;
    EXPECT_TRUE(fe.process(f).valid);
  }
  scan.points =
      transform_points(corridor, gtsam::Pose3(gtsam::Rot3(), Vector3d(-jump, 0, 0)).matrix());
  scan.stamp = f.stamp = 1'200'000'000LL;
  const FrontendResult r = fe.process(f);
  EXPECT_TRUE(r.valid);
  *state = r.quality.state;
  return r.T_odom_base.translation();
}

}  // namespace

TEST(LidarFrontend, ADegenerateRegistrationMovesNoFurtherThanAPhysicalAccelerationAllows) {
  TrackingState state;
  // Without the bound the registration follows the box: the jump is seen.
  const Vector3d free = after_a_jump(0.8, 0.0, &state);
  ASSERT_EQ(state, TrackingState::Degraded);
  ASSERT_NEAR(free.x(), 0.8, 0.1);
  // 30 m/s^2 over 0.1 s: 0.3 m along the corridor; across it nothing changes.
  const Vector3d bounded = after_a_jump(0.8, 30.0, &state);
  EXPECT_NEAR(bounded.x(), 0.3, 0.02);
  EXPECT_NEAR(bounded.y(), free.y(), 0.02);
  EXPECT_NEAR(bounded.z(), free.z(), 0.02);
  // A step a sensor can make (0.1 m from standing, 20 m/s^2) is taken as registered.
  const Vector3d step = after_a_jump(0.1, 30.0, &state);
  EXPECT_NEAR(step.x(), 0.1, 0.03);
}

// ---------------- A planar scanner (lidar.type: 2d, ADR-0066) ----------------

namespace {

/// The walls of a room of 8 x 6 m with three boxes (tables, a cupboard), as 2D segments.
std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> floor_plan() {
  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> s;
  const auto box = [&](double x0, double y0, double x1, double y1) {
    s.push_back({{x0, y0}, {x1, y0}});
    s.push_back({{x1, y0}, {x1, y1}});
    s.push_back({{x1, y1}, {x0, y1}});
    s.push_back({{x0, y1}, {x0, y0}});
  };
  box(-4, -3, 4, 3);
  box(1.0, 0.5, 2.0, 1.5);
  box(-3.0, -2.0, -2.2, -1.2);
  box(-1.0, 1.8, 0.2, 2.6);
  return s;
}

/// A sweep of 190 deg in 0.33 deg steps (a Tiago's base laser) from (x, y, heading) in the plan:
/// the nearest wall along each ray, in the scanner's frame (z = 0), up to 10 m.
std::vector<Vector3f> sweep(double x, double y, double heading) {
  std::vector<Vector3f> pts;
  const auto plan = floor_plan();
  for (double a = -95.0; a <= 95.0; a += 0.333) {
    const double th = heading + a * M_PI / 180.0;
    const Eigen::Vector2d o(x, y), d(std::cos(th), std::sin(th));
    double best = 10.0;
    for (const auto& [p, q] : plan) {
      const Eigen::Vector2d e = q - p;
      const double den = d.x() * e.y() - d.y() * e.x();
      if (std::abs(den) < 1e-12) continue;
      const Eigen::Vector2d w = p - o;
      const double t = (w.x() * e.y() - w.y() * e.x()) / den;
      const double u = (w.x() * d.y() - w.y() * d.x()) / den;
      if (t > 0.05 && u >= 0.0 && u <= 1.0) best = std::min(best, t);
    }
    if (best < 10.0) {
      const double r = best, b = a * M_PI / 180.0;
      pts.emplace_back(static_cast<float>(r * std::cos(b)), static_cast<float>(r * std::sin(b)),
                       0.f);
    }
  }
  return pts;
}

LidarFrontendConfig planar_config() {
  LidarFrontendConfig c;
  c.planar_scan = true;
  c.scan_voxel = c.map_voxel = c.keyframe_cloud_voxel = 0.05;
  c.max_correspondence = 0.5;
  c.min_points = 30;
  c.min_range = 0.1;
  c.max_range = 10.0;
  return c;
}

}  // namespace

TEST(PlanarScan, TheRegistrationRecoversTheMotionInThePlaneAndHoldsTheRest) {
  // Two sweeps 0.3 m, -0.2 m and 8 deg apart; the guess 0.1 m and 3 deg off in the plane, and with
  // a height, roll and pitch (the prediction's) that nothing in the sweeps measures.
  const auto target = sweep(0.0, 0.0, 0.0);
  const auto source = sweep(0.30, -0.20, 8.0 * M_PI / 180.0);
  const double roll = 0.5 * M_PI / 180.0, pitch = -0.3 * M_PI / 180.0;
  const gtsam::Pose3 init(gtsam::Rot3::Ypr(5.0 * M_PI / 180.0, pitch, roll),
                          Vector3d(0.22, -0.14, 0.02));
  const auto out = LidarFrontend::register_clouds(target, source, init, planar_config());
  ASSERT_TRUE(out.converged);
  EXPECT_EQ(out.quality.state, TrackingState::Good);
  const gtsam::Pose3& T = out.T_target_source;
  EXPECT_NEAR(T.x(), 0.30, 0.01);
  EXPECT_NEAR(T.y(), -0.20, 0.01);
  EXPECT_NEAR(T.rotation().yaw() * 180.0 / M_PI, 8.0, 0.2);
  // They stay at the guess (the copies stacked for the registration would otherwise "measure" a
  // roll and pitch of walls they only assume vertical), up to the coupling of turning about the
  // sensor's own z, tilted by the guess: 0.2 mm and 0.03 deg here; freed, they would go to 0.
  EXPECT_NEAR(T.z(), 0.02, 1e-3);
  EXPECT_NEAR(T.rotation().roll(), roll, 0.05 * M_PI / 180.0);
  EXPECT_NEAR(T.rotation().pitch(), pitch, 0.05 * M_PI / 180.0);
}

TEST(PlanarScan, ASweepAlongOneWallIsDegradedNotGood) {
  // Facing a long wall at 1 m with nothing else in reach: the motion along it is unobserved.
  std::vector<Vector3f> wall;
  for (double y = -3.0; y <= 3.0; y += 0.02) wall.emplace_back(1.0f, static_cast<float>(y), 0.f);
  const auto out = LidarFrontend::register_clouds(wall, wall, gtsam::Pose3(), planar_config());
  EXPECT_EQ(out.quality.state, TrackingState::Degraded);
}

TEST(PlanarScan, TheFrontendTracksAPathThroughTheRoom) {
  // 4 s at 10 Hz: forward along x, then turning at 0.6 rad/s.
  LidarFrontend fe{planar_config()};
  double x = -1.5, y = 0.0, th = 0.0;
  FrontendResult r;
  for (int k = 0; k <= 40; ++k) {
    if (k > 0) {
      if (k <= 20) {
        x += 0.05;
      } else {
        th += 0.06;
        x += 0.04 * std::cos(th);
        y += 0.04 * std::sin(th);
      }
    }
    LidarScan scan;
    scan.points = sweep(x, y, th);
    scan.stamp = 1'000'000'000LL + k * 100'000'000LL;
    SensorFrame f;
    f.lidar = &scan;
    f.stamp = scan.stamp;
    r = fe.process(f);
    ASSERT_TRUE(r.valid) << "sweep " << k;
  }
  // The frontend's frame is the first sweep's: the truth relative to it.
  const gtsam::Pose3 start(gtsam::Rot3(), Vector3d(-1.5, 0.0, 0.0));
  const gtsam::Pose3 truth = start.between(gtsam::Pose3(gtsam::Rot3::Yaw(th), Vector3d(x, y, 0)));
  const gtsam::Pose3 err = r.T_odom_base.between(truth);
  EXPECT_LT(err.translation().norm(), 0.03);
  EXPECT_LT(std::abs(err.rotation().yaw()) * 180.0 / M_PI, 0.5);
  EXPECT_NEAR(r.T_odom_base.z(), 0.0, 1e-3);
}

TEST(PlanarScan, ASweepOfAFewCellsIsTrackedNotDropped) {
  // In a corner, a metre from both walls: some 50 cells of 5 cm, which a 3D LiDAR's scan never is
  // (below its 100 the scan is not tracked; on slam_test the Tiago turned 16 deg unmeasured there).
  std::vector<Vector3f> corner;
  for (double y = -0.5; y <= 1.0; y += 0.02) corner.emplace_back(1.0f, static_cast<float>(y), 0.f);
  for (double x = 0.0; x < 1.0; x += 0.02) corner.emplace_back(static_cast<float>(x), 1.0f, 0.f);
  const auto second_valid = [&](const LidarFrontendConfig& cfg) {
    LidarFrontend fe{cfg};
    LidarScan scan;
    scan.points = corner;
    SensorFrame f;
    f.lidar = &scan;
    scan.stamp = f.stamp = 1'000'000'000LL;
    fe.process(f);
    scan.stamp = f.stamp = 1'100'000'000LL;
    return fe.process(f).valid;
  };
  EXPECT_TRUE(second_valid(planar_config()));
  LidarFrontendConfig as_3d = planar_config();
  as_3d.min_points = 100;
  EXPECT_FALSE(second_valid(as_3d));
}
