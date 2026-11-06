#include "strataslam/sensors/imu_mount.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "strataslam/common/logging.hpp"
#include "strataslam/config/config.hpp"
#include "strataslam/pipeline/slam_system.hpp"
#include "strataslam/sensors/barometer.hpp"

using namespace strataslam;
using Eigen::Vector3d;

namespace {

/// A ground robot turning left and right on a level floor for 40 s, with an IMU mounted as
/// `R_base_imu` whose driver reports gravity times `sign` and `scale`; the odometry reports the
/// base's rotation every 0.1 s.
ImuMount::Estimate learn(const gtsam::Rot3& R_base_imu_true, const gtsam::Rot3& R_base_imu_tf,
                         double sign, double scale) {
  ImuMount mount(R_base_imu_tf);
  const double hz = 100.0;
  const Vector3d f_base(0, 0, kStandardGravity);  // specific force at rest, base frame (level)
  Timestamp last_odom = 0;
  Vector3d turned = Vector3d::Zero();
  for (int i = 0; i <= 40 * hz; ++i) {
    const double t = i / hz;
    const Timestamp stamp = 1'000'000'000LL + static_cast<Timestamp>(t * 1e9);
    const Vector3d w_base(0.01 * std::sin(3 * t), 0.01 * std::cos(2 * t),
                          0.6 * std::sin(0.4 * t));  // turns, some wobble
    ImuSample s;
    s.stamp = stamp;
    s.gyro = R_base_imu_true.matrix().transpose() * w_base;
    s.accel = sign * scale * (R_base_imu_true.matrix().transpose() * f_base);
    mount.add_imu(s);
    if (i > 0) turned += w_base / hz;
    if (i % 10 == 0) {
      if (last_odom) mount.add_base_rotation(last_odom, stamp, turned, Vector3d::UnitZ());
      last_odom = stamp;
      turned.setZero();
    }
  }
  const auto e = mount.estimate();
  EXPECT_TRUE(e.has_value());
  return e.value_or(ImuMount::Estimate{});
}

}  // namespace

TEST(ImuMount, ACorrectTfAndTheRosConventionAreConfirmed) {
  const gtsam::Rot3 R = gtsam::Rot3::Yaw(-M_PI / 2);
  const auto e = learn(R, R, 1.0, 1.0);
  EXPECT_LT(e.tf_error_deg, 1.0);
  EXPECT_EQ(e.accel_sign, 1.0);
  EXPECT_NEAR(e.accel_scale, 1.0, 1e-4);
}

TEST(ImuMount, ADriverThatReportsGravityPointingDownIsDetected) {
  const gtsam::Rot3 R =
      gtsam::Rot3::Yaw(-M_PI / 2);  // the PAL Tiago's base IMU: -10.2 m/s^2 on z at rest
  const auto e = learn(R, R, -1.0, 10.2 / kStandardGravity);
  EXPECT_LT(e.tf_error_deg, 1.0);
  EXPECT_EQ(e.accel_sign, -1.0);
  EXPECT_NEAR(e.accel_scale, kStandardGravity / 10.2, 1e-4);
}

TEST(ImuMount, AnUpsideDownMountingMissingFromTheTfIsLearned) {
  const gtsam::Rot3 tf = gtsam::Rot3::Yaw(0.3);
  const gtsam::Rot3 truth = tf * gtsam::Rot3::Roll(M_PI);
  const auto e = learn(truth, tf, 1.0, 1.0);
  EXPECT_NEAR(e.tf_error_deg, 180.0, 1.0);
  EXPECT_EQ(e.accel_sign, 1.0);
  // Up and the turning axis are what a ground robot shows: the IMU's up axis lands on the base's.
  EXPECT_NEAR((e.R_base_imu.matrix() * truth.matrix().transpose() * Vector3d::UnitZ()).z(), 1.0,
              1e-3);
}

namespace {

/// A ground robot driving on a level floor for 60 s, speeding up, slowing down and turning; the IMU
/// is mounted as `R_base_imu_true`; the odometry reports, every second, the base's rotation and its
/// mean acceleration (dv/dt + w x v, base frame).
ImuMount::Estimate drive(const gtsam::Rot3& R_base_imu_true, const gtsam::Rot3& R_base_imu_tf,
                         bool report_motion) {
  ImuMount mount(R_base_imu_tf);
  const double hz = 100.0;
  Timestamp last_odom = 0;
  Vector3d turned = Vector3d::Zero(), accel_sum = Vector3d::Zero();
  for (int i = 0; i <= 60 * hz; ++i) {
    const double t = i / hz;
    const Timestamp stamp = 1'000'000'000LL + static_cast<Timestamp>(t * 1e9);
    const double v = 0.8 + 0.5 * std::sin(0.5 * t), dv = 0.25 * std::cos(0.5 * t);
    const double wz = 0.6 * std::sin(0.4 * t);
    const Vector3d w_base(0, 0, wz);
    const Vector3d a_base(dv, v * wz, 0);  // dv/dt + w x v
    ImuSample s;
    s.stamp = stamp;
    s.gyro = R_base_imu_true.matrix().transpose() * w_base;
    s.accel = R_base_imu_true.matrix().transpose() * (a_base + Vector3d(0, 0, kStandardGravity));
    mount.add_imu(s);
    if (i > 0) {
      turned += w_base / hz;
      accel_sum += a_base;
    }
    if (i % 100 == 0) {
      if (last_odom) {
        mount.add_base_rotation(last_odom, stamp, turned, Vector3d::UnitZ());
        if (report_motion)
          mount.add_base_motion(last_odom, stamp, accel_sum / hz, Vector3d::UnitZ());
      }
      last_odom = stamp;
      turned.setZero();
      accel_sum.setZero();
    }
  }
  const auto e = mount.estimate();
  EXPECT_TRUE(e.has_value());
  return e.value_or(ImuMount::Estimate{});
}

}  // namespace

// The Robotnik Summit of the `prueba_03` bags: the TF gives the IMU the base's axes, the IMU reads
// x and y backwards. The turns about the vertical cannot show it; braking and the centripetal
// acceleration do, and with the TF's heading every slope read with the wrong sign (ADR-0058).
TEST(ImuMount, AnImuTurnedHalfAroundTheVerticalIsLearnedFromTheAccelerations) {
  const gtsam::Rot3 truth = gtsam::Rot3::Yaw(M_PI);
  const auto e = drive(truth, gtsam::Rot3(), true);
  EXPECT_TRUE(e.yaw_observed);
  EXPECT_NEAR(e.tf_error_deg, 180.0, 1.0);
  EXPECT_LT(gtsam::Rot3::Logmap(e.R_base_imu.between(truth)).norm() * 180.0 / M_PI, 1.0);
}

TEST(ImuMount, ACorrectHeadingIsConfirmedByTheAccelerations) {
  const gtsam::Rot3 truth = gtsam::Rot3::Yaw(-M_PI / 2);
  const auto e = drive(truth, truth, true);
  EXPECT_TRUE(e.yaw_observed);
  EXPECT_LT(e.tf_error_deg, 1.0);
}

TEST(ImuMount, WithoutTheAccelerationsTheHeadingStaysTheTfs) {
  const auto e = drive(gtsam::Rot3::Yaw(M_PI), gtsam::Rot3(), false);
  EXPECT_FALSE(e.yaw_observed);
  EXPECT_LT(e.tf_error_deg, 1.0);
}

TEST(ImuMount, NothingIsLearnedWithoutTurning) {
  ImuMount mount{gtsam::Rot3()};
  for (int i = 0; i < 1000; ++i) {
    ImuSample s;
    s.stamp = 1'000'000'000LL + i * 10'000'000LL;
    s.accel = Vector3d(0, 0, kStandardGravity);
    mount.add_imu(s);
    if (i > 0 && i % 10 == 0)
      mount.add_base_rotation(s.stamp - 100'000'000LL, s.stamp, Vector3d::Zero(),
                              Vector3d::UnitZ());
  }
  EXPECT_FALSE(mount.estimate().has_value());
}

TEST(GravityTracker, FollowsATiltAndIgnoresTheRobotsOwnAcceleration) {
  GravityTracker g;
  const double hz = 100.0;
  double pitch = 0, worst_deg = 0;
  const auto error_deg = [&] {
    const Vector3d expected = gtsam::Rot3::Pitch(pitch).matrix().transpose() * Vector3d::UnitZ();
    return std::acos(std::clamp(g.up()->dot(expected), -1.0, 1.0)) * 180.0 / M_PI;
  };
  for (int i = 0; i <= 10 * hz; ++i) {
    const double t = i / hz;
    const double rate =
        (t > 3 && t < 4) ? 0.1 : 0.0;  // pitches up by 0.1 rad (a ramp) between 3 and 4 s
    pitch += rate / hz;
    const gtsam::Rot3 R_wb = gtsam::Rot3::Pitch(pitch);
    Vector3d f = R_wb.matrix().transpose() * Vector3d(0, 0, kStandardGravity);
    if (t > 6 && t < 7)
      f += Vector3d(1.0, 0, 0);  // accelerating forward for a second (a robot starting): not a tilt
    g.add(1'000'000'000LL + static_cast<Timestamp>(t * 1e9), f, Vector3d(0, rate, 0));
    if (t > 2 && g.up()) worst_deg = std::max(worst_deg, error_deg());
  }
  ASSERT_TRUE(g.up().has_value());
  EXPECT_LT(error_deg(), 0.5);  // the ramp's tilt, carried by the gyroscope
  EXPECT_LT(worst_deg, 0.75);   // the acceleration (5.8 deg of apparent tilt) moved it little
}

namespace {

/// A hand-held sensor turning about the vertical while it tilts back and forth by up to 40 deg,
/// its IMU on its axes reporting gravity times `scale`; the odometry reports its rotation every
/// 0.1 s.
ImuMount::Estimate learn_tilting(double scale) {
  ImuMount mount(gtsam::Rot3{});
  const double hz = 200.0;
  const auto R_wb = [](double t) {
    return gtsam::Rot3::Yaw(0.5 * t) * gtsam::Rot3::Pitch(0.7 * std::sin(0.9 * t)) *
           gtsam::Rot3::Roll(0.4 * std::sin(1.3 * t));
  };
  Timestamp last_odom = 0;
  gtsam::Rot3 R_last_odom = R_wb(0);
  for (int i = 0; i <= 30 * hz; ++i) {
    const double t = i / hz;
    const Timestamp stamp = 1'000'000'000LL + static_cast<Timestamp>(t * 1e9);
    ImuSample s;
    s.stamp = stamp;
    s.gyro = gtsam::Rot3::Logmap(R_wb(t - 1 / hz).between(R_wb(t))) * hz;  // body rate
    s.accel = scale * (R_wb(t).matrix().transpose() * Vector3d(0, 0, kStandardGravity));
    mount.add_imu(s);
    if (i % 20 == 0) {
      const Vector3d up = R_wb(t).matrix().transpose() * Vector3d::UnitZ();
      if (last_odom)
        mount.add_base_rotation(last_odom, stamp, gtsam::Rot3::Logmap(R_last_odom.between(R_wb(t))),
                                up);
      last_odom = stamp;
      R_last_odom = R_wb(t);
    }
  }
  const auto e = mount.estimate();
  EXPECT_TRUE(e.has_value());
  return e.value_or(ImuMount::Estimate{});
}

}  // namespace

// The mean of the raw readings of a sensor that tilts is shorter than gravity (8.9 instead of
// 9.97 m/s^2 with a hand-held OAK-D Lite): read as a scale error, it was "corrected" by 1.099 and
// the IMU then saw a lift wherever it went. Measured in a frame the gyroscope holds still, it is
// gravity again.
TEST(ImuMount, ASensorThatTiltsKeepsItsScale) {
  EXPECT_NEAR(learn_tilting(1.0).accel_scale, 1.0, 1e-3);
  EXPECT_NEAR(learn_tilting(1.05).accel_scale, 1 / 1.05, 1e-3);
}

// ---------------- the whole system: LiDAR odometry and an IMU turned half around ----------------

namespace {

struct Messages final : Logger {
  std::vector<std::string> lines;
  void log(LogLevel, const std::string& m) override { lines.push_back(m); }
};

/// Random samples on a surface o + u a + v b (u, v in [0, 1]), `per_m2` of them per square metre;
/// `lift` raises each sample by a height that depends on its x (a ramp).
void face(std::vector<Vector3d>& pts, std::mt19937& rng, const Vector3d& o, const Vector3d& a,
          const Vector3d& b, double per_m2, const std::function<double(double)>& lift = {}) {
  std::uniform_real_distribution<double> u(0, 1);
  const int n = static_cast<int>(a.norm() * b.norm() * per_m2);
  for (int i = 0; i < n; ++i) {
    Vector3d p = o + u(rng) * a + u(rng) * b;
    if (lift) p.z() += lift(p.x());
    pts.push_back(p);
  }
}

/// The four sides and the top of a box standing on [lo, hi].
void box(std::vector<Vector3d>& pts, std::mt19937& rng, const Vector3d& lo, const Vector3d& hi,
         double per_m2) {
  const Vector3d d = hi - lo;
  face(pts, rng, lo, Vector3d(d.x(), 0, 0), Vector3d(0, 0, d.z()), per_m2);
  face(pts, rng, Vector3d(lo.x(), hi.y(), lo.z()), Vector3d(d.x(), 0, 0), Vector3d(0, 0, d.z()),
       per_m2);
  face(pts, rng, lo, Vector3d(0, d.y(), 0), Vector3d(0, 0, d.z()), per_m2);
  face(pts, rng, Vector3d(hi.x(), lo.y(), lo.z()), Vector3d(0, d.y(), 0), Vector3d(0, 0, d.z()),
       per_m2);
  face(pts, rng, Vector3d(lo.x(), lo.y(), hi.z()), Vector3d(d.x(), 0, 0), Vector3d(0, d.y(), 0),
       per_m2);
}

/// A ground robot with LiDAR (at the base) and an IMU mounted turned half around the vertical,
/// while the TF gives it the base's axes: the Summit of `prueba_03`.
struct SummitRig {
  static constexpr double kBaseHeight = 0.4;
  gtsam::Rot3 R_base_imu;  ///< the real mounting; the TF says the IMU is on the base's axes
  std::shared_ptr<Messages> messages = std::make_shared<Messages>();
  std::unique_ptr<SlamSystem> slam;
  /// Where the odometry started: at the first scan once the IMU has had half a second at rest (or
  /// after two seconds without), so that the start is levelled.
  std::optional<gtsam::Pose3> start;
  std::mt19937 rng{5};
  std::normal_distribution<double> noise{0, 0.005};

  explicit SummitRig(const gtsam::Rot3& mounting = gtsam::Rot3::Yaw(M_PI)) : R_base_imu(mounting) {
    Config cfg;
    cfg.robot.platform = Platform::Ground;
    cfg.sensors.lidar.enabled = true;
    cfg.sensors.imu.enabled = true;
    cfg.fusion.odometry_from = OdometryFrom::Lidar;
    slam = std::make_unique<SlamSystem>(cfg, Extrinsics(), SystemTuning(), messages);
  }
  /// The IMU's reading of the base turning at w_base with specific force f_base (base frame).
  void imu(Timestamp stamp, const Vector3d& w_base, const Vector3d& f_base) {
    ImuSample s;
    s.stamp = stamp;
    s.gyro = R_base_imu.matrix().transpose() * w_base;
    s.accel = R_base_imu.matrix().transpose() * f_base;
    slam->add_imu(s);
  }
  /// What the LiDAR sees of `world` with the base at `T_world_base`.
  void lidar(Timestamp stamp, const std::vector<Vector3d>& world,
             const gtsam::Pose3& T_world_base) {
    LidarScan scan;
    scan.stamp = stamp;
    scan.points.reserve(world.size());
    for (const auto& p : world)
      scan.points.push_back(
          (T_world_base.transformTo(p) + Vector3d(noise(rng), noise(rng), noise(rng)))
              .cast<float>());
    slam->add_lidar(scan);
    if (!start && slam->current_pose_odom()) start = T_world_base;
  }
  /// The IMU's heading relative to the TF's, as the system reported learning it from the
  /// accelerations (-1: never).
  double learned_heading_deg() const {
    double deg = -1;
    for (const auto& m : messages->lines) {
      const auto at = m.find(" deg from the TF");
      if (m.rfind("IMU: mounting learned", 0) == 0 &&
          m.find("accelerations") != std::string::npos && at != std::string::npos)
        deg = std::stod(m.substr(m.rfind(' ', at - 1) + 1));
    }
    return deg;
  }
  /// How far the odometry is from the truth (m, deg): its frame is the base's where it started (on
  /// a level floor in these tests, so that levelling it changes nothing).
  std::pair<double, double> odometry_error(const gtsam::Pose3& T_world_base) const {
    const auto pose = slam->current_pose_odom();
    if (!pose || !start) return {1e9, 1e9};
    const gtsam::Pose3 expected = start->between(T_world_base);
    return {(pose->translation() - expected.translation()).norm(),
            gtsam::Rot3::Logmap(pose->rotation().between(expected.rotation())).norm() * 180 / M_PI};
  }
  void expect_odometry_at(const gtsam::Pose3& T_world_base, double tol_m) const {
    const auto [m, deg] = odometry_error(T_world_base);
    EXPECT_LT(m, tol_m);
    EXPECT_LT(deg, 1.0);
  }
};

/// Tight loops, speeding up and slowing down, on a level floor (base frame: w = wz about the
/// vertical, a = dv/dt + w x v): about 1 m/s^2 of centripetal acceleration, so that the heading is
/// learned within a few seconds of them (cheap enough for the Debug builds of CI, where every scan
/// costs a second); they start at 1 m/s and stay within some 3 m of where they start.
struct Loops {
  double v(double t) const { return 1.0 + 0.4 * std::sin(0.5 * t); }
  double dv(double t) const { return 0.2 * std::cos(0.5 * t); }
  double wz(double t) const { return 0.9 + 0.3 * std::sin(0.4 * t); }
};

}  // namespace

// On a level floor: the system learns the heading from the accelerations and the odometry stays on
// the robot's path throughout.
TEST(ImuMount, TheSystemLearnsAnImuTurnedHalfAroundAndKeepsTheLidarOdometry) {
  SummitRig rig;
  std::vector<Vector3d> world;  // a 9 x 9 m room with some boxes in it
  std::mt19937 rng(3);
  box(world, rng, {-4, -3, 0}, {5, 6, 3}, 10);  // walls and ceiling
  face(world, rng, {-4, -3, 0}, {9, 0, 0}, {0, 9, 0}, 10);
  box(world, rng, {-3.5, 3, 0}, {-2.5, 4, 1.5}, 10);
  box(world, rng, {3.8, 0, 0}, {4.6, 1.5, 2.2}, 10);
  box(world, rng, {3, 4.5, 0}, {4, 5.5, 1.0}, 10);
  box(world, rng, {0.8, -2.6, 0}, {1.6, -2, 0.8}, 10);
  // 6 s of loops: the heading is learned after 3, the odometry runs on with it for 3 more.
  const Loops loops;
  double x = 0, y = 0, th = 0;
  const double dt = 0.001;
  gtsam::Pose3 truth;
  for (int i = 0; i <= 6'000; ++i) {
    const double t = i * dt;
    const Timestamp stamp = 1'000'000'000LL + static_cast<Timestamp>(i) * 1'000'000LL;
    const double v = loops.v(t), wz = loops.wz(t);
    if (i % 10 == 0)
      rig.imu(stamp, Vector3d(0, 0, wz), Vector3d(loops.dv(t), v * wz, kStandardGravity));
    truth = gtsam::Pose3(gtsam::Rot3::Yaw(th), gtsam::Point3(x, y, SummitRig::kBaseHeight));
    if (i % 100 == 0) rig.lidar(stamp, world, truth);
    x += v * std::cos(th) * dt;
    y += v * std::sin(th) * dt;
    th += wz * dt;
  }
  EXPECT_NEAR(rig.learned_heading_deg(), 180.0, 5.0);
  rig.expect_odometry_at(truth, 0.3);
}

// A ramp before the heading is known: the IMU's gravity, read through the TF, has the slope
// backwards, and bounding the LiDAR's attitude by it would turn the climb into a dive (some 19 deg
// and 1 m off before the level mezzanine pulls the odometry back). The system keeps the IMU out of
// the odometry until the accelerations have shown the heading (ADR-0058).
TEST(ImuMount, ARampDrivenBeforeTheHeadingIsLearnedKeepsItsHeight) {
  SummitRig rig;
  // A 14 x 8 m hall whose floor rises 0.59 m on a 5 m ramp (x from 1.5 to 6.5; 10 deg at its
  // steepest) to a mezzanine; pillars along the walls, boxes on both levels. The ramp starts after
  // the IMU's first second at rest on the level floor (checked against the TF over it, it levels
  // the start and its gravity settles): from then on it is there to bound the LiDAR, and the ramp
  // shows whether it may.
  constexpr double kRampStart = 1.5, kRampLength = 5.0;
  const double H = std::tan(10.0 * M_PI / 180) * kRampLength / 1.5;
  const auto u = [](double x) { return std::clamp((x - kRampStart) / kRampLength, 0.0, 1.0); };
  const auto h = [&](double x) {  // the floor's height; smooth, so the IMU sees no jolt
    return H * u(x) * u(x) * (3 - 2 * u(x));
  };
  const auto dh = [&](double x) { return H * 6 * u(x) * (1 - u(x)) / kRampLength; };
  const auto ddh = [&](double x) {
    return u(x) <= 0 || u(x) >= 1 ? 0.0 : H * (6 - 12 * u(x)) / (kRampLength * kRampLength);
  };
  std::vector<Vector3d> world;
  std::mt19937 rng(3);
  box(world, rng, {-2, -3, 0}, {12, 5, 3}, 10);  // walls and ceiling
  face(world, rng, {-2, -3, 0}, {14, 0, 0}, {0, 8, 0}, 10, h);
  for (double px = 0; px <= 11; px += 3.5) {
    box(world, rng, {px, -2.8, 0}, {px + 0.4, -2.4, 3}, 10);
    box(world, rng, {px, 4.4, 0}, {px + 0.4, 4.8, 3}, 10);
  }
  box(world, rng, {-1.5, -2, 0}, {-0.8, -1.3, 1.5}, 10);
  box(world, rng, {1, 3.5, 0}, {2, 4.2, 2}, 10);
  box(world, rng, {10.5, -2.2, H}, {11.5, -1.4, H + 1.5}, 10);
  box(world, rng, {10, 3.6, H}, {11.4, 4.3, H + 1}, 10);
  box(world, rng, {6.5, -2.6, H}, {7.3, -2.0, H + 0.8}, 10);

  // 8.5 s straight on at 1 m/s (up the ramp from 1.5 to 6.5 s), then 5 s of loops on the mezzanine:
  // the heading is learned after 3, the odometry runs on with it for 2 more.
  const Loops loops;
  double x = 0, y = 0, th = 0;
  const double dt = 0.001, t_loops = 8.5;
  double worst_m = 0, worst_deg = 0;  // of the odometry while driving straight on
  gtsam::Pose3 truth;
  for (int i = 0; i <= 13'500; ++i) {
    const double t = i * dt;
    const Timestamp stamp = 1'000'000'000LL + static_cast<Timestamp>(i) * 1'000'000LL;
    Vector3d w_base, f_base;
    if (t < t_loops) {
      x = t;                                   // 1 m/s along x, the height following the floor
      const double pitch = -std::atan(dh(x));  // nose up
      truth = gtsam::Pose3(gtsam::Rot3::Pitch(pitch),
                           gtsam::Point3(x, 0, h(x) + SummitRig::kBaseHeight));
      w_base = Vector3d(0, -ddh(x) / (1 + dh(x) * dh(x)), 0);
      f_base = truth.rotation().matrix().transpose() * Vector3d(0, 0, ddh(x) + kStandardGravity);
    } else {
      const double tl = t - t_loops, v = loops.v(tl), wz = loops.wz(tl);
      truth = gtsam::Pose3(gtsam::Rot3::Yaw(th), gtsam::Point3(x, y, H + SummitRig::kBaseHeight));
      w_base = Vector3d(0, 0, wz);
      f_base = Vector3d(loops.dv(tl), v * wz, kStandardGravity);
      x += v * std::cos(th) * dt;
      y += v * std::sin(th) * dt;
      th += wz * dt;
    }
    if (i % 10 == 0) rig.imu(stamp, w_base, f_base);
    if (i % 100 == 0) {
      rig.lidar(stamp, world, truth);
      if (t < t_loops && rig.start) {
        const auto [m, deg] = rig.odometry_error(truth);
        worst_m = std::max(worst_m, m);
        worst_deg = std::max(worst_deg, deg);
      }
    }
    if (i == 1'400) {  // the IMU at rest levelled the start, before the ramp
      EXPECT_TRUE(rig.start.has_value());
    }
    if (i == 8'400) {  // the ramp was driven before the heading was known
      EXPECT_LT(rig.learned_heading_deg(), 0.0);
    }
  }
  // Up the ramp the odometry followed the robot, at every scan.
  EXPECT_LT(worst_m, 0.15);
  EXPECT_LT(worst_deg, 1.0);
  EXPECT_NEAR(rig.learned_heading_deg(), 180.0, 5.0);
  rig.expect_odometry_at(truth, 0.3);
}

// ---------------- the start: levelled by the IMU at rest ----------------

namespace {

/// A hand-held LiDAR with an IMU on its axes, in a room whose floor it does not see (walls and
/// pillars only): only the IMU can level the start. Scans and IMU start together, as a camera with
/// its own IMU does.
struct HandheldRig {
  std::shared_ptr<Messages> messages = std::make_shared<Messages>();
  std::unique_ptr<SlamSystem> slam;
  std::vector<Vector3d> world;
  HandheldRig() {
    Config cfg;
    cfg.robot.platform = Platform::Handheld;
    cfg.sensors.lidar.enabled = true;
    cfg.sensors.imu.enabled = true;
    cfg.fusion.odometry_from = OdometryFrom::Lidar;
    slam = std::make_unique<SlamSystem>(cfg, Extrinsics(), SystemTuning(), messages);
    std::mt19937 rng(4);
    face(world, rng, {-4, -3, -1}, {9, 0, 0}, {0, 0, 4}, 10);
    face(world, rng, {-4, 5, -1}, {9, 0, 0}, {0, 0, 4}, 10);
    face(world, rng, {-4, -3, -1}, {0, 8, 0}, {0, 0, 4}, 10);
    face(world, rng, {5, -3, -1}, {0, 8, 0}, {0, 0, 4}, 10);
    for (const auto& [x, y] : {std::pair{-2.0, 2.0}, {3.0, -1.5}, {1.0, 3.5}}) {
      face(world, rng, {x, y, -1}, {0.5, 0, 0}, {0, 0, 4}, 10);
      face(world, rng, {x, y, -1}, {0, 0.5, 0}, {0, 0, 4}, 10);
    }
  }
  /// `seconds` of IMU (200 Hz) and scans (10 Hz) with the sensor at `T_world_base(t)`, turning at
  /// `w_base` (base frame) and not accelerating.
  void run(double seconds, const std::function<gtsam::Pose3(double)>& T_world_base,
           const Vector3d& w_base) {
    for (int i = 0; i <= static_cast<int>(seconds * 1000); ++i) {
      const double t = i * 1e-3;
      const Timestamp stamp = 1'000'000'000LL + static_cast<Timestamp>(i) * 1'000'000LL;
      const gtsam::Pose3 T = T_world_base(t);
      if (i % 5 == 0) {
        ImuSample s;
        s.stamp = stamp;
        s.gyro = w_base;
        s.accel = T.rotation().matrix().transpose() * Vector3d(0, 0, kStandardGravity);
        slam->add_imu(s);
      }
      if (i % 100 == 0) {
        LidarScan scan;
        scan.stamp = stamp;
        for (const auto& p : world) scan.points.push_back(T.transformTo(p).cast<float>());
        slam->add_lidar(scan);
        if (!started_at && slam->current_pose_odom()) {
          started_at = t;
          start_tilt_error_deg = tilt_error_deg(T);
        }
      }
    }
  }
  bool said(const std::string& text) const {
    return std::any_of(messages->lines.begin(), messages->lines.end(),
                       [&](const std::string& m) { return m.find(text) != std::string::npos; });
  }
  /// Angle between up by the odometry and up by the truth (deg).
  double tilt_error_deg(const gtsam::Pose3& T_world_base) const {
    const auto pose = slam->current_pose_odom();
    const Vector3d up_odom = pose->rotation().matrix().transpose() * Vector3d::UnitZ();
    const Vector3d up_true = T_world_base.rotation().matrix().transpose() * Vector3d::UnitZ();
    return std::acos(std::clamp(up_odom.dot(up_true), -1.0, 1.0)) * 180 / M_PI;
  }
  std::optional<double> started_at;  ///< s, when the odometry produced its first pose
  double start_tilt_error_deg = 0;   ///< at that first pose
};

}  // namespace

// Scans arrive before the IMU has half a second (and before it is checked against the TF, over its
// first second): the start waits for it, then the odometry starts with the sensor's real tilt.
// Without a ground plane in view it started at the sensor's attitude, 25 deg off (a hand-held
// OAK-D Lite facing a wall).
TEST(ImuMount, AStartAtRestIsLevelledByTheImuEvenIfTheScansCameFirst) {
  HandheldRig rig;
  const gtsam::Pose3 T(gtsam::Rot3::RzRyRx(15 * M_PI / 180, -20 * M_PI / 180, 0),
                       gtsam::Point3(0, 0, 1.2));
  rig.run(2.0, [&](double) { return T; }, Vector3d::Zero());
  EXPECT_TRUE(rig.said("gravity alignment applied"));
  ASSERT_TRUE(rig.started_at.has_value());
  EXPECT_LT(*rig.started_at, 1.5);
  // From its first pose: the IMU's gravity would bound the tilt to 1 deg later on (once tracked),
  // but until then the floor manager reads the vertical through it.
  EXPECT_LT(rig.start_tilt_error_deg, 0.5);
  EXPECT_LT(rig.tilt_error_deg(T), 0.5);
}

// A start that never rests: the odometry waits two seconds, then starts at the sensor's attitude
// and says so.
TEST(ImuMount, AStartThatDoesNotRestGoesOnUnlevelledAfterTwoSeconds) {
  HandheldRig rig;
  const double w = 0.4;  // rad/s about the vertical
  rig.run(
      3.0,
      [&](double t) { return gtsam::Pose3(gtsam::Rot3::Yaw(w * t), gtsam::Point3(0, 0, 1.2)); },
      Vector3d(0, 0, w));
  ASSERT_TRUE(rig.started_at.has_value());
  EXPECT_NEAR(*rig.started_at, 2.0, 0.15);
  EXPECT_TRUE(rig.said("no gravity alignment"));
  EXPECT_FALSE(rig.said("gravity alignment applied"));
}

// A start that could not be levelled leaves the odometry frame tilted (here 20 deg): read through
// it, gravity has a vertical component of -0.6 m/s^2, a lift that never was, which opened a floor
// transition three seconds into the run of a hand-held camera. The floor manager reads the vertical
// through the IMU's own gravity instead, and not before it can be trusted.
TEST(ImuMount, ATiltedOdometryFrameDoesNotMakeTheImuSeeALift) {
  HandheldRig rig;
  const double w = 0.4, pitch = 20 * M_PI / 180;
  rig.run(
      10.0,
      [&](double t) {
        return gtsam::Pose3(gtsam::Rot3::Yaw(w * t) * gtsam::Rot3::Pitch(pitch),
                            gtsam::Point3(0, 0, 1.2));
      },
      gtsam::Rot3::Pitch(pitch).matrix().transpose() * Vector3d(0, 0, w));
  ASSERT_TRUE(rig.started_at.has_value());
  EXPECT_GT(rig.start_tilt_error_deg, 10.0);  // the start could not be levelled
  EXPECT_FALSE(rig.said("floor transition started"));
}

// A ground robot at rest on a level floor, its IMU turned half around and pitched 1 deg on its
// mount while the TF has it on the base's axes: read through the TF, its gravity tilts the start
// by 1 deg (the Summit's 0.46 deg read backwards gave `prueba_03` a fourth floor). Until the
// accelerations have shown the heading the IMU does not level a ground robot (ADR-0058): the
// ground plane does.
TEST(ImuMount, AGroundRobotIsLevelledByItsFloorWhileItsImuHeadingIsUnknown) {
  SummitRig rig(gtsam::Rot3::Yaw(M_PI) * gtsam::Rot3::Pitch(M_PI / 180));
  std::vector<Vector3d> world;
  std::mt19937 rng(3);
  box(world, rng, {-4, -3, 0}, {5, 6, 3}, 10);
  face(world, rng, {-4, -3, 0}, {9, 0, 0}, {0, 9, 0}, 10);
  box(world, rng, {3.8, 0, 0}, {4.6, 1.5, 2.2}, 10);
  const gtsam::Pose3 truth(gtsam::Rot3(), gtsam::Point3(0, 0, SummitRig::kBaseHeight));
  for (int i = 0; i <= 3'000; ++i) {
    const Timestamp stamp = 1'000'000'000LL + static_cast<Timestamp>(i) * 1'000'000LL;
    if (i % 10 == 0) rig.imu(stamp, Vector3d::Zero(), Vector3d(0, 0, kStandardGravity));
    if (i % 100 == 0) rig.lidar(stamp, world, truth);
  }
  ASSERT_TRUE(rig.start.has_value());
  EXPECT_LT(rig.odometry_error(truth).second, 0.3);
}
