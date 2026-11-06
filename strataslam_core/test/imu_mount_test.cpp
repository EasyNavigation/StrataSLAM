#include "strataslam/sensors/imu_mount.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <string>
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
  EXPECT_NEAR(e.accel_scale, 1.0, 1e-6);
}

TEST(ImuMount, ADriverThatReportsGravityPointingDownIsDetected) {
  const gtsam::Rot3 R =
      gtsam::Rot3::Yaw(-M_PI / 2);  // the PAL Tiago's base IMU: -10.2 m/s^2 on z at rest
  const auto e = learn(R, R, -1.0, 10.2 / kStandardGravity);
  EXPECT_LT(e.tf_error_deg, 1.0);
  EXPECT_EQ(e.accel_sign, -1.0);
  EXPECT_NEAR(e.accel_scale, kStandardGravity / 10.2, 1e-6);
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
  const gtsam::Rot3 R_base_imu = gtsam::Rot3::Yaw(M_PI);
  std::shared_ptr<Messages> messages = std::make_shared<Messages>();
  std::unique_ptr<SlamSystem> slam;
  std::mt19937 rng{5};
  std::normal_distribution<double> noise{0, 0.005};

  SummitRig() {
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
  /// The odometry against the truth: it starts at the first scan, at the world's origin with no
  /// heading, so the odometry frame is the world's, the base's height down.
  void expect_odometry_at(const gtsam::Pose3& T_world_base, double tol_m) const {
    const auto pose = slam->current_pose_odom();
    ASSERT_TRUE(pose.has_value());
    const Vector3d expected = T_world_base.translation() - Vector3d(0, 0, kBaseHeight);
    EXPECT_LT((pose->translation() - expected).norm(), tol_m)
        << "odometry at " << pose->translation().transpose() << ", robot at "
        << expected.transpose();
    EXPECT_LT(
        gtsam::Rot3::Logmap(pose->rotation().between(T_world_base.rotation())).norm() * 180 / M_PI,
        1.0);
  }
};

/// Loops, speeding up and slowing down, on a level floor (base frame: w = wz about the vertical,
/// a = dv/dt + w x v); `v0` is the speed it starts them at.
struct Loops {
  double v0;
  double v(double t) const { return v0 + 0.4 * std::sin(0.5 * t); }
  double dv(double t) const { return 0.2 * std::cos(0.5 * t); }
  double wz(double t) const { return 0.5 + 0.3 * std::sin(0.4 * t); }
};

}  // namespace

// On a level floor: the system learns the heading from the accelerations and the odometry stays on
// the robot's path throughout.
TEST(ImuMount, TheSystemLearnsAnImuTurnedHalfAroundAndKeepsTheLidarOdometry) {
  SummitRig rig;
  std::vector<Vector3d> world;  // a 10 x 16 m hall with some boxes in it
  std::mt19937 rng(3);
  box(world, rng, {-4, -3, 0}, {6, 13, 3}, 20);  // walls and ceiling
  face(world, rng, {-4, -3, 0}, {10, 0, 0}, {0, 16, 0}, 20);
  box(world, rng, {-3.5, 4, 0}, {-2.5, 5, 1.5}, 20);
  box(world, rng, {4.5, 1, 0}, {5.5, 2.5, 2.2}, 20);
  box(world, rng, {4, 9, 0}, {5, 10, 1.0}, 20);
  box(world, rng, {-3.5, 11, 0}, {-2, 12, 2.5}, 20);
  box(world, rng, {0.8, -2.5, 0}, {1.6, -1.8, 0.8}, 20);
  // 30 s of loops: 870 deg of turning.
  const Loops loops{0.8};
  double x = 0, y = 0, th = 0;
  const double dt = 0.001;
  gtsam::Pose3 truth;
  for (int i = 0; i <= 30'000; ++i) {
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
// backwards, and bounding the LiDAR's attitude by it would turn the climb into a dive. The system
// keeps the IMU out of the odometry until the accelerations have shown the heading (ADR-0058).
TEST(ImuMount, ARampDrivenBeforeTheHeadingIsLearnedKeepsItsHeight) {
  SummitRig rig;
  // A 30 x 16 m hall whose floor rises 1.18 m on a 10 m ramp (x from 2 to 12; 10 deg at its
  // steepest) to a mezzanine; pillars along the walls, boxes on both levels.
  const double H = std::tan(10.0 * M_PI / 180) * 10.0 / 1.5;
  const auto h = [H](double x) {  // the floor's height; smooth, so the IMU sees no jolt
    const double u = std::clamp((x - 2.0) / 10.0, 0.0, 1.0);
    return H * u * u * (3 - 2 * u);
  };
  const auto dh = [H](double x) {
    const double u = std::clamp((x - 2.0) / 10.0, 0.0, 1.0);
    return H * 6 * u * (1 - u) / 10.0;
  };
  const auto ddh = [H](double x) {
    return x < 2 || x > 12 ? 0.0 : H * (6 - 12 * (x - 2) / 10.0) / 100.0;
  };
  std::vector<Vector3d> world;
  std::mt19937 rng(3);
  box(world, rng, {-4, -6, 0}, {26, 10, 5}, 10);  // walls and ceiling
  face(world, rng, {-4, -6, 0}, {30, 0, 0}, {0, 16, 0}, 15, h);
  for (double px = 0; px <= 24; px += 6) {
    box(world, rng, {px, -5.8, 0}, {px + 0.4, -5.4, 5}, 15);
    box(world, rng, {px, 9.4, 0}, {px + 0.4, 9.8, 5}, 15);
  }
  box(world, rng, {-3, -5, 0}, {-2, -4, 1.5}, 15);
  box(world, rng, {0, 6, 0}, {1, 7, 2}, 15);
  box(world, rng, {22, -5, H}, {23, -4, H + 1.5}, 15);
  box(world, rng, {18, 8, H}, {19.5, 9, H + 1}, 15);
  box(world, rng, {24, 3, H}, {25, 4.5, H + 2}, 15);
  box(world, rng, {13, -5, H}, {14, -4, H + 0.8}, 15);

  // 16 s straight on at 1 m/s (up the ramp from 2 to 12 s), then 25 s of loops on the mezzanine,
  // where the heading is learned.
  const Loops loops{1.0};
  double x = 0, y = 0, th = 0;
  const double dt = 0.001, t_loops = 16.0;
  gtsam::Pose3 truth;
  for (int i = 0; i <= 41'000; ++i) {
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
    if (i % 100 == 0) rig.lidar(stamp, world, truth);
    if (i == 14'000) {
      EXPECT_LT(rig.learned_heading_deg(),
                0.0);                      // the ramp was driven before the heading was known
      rig.expect_odometry_at(truth, 0.1);  // and the odometry is on the mezzanine
    }
  }
  EXPECT_NEAR(rig.learned_heading_deg(), 180.0, 5.0);
  rig.expect_odometry_at(truth, 0.3);
}
