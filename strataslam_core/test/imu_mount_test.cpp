#include "strataslam/sensors/imu_mount.hpp"

#include <gtest/gtest.h>

#include <cmath>

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
