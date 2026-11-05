#pragma once

#include <deque>
#include <optional>

#include <Eigen/Core>
#include <gtsam/geometry/Rot3.h>

#include "strataslam/common/types.hpp"
#include "strataslam/sensors/measurements.hpp"

namespace strataslam {

/// How an IMU is really mounted, learned from its own data, so that it is used the same whatever the TF and the
/// driver say. The gyroscope is compared with the rotation that the odometry measures while the robot turns (Wahba's
/// problem on the paired rotations): that gives the rotation IMU -> base, the TF filling in only what the motion does
/// not show (a ground robot only turns about the vertical). With that rotation, gravity gives the sign convention of
/// the accelerometer (the specific force points up at rest in ROS; some drivers report it pointing down) and its
/// scale.
class ImuMount {
 public:
  struct Estimate {
    gtsam::Rot3 R_base_imu;   ///< IMU frame -> base frame
    double accel_sign = 1.0;  ///< -1: the driver reports gravity pointing down
    double accel_scale = 1.0; ///< standard gravity over the measured one (1 if far off: not a scale error)
    double tf_error_deg = 0;  ///< angle between this rotation and the TF's
    double turned_deg = 0;    ///< rotation of the base that it was learned from
  };

  /// `R_base_imu_tf`: the mounting the TF gives (the prior for what the motion does not show).
  explicit ImuMount(const gtsam::Rot3& R_base_imu_tf, double min_turn_deg = 180.0);

  void add_imu(const ImuSample& s);
  /// The base turned by `rotvec_base` (rotation vector, base frame) between `t0` and `t1`, by the odometry;
  /// `up_base` is up in the base frame then (for the accelerometer's sign). It waits for the IMU to cover the
  /// interval (an IMU may trail the odometry).
  void add_base_rotation(Timestamp t0, Timestamp t1, const Eigen::Vector3d& rotvec_base, const Eigen::Vector3d& up_base);
  /// Once the base has turned `min_turn_deg` in all (and the gyroscope followed it).
  std::optional<Estimate> estimate() const;
  /// Mean specific force so far, IMU frame (gravity, once the robot's own accelerations average out).
  Eigen::Vector3d mean_accel() const { return n_accel_ ? Eigen::Vector3d(accel_sum_ / n_accel_) : Eigen::Vector3d::Zero(); }
  size_t num_samples() const { return n_accel_; }
  /// The last seconds of IMU, as received.
  const std::deque<ImuSample>& recent() const { return recent_; }

 private:
  gtsam::Rot3 R_tf_;
  double min_turn_rad_;
  struct Turn {
    Timestamp t0, t1;
    Eigen::Vector3d rotvec_base, up_base;
  };
  void pair(const Turn& t);
  void pair_ready();
  std::deque<ImuSample> recent_;  // for the gyro integral over each odometry interval
  std::deque<Turn> waiting_;      // odometry turns the IMU does not cover yet
  Eigen::Vector3d accel_sum_ = Eigen::Vector3d::Zero();
  size_t n_accel_ = 0;
  Eigen::Matrix3d B_ = Eigen::Matrix3d::Zero();  // sum of rotvec_base * rotvec_imu^T
  double energy_ = 0;                            // sum of |rotvec_base|^2
  double turned_ = 0;                            // sum of |rotvec_base|
  Eigen::Vector3d up_sum_ = Eigen::Vector3d::Zero();
};

/// Up in the base frame from a calibrated IMU (a Mahony filter on the gravity direction): the gyroscope carries it and
/// the accelerometer only corrects its drift, with a long time constant `tau_s`, so that the robot's own accelerations
/// (which barely change the magnitude of the specific force when they are horizontal) do not tilt it; samples far
/// from gravity are ignored. The integral term estimates the gyroscope's bias, which a long time constant would
/// otherwise turn into a fixed tilt (bias times tau). During the first `settle_s` the time constant is short; if the
/// body was at rest then, the bias is the gyroscope's mean and `up` is ready at once, else only once the integral
/// term has had three time constants to converge.
class GravityTracker {
 public:
  explicit GravityTracker(double tau_s = 10.0, double settle_s = 1.0) : tau_s_(tau_s), settle_s_(settle_s) {}
  /// A sample already in the base frame, with its accelerometer corrected (sign and scale).
  void add(Timestamp stamp, const Eigen::Vector3d& accel_base, const Eigen::Vector3d& gyro_base);
  /// Up in the base frame, once it can be trusted (see the class).
  std::optional<Eigen::Vector3d> up() const;
  void reset() {
    up_.reset();
    bias_.setZero();
    first_ = last_ = 0;
    gyro_sum_.setZero();
    gyro_n_ = 0;
    static_start_ = still_ = true;
  }
  const Eigen::Vector3d& gyro_bias() const { return bias_; }

 private:
  double tau_s_, settle_s_;
  std::optional<Eigen::Vector3d> up_;
  Eigen::Vector3d bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_sum_ = Eigen::Vector3d::Zero();  // during the settling: the bias if the body was at rest
  size_t gyro_n_ = 0;
  bool still_ = true;         // at rest so far during the settling
  bool static_start_ = true;  // it was, once settled
  Timestamp first_ = 0, last_ = 0;
};

}  // namespace strataslam
