#include "strataslam/sensors/imu_mount.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/SVD>

#include "strataslam/sensors/barometer.hpp"  // kStandardGravity

namespace strataslam {

namespace {
constexpr Timestamp kKeep = 10'000'000'000;  // ns of IMU kept for the gyro integrals
constexpr double kAccelTolerance = 0.1;      // relative: specific force this close to gravity is "at rest"
}  // namespace

ImuMount::ImuMount(const gtsam::Rot3& R_base_imu_tf, double min_turn_deg)
    : R_tf_(R_base_imu_tf), min_turn_rad_(min_turn_deg * M_PI / 180.0) {}

void ImuMount::add_imu(const ImuSample& s) {
  if (!recent_.empty() && s.stamp <= recent_.back().stamp) return;
  recent_.push_back(s);
  while (recent_.front().stamp < s.stamp - kKeep) recent_.pop_front();
  accel_sum_ += s.accel;
  ++n_accel_;
  pair_ready();
}

void ImuMount::add_base_rotation(Timestamp t0, Timestamp t1, const Eigen::Vector3d& rotvec_base, const Eigen::Vector3d& up_base) {
  if (t1 <= t0) return;
  waiting_.push_back({t0, t1, rotvec_base, up_base});
  pair_ready();
}

void ImuMount::pair_ready() {
  while (!waiting_.empty() && !recent_.empty() && recent_.back().stamp >= waiting_.front().t1) {
    if (recent_.front().stamp <= waiting_.front().t0) pair(waiting_.front());  // else the IMU started later: dropped
    waiting_.pop_front();
  }
  while (waiting_.size() > 1000) waiting_.pop_front();  // an IMU that stopped
}

void ImuMount::pair(const Turn& turn) {
  // Gyro integral over [t0, t1], each sample held until the next.
  Eigen::Vector3d rotvec_imu = Eigen::Vector3d::Zero();
  for (size_t i = 0; i + 1 < recent_.size(); ++i) {
    const Timestamp a = std::max(recent_[i].stamp, turn.t0), b = std::min(recent_[i + 1].stamp, turn.t1);
    if (b > a) rotvec_imu += recent_[i].gyro * ((b - a) * 1e-9);
  }
  B_ += turn.rotvec_base * rotvec_imu.transpose();
  energy_ += turn.rotvec_base.squaredNorm();
  turned_ += turn.rotvec_base.norm();
  up_sum_ += turn.up_base;
}

std::optional<ImuMount::Estimate> ImuMount::estimate() const {
  if (turned_ < min_turn_rad_ || n_accel_ == 0) return std::nullopt;
  // Wahba: the rotation that best maps the gyro's rotations onto the odometry's, with the TF as a weak prior for
  // the axes the motion leaves unobserved.
  const Eigen::Matrix3d B = B_ + 0.02 * energy_ * R_tf_.matrix();
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Matrix3d& U = svd.matrixU();  // references: no copies (clang-tidy performance-unnecessary-copy-initialization)
  const Eigen::Matrix3d& V = svd.matrixV();
  const double d = (U * V.transpose()).determinant() < 0 ? -1.0 : 1.0;
  const Eigen::Matrix3d R = U * Eigen::Vector3d(1, 1, d).asDiagonal() * V.transpose();
  // The gyroscope must follow the odometry: most of the base's rotation explained (else the data are not usable).
  if ((R.transpose() * B_).trace() < 0.75 * energy_) return std::nullopt;
  Estimate e;
  e.R_base_imu = gtsam::Rot3(R);
  const Eigen::Vector3d g = mean_accel();
  e.accel_sign = (R * g).dot(up_sum_) >= 0 ? 1.0 : -1.0;
  const double ratio = g.norm() / kStandardGravity;
  e.accel_scale = std::abs(ratio - 1.0) < 0.15 ? 1.0 / ratio : 1.0;
  e.tf_error_deg = gtsam::Rot3::Logmap(R_tf_.between(e.R_base_imu)).norm() * 180.0 / M_PI;
  e.turned_deg = turned_ * 180.0 / M_PI;
  return e;
}

void GravityTracker::add(Timestamp stamp, const Eigen::Vector3d& accel_base, const Eigen::Vector3d& gyro_base) {
  const double n = accel_base.norm();
  const bool at_rest = std::abs(n - kStandardGravity) < kAccelTolerance * kStandardGravity;
  if (!up_) {
    if (!at_rest) return;
    up_ = accel_base / n;
    first_ = last_ = stamp;
    return;
  }
  const double dt = std::clamp((stamp - last_) * 1e-9, 0.0, 0.5);
  last_ = stamp;
  // Mahony: the body rate corrected by the bias and pulled so that `up` turns toward the measured gravity.
  const bool settling = (stamp - first_) * 1e-9 < settle_s_;
  if (settling) {
    gyro_sum_ += gyro_base;
    ++gyro_n_;
    still_ = still_ && at_rest && gyro_base.norm() < 0.05;
  } else if (gyro_n_) {  // settled: at rest all along, the mean rate is the bias
    static_start_ = still_ && gyro_n_ >= 10;
    if (static_start_) bias_ = gyro_sum_ / static_cast<double>(gyro_n_);
    gyro_n_ = 0;
  }
  const double kp = 1.0 / (settling ? 0.2 * settle_s_ : tau_s_);
  const double ki = settling ? 0.0 : 0.25 * kp * kp;  // critically damped
  Eigen::Vector3d w = gyro_base - bias_;
  if (at_rest) {
    const Eigen::Vector3d e = (accel_base / n).cross(*up_);
    w += kp * e;
    bias_ -= ki * e * dt;
  }
  up_ = (*up_ - w.cross(*up_) * dt).normalized();  // a world-fixed direction seen from the turning body
}

std::optional<Eigen::Vector3d> GravityTracker::up() const {
  const double since = (last_ - first_) * 1e-9;
  if (!up_ || since < settle_s_ || gyro_n_ || (!static_start_ && since < settle_s_ + 3.0 * tau_s_)) return std::nullopt;
  return up_;
}

}  // namespace strataslam
