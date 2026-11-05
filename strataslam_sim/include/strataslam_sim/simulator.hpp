#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "strataslam/sensors/measurements.hpp"
#include "strataslam_sim/building.hpp"
#include "strataslam_sim/trajectory.hpp"

namespace strataslam_sim {

struct SimConfig {
  bool camera = true, lidar = false, imu = true, baro = false, external_odom = false;
  double camera_hz = 10, lidar_hz = 10, imu_hz = 200, baro_hz = 20, odom_hz = 20;

  // Camera (feature level) + depth. The image is 320x240 by default; 640x480 is supported.
  int image_width = 320, image_height = 240;
  double hfov_deg = 70.0;
  int max_keypoints = 500;
  double pixel_sigma = 0.5;       ///< px
  double depth_coeff = 0.0015;    ///< depth noise sigma = coeff * z^2 (m)
  double depth_max = 10.0;        ///< m, sensor limit
  int desc_flip_bits = 6;         ///< descriptor bits flipped per observation
  double outlier_ratio = 0.05;    ///< spurious keypoints relative to real ones
  /// Time windows (s) in which the camera delivers empty frames (no keypoints, no depth).
  std::vector<std::pair<double, double>> camera_blackouts;

  // LiDAR.
  int lidar_beams = 16;           ///< 16 / 32 / 64
  int lidar_columns = 360;
  double lidar_fov_deg = 30.0;
  double lidar_range_sigma = 0.02;
  double lidar_min_range = 0.5, lidar_max_range = 60.0;

  // IMU (continuous-time densities, SI) and bias walk.
  double accel_noise_density = 2.0e-3, gyro_noise_density = 1.7e-4;
  double accel_random_walk = 3.0e-3, gyro_random_walk = 2.0e-5;
  double accel_bias_init = 0.02, gyro_bias_init = 0.002;
  double gravity = 9.80665;

  // Barometer.
  double baro_sigma_pa = 1.0;
  double baro_drift_pa = 0.05;    ///< random-walk, Pa per sqrt(s)
  double temperature_k = 288.15;
  double base_pressure_pa = 101325.0;

  // External (wheel) odometry: relative noise.
  double odom_scale_sigma = 0.01, odom_trans_sigma = 5e-4, odom_rot_sigma = 5e-4;

  strataslam::Extrinsics extrinsics;
  uint64_t seed = 42;
  int64_t start_stamp_ns = 1'000'000'000'000LL;

  SimConfig() {
    extrinsics.T_base_camera = strataslam::Extrinsics::ForwardCamera();
  }
};

struct SensorEvent {
  enum class Type { Imu, Baro, Odom, Rgbd, Lidar };
  int64_t stamp;
  Type type;
  size_t index;
};

/// Generates sensor data with exact ground truth from a building and a trajectory. Frames are
/// produced lazily by index; all noise is a pure function of (seed, sensor, index).
class Simulator {
 public:
  Simulator(std::shared_ptr<const Building> building, std::shared_ptr<const Trajectory> trajectory,
            const SimConfig& config);

  const SimConfig& config() const { return cfg_; }
  const Building& building() const { return *building_; }
  const Trajectory& trajectory() const { return *trajectory_; }
  strataslam::CameraIntrinsics intrinsics() const;

  /// All events sorted by time (ties: IMU, baro, odometry, then camera, then LiDAR).
  const std::vector<SensorEvent>& events() const { return events_; }

  strataslam::ImuSample imu(size_t i) const;
  strataslam::BaroSample baro(size_t i) const;
  strataslam::OdomMeasurement odom(size_t i) const;
  strataslam::RgbdFrame rgbd(size_t i) const;
  strataslam::LidarScan lidar(size_t i) const;

  // Ground truth.
  int64_t stamp_at(double t) const { return cfg_.start_stamp_ns + static_cast<int64_t>(std::llround(t * 1e9)); }
  double time_of(int64_t stamp) const { return (stamp - cfg_.start_stamp_ns) * 1e-9; }
  gtsam::Pose3 gt_pose(int64_t stamp) const { return trajectory_->pose_at(time_of(stamp)); }
  int true_floor(int64_t stamp) const { return trajectory_->state_at(time_of(stamp)).true_floor; }
  /// Barometric pressure for a given altitude (m).
  double pressure_at(double z) const;

  size_t count(SensorEvent::Type type) const { return counts_[static_cast<int>(type)]; }

 private:
  size_t n_imu() const { return counts_[0]; }
  std::shared_ptr<const Building> building_;
  std::shared_ptr<const Trajectory> trajectory_;
  SimConfig cfg_;
  std::vector<SensorEvent> events_;
  size_t counts_[5] = {0, 0, 0, 0, 0};
  std::vector<Eigen::Vector3d> accel_bias_, gyro_bias_;
  std::vector<double> baro_drift_;
  std::vector<gtsam::Pose3> odom_poses_;
};

}  // namespace strataslam_sim
