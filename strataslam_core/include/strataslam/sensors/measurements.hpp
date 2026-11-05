#pragma once

#include <gtsam/geometry/Pose3.h>

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <vector>

#include "strataslam/common/types.hpp"

namespace strataslam {

/// IMU sample in the IMU frame. `accel` is specific force (reads +g upwards at rest).
struct ImuSample {
  Timestamp stamp = 0;
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();  // m/s^2
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();   // rad/s
};

struct BaroSample {
  Timestamp stamp = 0;
  double pressure_pa = 0.0;
};

/// Externally supplied odometry (wheel or VIO): pose of base in the odom frame.
struct OdomMeasurement {
  Timestamp stamp = 0;
  gtsam::Pose3 pose;
};

/// Pinhole intrinsics, optical frame (x right, y down, z forward).
struct CameraIntrinsics {
  double fx = 0, fy = 0, cx = 0, cy = 0;
  int width = 0, height = 0;
};

/// Depth image in meters (z-depth); 0 = invalid.
struct DepthImage {
  int width = 0, height = 0;
  std::vector<float> meters;
  float at(int u, int v) const { return meters[static_cast<size_t>(v) * width + u]; }
};

constexpr size_t kDescriptorBytes = 32;  // 256-bit binary descriptor (ORB-like)
using Descriptor = std::array<uint8_t, kDescriptorBytes>;

struct RgbKeypoint {
  float u = 0, v = 0;
  float response = 0;
  Descriptor descriptor{};
};

/// Color image (8-bit RGB, row-major), registered with the depth image.
struct RgbImage {
  int width = 0, height = 0;
  std::vector<uint8_t> data;  ///< width * height * 3
  bool empty() const { return data.empty(); }
};

/// Feature-level RGB-D frame: keypoints with descriptors plus the registered depth image.
struct RgbdFrame {
  Timestamp stamp = 0;
  CameraIntrinsics K;
  std::vector<RgbKeypoint> keypoints;
  DepthImage depth;
  RgbImage rgb;  ///< optional: only needed to colorize the LiDAR cloud
};

/// 3D LiDAR scan; points are in the sensor frame at their own measurement time.
struct LidarScan {
  Timestamp stamp = 0;  ///< start of the sweep
  std::vector<Eigen::Vector3f> points;
  std::vector<float> time_offsets_s;  ///< per point, from `stamp`; empty if unknown
};

/// Fixed sensor extrinsics (T_base_sensor).
struct Extrinsics {
  gtsam::Pose3 T_base_camera;
  gtsam::Pose3 T_base_lidar;
  gtsam::Pose3 T_base_imu;

  /// Camera looking along base +x, optical z forward / x right / y down.
  static gtsam::Pose3 ForwardCamera(const Eigen::Vector3d& t = Eigen::Vector3d(0.1, 0, 0)) {
    Eigen::Matrix3d R;
    R << 0, 0, 1, -1, 0, 0, 0, -1, 0;  // columns: optical x,y,z in base axes
    return gtsam::Pose3(gtsam::Rot3(R), t);
  }
};

}  // namespace strataslam
