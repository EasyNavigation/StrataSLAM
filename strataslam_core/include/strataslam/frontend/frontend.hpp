#pragma once

#include <gtsam/geometry/Pose3.h>

#include <Eigen/Core>
#include <deque>
#include <optional>

#include "strataslam/sensors/measurements.hpp"

namespace strataslam {

enum class TrackingState { Good, Degraded, Lost };

/// Quality report used by the floor manager and the backend.
struct TrackingQuality {
  TrackingState state = TrackingState::Good;
  int inliers = 0;       ///< visual: PnP inliers; LiDAR: correspondences
  double fitness = 1.0;  ///< LiDAR: inlier fraction of the source points
  double hessian_condition =
      1.0;  ///< LiDAR: max/min eigenvalue of the normalized translation Hessian
  double min_eigenvalue =
      1.0;  ///< LiDAR: smallest normalized translation eigenvalue (0 = degenerate)
};

/// One synchronized input for a frontend. Pointers are valid during `process` only.
struct SensorFrame {
  Timestamp stamp = 0;
  const RgbdFrame* rgbd = nullptr;
  const LidarScan* lidar = nullptr;
  const OdomMeasurement* odom =
      nullptr;  ///< the robot's own odometry (wheels) at this stamp: a motion prior
  std::optional<Eigen::Vector3d>
      up_base;  ///< up in the base frame from the IMU's gravity: the pose's roll and pitch
};

struct FrontendResult {
  bool valid = false;        ///< a pose was produced
  gtsam::Pose3 T_odom_base;  ///< base pose in the (continuous) odometry frame
  bool is_keyframe = false;
  TrackingQuality quality;
  /// Pose increment since the previous result, expressed in the base frame.
  gtsam::Pose3 delta;
};

/// Common interface of the odometry frontends (visual, LiDAR, external).
class OdometryFrontend {
 public:
  virtual ~OdometryFrontend() = default;
  virtual FrontendResult process(const SensorFrame& frame) = 0;
  virtual TrackingQuality quality() const = 0;
  /// Clears all local state and restarts at `T_init` (used after tracking loss or floor changes).
  virtual void reset(const gtsam::Pose3& T_init) = 0;
  /// Makes the next valid frame a keyframe (e.g. right after a floor change, so that the new floor
  /// has a keyframe even when the robot is hovering).
  virtual void request_keyframe() = 0;
};

}  // namespace strataslam
