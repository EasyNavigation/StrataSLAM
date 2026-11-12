#pragma once

#include <memory>
#include <vector>

#include "strataslam/frontend/frontend.hpp"

namespace strataslam {

struct LidarFrontendConfig {
  double min_range = 0.5, max_range = 50.0;  ///< m
  double scan_voxel = 0.4;                   ///< m, downsampling of the registered scan
  double map_voxel = 0.4;                    ///< m, downsampling of the local map
  double map_radius = 35.0;                  ///< m, keyframes farther than this leave the local map
  int map_max_keyframes = 60;
  double max_correspondence = 1.5;  ///< m
  int num_neighbors = 10;           ///< covariance estimation
  int max_iterations = 20;
  int threads = 1;
  bool deskew = true;                  ///< constant-velocity deskew when per-point times exist
  double keyframe_translation = 0.25;  ///< m
  double keyframe_rotation_deg = 10.0;
  /// Registration is `Degraded` when the smallest normalized translation eigenvalue of the
  /// Hessian falls below this (long corridors, elevator shafts).
  double degeneracy_threshold = 0.05;
  /// Keep the constant-velocity prediction along the weak direction when `Degraded`. Off by
  /// default: it diverged at the turn of a U-shaped stairwell (ADR-0012); the backend already
  /// widens the odometry covariance of degraded keyframes.
  bool pin_degenerate = false;
  /// m/s^2. When `Degraded` (and not pinned), how far the registration may leave the
  /// constant-velocity prediction along the weak direction in one scan: this acceleration over the
  /// scan interval squared (0.3 m at 10 Hz). Along that direction the registration can slide to a
  /// repeated structure, such as the next flight of a stairwell, and the velocity model would then
  /// carry the slide on; no robot or person moves its sensor at 3 g. 0 = no bound.
  double max_acceleration = 30.0;
  double min_fitness = 0.25;  ///< below: `Lost`
  /// Voxel size of the downsampled scans kept in keyframes (loop closure / mapping).
  double keyframe_cloud_voxel = 0.3;
  /// With the IMU's gravity (`SensorFrame::up_base`): the registered roll and pitch are kept within
  /// this of it, beyond it brought back to the bound (as the visual frontend does). A drifting
  /// attitude turns forward motion into a climb: on a ramp it is the error of the floor's height.
  double max_tilt_error_deg = 1.0;
  /// m. A scan whose points lie within this (95th percentile of their range) sees a confined
  /// space, an elevator cabin: a box of 1-2 m, often reflective, whose turns the registration
  /// gets wrong by tens of degrees. There, with the gyroscope (`SensorFrame::gyro_delta`), the
  /// rotation is the gyroscope's and the registration solves the translation only. Elsewhere the
  /// gyroscope is not used. Out of it once the scan sees twice as far. 0: never.
  double confined_range = 3.0;
  /// The scans come from a planar scanner (`lidar.type: 2d`, ADR-0066): the registration solves the
  /// sensor's x, y and heading only (the motion in its plane), its height, roll and pitch kept at
  /// the prediction. A sweep is one line of points in that plane, whose covariances cannot tell the
  /// wall's normal from the vertical: for the registration it is stacked in `planar_layers` copies
  /// `planar_layer_spacing` apart along the sensor's z, each wall a vertical strip. The clouds kept
  /// (keyframes, map) stay the sweep itself.
  bool planar_scan = false;
  /// Fewer points than this after the downsampling: the scan is not tracked. A 3D LiDAR always has
  /// hundreds; a planar sweep facing a wall a metre away has some 60 cells of 5 cm.
  int min_points = 100;
  int planar_layers = 3;
  double planar_layer_spacing = 0.1;  ///< m
  /// With the robot's own odometry as the prior (`SensorFrame::odom`, ADR-0071), a `Degraded`
  /// registration whose motion differs from the odometry's by more than `odom_veto_m` plus
  /// `odom_veto_ratio` of the step, or by more than `odom_veto_deg` plus that share of the turn, is
  /// taken for a slide and the odometry carries the scan. A `Good` one is the LiDAR's: wheels slip,
  /// and a skid-steer base's turns are what its wheels measure worst.
  double odom_veto_m = 0.10, odom_veto_ratio = 0.2, odom_veto_deg = 3.0;
};

/// Scan-to-local-map LiDAR odometry with GICP (small_gicp).
class LidarFrontend final : public OdometryFrontend {
 public:
  explicit LidarFrontend(const LidarFrontendConfig& config,
                         const gtsam::Pose3& T_base_lidar = gtsam::Pose3());
  ~LidarFrontend() override;

  FrontendResult process(const SensorFrame& frame) override;
  TrackingQuality quality() const override;
  void reset(const gtsam::Pose3& T_init) override;
  void request_keyframe() override;

  /// Downsampled scan (lidar frame) of the last keyframe, for the backend; empty if not a keyframe.
  const std::vector<Eigen::Vector3f>& last_keyframe_cloud() const;

  /// Registers `source` to `target` (both in their own frames), returning T_target_source and the
  /// quality indicators. Exposed for loop-closure verification and tests.
  struct RegistrationOutput {
    gtsam::Pose3 T_target_source;
    bool converged = false;
    TrackingQuality quality;
    double mean_error = 0;
    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
  };
  static RegistrationOutput register_clouds(const std::vector<Eigen::Vector3f>& target,
                                            const std::vector<Eigen::Vector3f>& source,
                                            const gtsam::Pose3& init_T_target_source,
                                            const LidarFrontendConfig& config);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace strataslam
