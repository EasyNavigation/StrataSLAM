#pragma once

#include <memory>
#include <vector>

#include "strataslam/frontend/frontend.hpp"

namespace strataslam {

struct LidarFrontendConfig {
  double min_range = 0.5, max_range = 50.0;   ///< m
  double scan_voxel = 0.4;                    ///< m, downsampling of the registered scan
  double map_voxel = 0.4;                     ///< m, downsampling of the local map
  double map_radius = 35.0;                   ///< m, keyframes farther than this leave the local map
  int map_max_keyframes = 60;
  double max_correspondence = 1.5;            ///< m
  int num_neighbors = 10;                     ///< covariance estimation
  int max_iterations = 20;
  int threads = 1;
  bool deskew = true;                         ///< constant-velocity deskew when per-point times exist
  double keyframe_translation = 0.25;         ///< m
  double keyframe_rotation_deg = 10.0;
  /// Registration is `Degraded` when the smallest normalized translation eigenvalue of the
  /// Hessian falls below this (long corridors, elevator shafts).
  double degeneracy_threshold = 0.05;
  /// Keep the constant-velocity prediction along the weak direction when `Degraded`. Off by default:
  /// it diverged at the turn of a U-shaped stairwell (ADR-0012); the backend already widens the
  /// odometry covariance of degraded keyframes.
  bool pin_degenerate = false;
  double min_fitness = 0.25;                  ///< below: `Lost`
  /// Voxel size of the downsampled scans kept in keyframes (loop closure / mapping).
  double keyframe_cloud_voxel = 0.3;
};

/// Scan-to-local-map LiDAR odometry with GICP (small_gicp).
class LidarFrontend final : public OdometryFrontend {
 public:
  explicit LidarFrontend(const LidarFrontendConfig& config, const gtsam::Pose3& T_base_lidar = gtsam::Pose3());
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
