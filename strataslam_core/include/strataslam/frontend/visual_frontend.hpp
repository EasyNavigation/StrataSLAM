#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "strataslam/frontend/frontend.hpp"
#include "strataslam/frontend/visual_tracker.hpp"

namespace strataslam {

/// Data of the last keyframe that the backend and place recognition need.
struct VisualKeyframeData {
  std::vector<RgbKeypoint> keypoints;
  std::vector<std::optional<Eigen::Vector3d>> p_cam;  ///< camera-frame 3D point per keypoint
  std::vector<Eigen::Vector3f> cloud;                  ///< subsampled depth cloud, optical frame
  CameraIntrinsics K;
};

struct VisualFrontendConfig {
  VisualTrackerConfig tracker;
  int cloud_stride = 8;  ///< pixels between depth samples in the keyframe cloud
  /// With an IMU: largest roll/pitch difference from its gravity that the tracking may keep (about the gravity
  /// error of an IMU on a moving body).
  double max_tilt_error_deg = 2.0;
  /// With an external odometry: a frame whose motion differs from the odometry's by more than `veto_m` plus
  /// `veto_ratio` of the step, or by more than `veto_deg` of rotation, is a wrong solution (about what wheel
  /// odometry can err in one frame, plus the PnP noise): the odometry carries it.
  double veto_m = 0.02, veto_ratio = 0.1, veto_deg = 1.0;
};

/// RGB-D odometry frontend: depth back-projection, local-map tracking, keyframe selection.
class VisualFrontend final : public OdometryFrontend {
 public:
  VisualFrontend(const VisualFrontendConfig& config, const gtsam::Pose3& T_base_camera);
  ~VisualFrontend() override;

  FrontendResult process(const SensorFrame& frame) override;
  TrackingQuality quality() const override;
  void reset(const gtsam::Pose3& T_init) override;
  void request_keyframe() override;

  const VisualKeyframeData& last_keyframe() const;
  /// What the tracker did with each keypoint of the last frame (empty before the map starts).
  const std::vector<KeypointUse>& last_keypoint_use() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace strataslam
