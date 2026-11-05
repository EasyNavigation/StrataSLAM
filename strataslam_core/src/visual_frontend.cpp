#include "strataslam/frontend/visual_frontend.hpp"

#include <cmath>
#include <deque>
#include <optional>

#include "strataslam/geometry/gravity.hpp"

namespace strataslam {

struct VisualFrontend::Impl {
  VisualFrontendConfig cfg;
  gtsam::Pose3 T_base_cam;
  VisualTracker tracker;
  gtsam::Pose3 T_odom_base;
  gtsam::Pose3 velocity;  // per-frame body delta
  std::deque<gtsam::Pose3> recent;
  Timestamp last_stamp = 0;
  bool lost_previous = false;
  bool force_kf = false;
  TrackingQuality last_quality;
  VisualKeyframeData kf_data;
  std::optional<gtsam::Pose3>
      last_ext;  // external odometry at the previous frame (its increments are the prior)
  gtsam::Pose3 last_kf_base;  // base pose of the last keyframe

  Impl(const VisualFrontendConfig& c, const gtsam::Pose3& T)
      : cfg(c), T_base_cam(T), tracker(c.tracker) {}

  /// The camera pose with the base's roll and pitch bounded by gravity (`up_base`, from the IMU):
  /// within `max_tilt_error_deg` of it they are the tracking's own (more precise than an IMU on a
  /// moving body), beyond it they are brought back to that bound; heading and camera centre (what
  /// PnP measures) kept. The tracking alone can let them drift by tens of degrees (a corridor),
  /// which turns forward motion into a climb.
  gtsam::Pose3 level_camera(const gtsam::Pose3& T_odom_cam, const Eigen::Vector3d& up_base) const {
    const gtsam::Rot3 R_base = T_odom_cam.rotation() * T_base_cam.rotation().inverse();
    const Eigen::Vector3d up_now =
        R_base.matrix().transpose() * Eigen::Vector3d::UnitZ();  // the tracking's up
    const Eigen::Vector3d up_imu = up_base.normalized();
    const double angle = std::acos(std::clamp(up_now.dot(up_imu), -1.0, 1.0));
    const double bound = cfg.max_tilt_error_deg * M_PI / 180.0;
    if (angle <= bound) return T_odom_cam;
    // The direction at `bound` from gravity on the way to the tracking's: the smallest correction.
    const Eigen::Vector3d axis = up_imu.cross(up_now).normalized();
    const Eigen::Vector3d up_target = Eigen::AngleAxisd(bound, axis) * up_imu;
    return gtsam::Pose3(level_to(R_base, up_target) * T_base_cam.rotation(),
                        T_odom_cam.translation());
  }

  /// The camera lost the frame, or its motion was vetoed, but the external odometry has the motion:
  /// the pose follows it (degraded, so the backend weighs it less) and no tracking loss reaches the
  /// pipeline. After a loss
  /// (`restart_map`) the local map, which no longer matches, starts again from this frame when it
  /// has enough 3D points; after a veto it stays (only that solution was wrong). A keyframe is made
  /// once the robot has moved.
  FrontendResult& bridge(const RgbdFrame& f, const VisualObservation& obs,
                         const gtsam::Pose3& new_base, const gtsam::Pose3& delta,
                         FrontendResult& res, bool restart_map) {
    res.delta = delta;
    res.T_odom_base = new_base;
    res.valid = true;
    res.quality.state = TrackingState::Degraded;
    last_quality = res.quality;
    T_odom_base = new_base;
    velocity = delta;
    recent.clear();
    recent.push_back(new_base);
    int valid = 0;
    for (const auto& p : obs.p_cam) valid += p.has_value();
    lost_previous =
        restart_map &&
        valid < cfg.tracker.min_inliers;  // no new map: the old one, with a wider search
    if (lost_previous) return res;
    if (restart_map) tracker.initialize(f.K, obs, new_base * T_base_cam);
    const gtsam::Pose3 moved = last_kf_base.between(new_base);
    if (moved.translation().norm() > cfg.tracker.keyframe_translation ||
        gtsam::Rot3::Logmap(moved.rotation()).norm() * 180.0 / M_PI >
            cfg.tracker.keyframe_rotation_deg) {
      if (!restart_map) tracker.insert_keyframe(obs, new_base * T_base_cam);
      store_keyframe(f, obs);
      last_kf_base = new_base;
      res.is_keyframe = true;
    }
    return res;
  }

  void store_keyframe(const RgbdFrame& f, const VisualObservation& obs) {
    kf_data.keypoints = obs.keypoints;
    kf_data.p_cam = obs.p_cam;
    kf_data.K = f.K;
    kf_data.cloud = depth_to_cloud(f.depth, f.K, cfg.cloud_stride, cfg.tracker.depth);
  }
};

VisualFrontend::VisualFrontend(const VisualFrontendConfig& c, const gtsam::Pose3& T_base_camera)
    : impl_(std::make_unique<Impl>(c, T_base_camera)) {}
VisualFrontend::~VisualFrontend() = default;

TrackingQuality VisualFrontend::quality() const { return impl_->last_quality; }
const VisualKeyframeData& VisualFrontend::last_keyframe() const { return impl_->kf_data; }
const std::vector<KeypointUse>& VisualFrontend::last_keypoint_use() const {
  return impl_->tracker.last_keypoint_use();
}

void VisualFrontend::request_keyframe() { impl_->force_kf = true; }

void VisualFrontend::reset(const gtsam::Pose3& T_init) {
  auto& s = *impl_;
  s.tracker.reset();
  s.T_odom_base = T_init;
  s.velocity = gtsam::Pose3();
  s.recent.clear();
  s.recent.push_back(T_init);
  s.last_stamp = 0;
  s.lost_previous = false;
  s.last_quality = TrackingQuality();
  s.last_ext.reset();
}

FrontendResult VisualFrontend::process(const SensorFrame& frame) {
  auto& s = *impl_;
  FrontendResult res;
  if (!frame.rgbd) return res;
  const RgbdFrame& f = *frame.rgbd;
  const VisualObservation obs = make_observation(f, s.cfg.tracker.depth);
  const gtsam::Pose3 T_cam_base = s.T_base_cam.inverse();
  // Motion of the base since the previous frame by the robot's own odometry (wheels), when it has
  // both.
  std::optional<gtsam::Pose3> ext_delta;
  if (frame.odom && s.last_ext) ext_delta = s.last_ext->between(frame.odom->pose);
  s.last_ext = frame.odom ? std::optional<gtsam::Pose3>(frame.odom->pose) : std::nullopt;

  if (!s.tracker.initialized()) {
    // Needs enough 3D points to start a map.
    int valid = 0;
    for (const auto& p : obs.p_cam) valid += p.has_value();
    if (valid < s.cfg.tracker.min_inliers) {
      s.last_quality.state = TrackingState::Lost;
      res.quality = s.last_quality;
      return res;
    }
    s.tracker.initialize(f.K, obs, s.T_odom_base * s.T_base_cam);
    s.store_keyframe(f, obs);
    s.last_kf_base = s.T_odom_base;
    s.last_stamp = f.stamp;
    s.last_quality = TrackingQuality();
    s.last_quality.inliers = valid;
    res.valid = true;
    res.is_keyframe = true;
    res.T_odom_base = s.T_odom_base;
    res.quality = s.last_quality;
    return res;
  }

  // Prediction: the external odometry's motion when there is one; otherwise constant velocity (no
  // extrapolation across data gaps or after a loss).
  const double dt = s.last_stamp ? (f.stamp - s.last_stamp) * 1e-9 : 0.0;
  const bool gap = dt > 0.5;
  const gtsam::Pose3 pred_base = ext_delta                  ? s.T_odom_base * *ext_delta
                                 : (gap || s.lost_previous) ? s.T_odom_base
                                                            : s.T_odom_base * s.velocity;
  const auto tr = s.tracker.track(f.K, obs, pred_base * s.T_base_cam, gap || s.lost_previous);

  TrackingQuality q;
  q.inliers = tr.inliers;
  q.fitness = tr.matches > 0 ? static_cast<double>(tr.inliers) / tr.matches : 0.0;
  q.state = !tr.ok                                     ? TrackingState::Lost
            : tr.inliers >= s.cfg.tracker.good_inliers ? TrackingState::Good
                                                       : TrackingState::Degraded;
  s.last_quality = q;
  res.quality = q;
  s.last_stamp = f.stamp;
  // The robot's own odometry vetoes a motion it could not have made (a wrong PnP solution): such a
  // frame is carried by the odometry, like a lost one.
  bool implausible = false;
  if (tr.ok && ext_delta) {
    const gtsam::Pose3 cam_delta = s.T_odom_base.between(tr.T_odom_cam * T_cam_base);
    const double trans_err = (cam_delta.translation() - ext_delta->translation()).norm();
    const double rot_err_deg =
        gtsam::Rot3::Logmap(cam_delta.rotation().between(ext_delta->rotation())).norm() * 180.0 /
        M_PI;
    implausible = trans_err > s.cfg.veto_m + s.cfg.veto_ratio * ext_delta->translation().norm() ||
                  rot_err_deg > s.cfg.veto_deg;
  }
  if ((!tr.ok || implausible) && ext_delta) {
    // The camera measured nothing usable: the attitude is the IMU's, when there is one.
    const gtsam::Pose3 bridged =
        frame.up_base
            ? gtsam::Pose3(level_to(pred_base.rotation(), *frame.up_base), pred_base.translation())
            : pred_base;
    return s.bridge(f, obs, bridged, s.T_odom_base.between(bridged), res, !tr.ok);
  }
  if (!tr.ok) {
    s.lost_previous = true;
    return res;
  }
  s.lost_previous = false;
  const gtsam::Pose3 T_odom_cam =
      frame.up_base ? s.level_camera(tr.T_odom_cam, *frame.up_base) : tr.T_odom_cam;
  const gtsam::Pose3 new_base = T_odom_cam * T_cam_base;
  res.delta = s.T_odom_base.between(new_base);
  s.recent.push_back(new_base);
  if (s.recent.size() > 4) s.recent.pop_front();
  const size_t n = s.recent.size() - 1;
  s.velocity =
      n == 0
          ? res.delta
          : gtsam::Pose3::Expmap(gtsam::Pose3::Logmap(s.recent.front().between(s.recent.back())) /
                                 static_cast<double>(n));
  s.T_odom_base = new_base;
  res.T_odom_base = new_base;
  res.valid = true;
  if (tr.want_keyframe || s.force_kf) {
    s.force_kf = false;
    s.tracker.insert_keyframe(obs, T_odom_cam);
    s.store_keyframe(f, obs);
    s.last_kf_base = new_base;
    res.is_keyframe = true;
  }
  return res;
}

}  // namespace strataslam
