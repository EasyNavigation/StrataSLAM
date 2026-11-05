#pragma once

#include <gtsam/geometry/Pose3.h>

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "strataslam/frontend/depth_utils.hpp"
#include "strataslam/sensors/measurements.hpp"

namespace strataslam {

struct VisualTrackerConfig {
  BackprojectionConfig depth;
  int min_inliers = 30;   ///< below: tracking is Lost
  int good_inliers = 80;  ///< at or above: Good (between: Degraded)
  double ransac_reprojection_px = 3.0;
  int ransac_iterations = 200;
  double match_radius_px = 25.0;  ///< guided matching search window
  double wide_radius_px = 90.0;   ///< window after a loss or poor prediction
  int max_hamming = 48;           ///< of 256 bits
  double ratio = 0.8;             ///< best/second-best Hamming ratio inside the window
  int local_map_keyframes = 10;
  int max_new_landmarks = 400;
  double keyframe_translation = 0.25;  ///< m
  double keyframe_rotation_deg = 10.0;
  double keyframe_tracked_ratio =
      0.6;                       ///< new keyframe when tracked points fall below this fraction
  bool refine_landmarks = true;  ///< average landmark positions over successive measurements
  int landmark_max_obs = 8;
  bool use_depth_in_refinement = true;  ///< joint reprojection + depth residual refinement
  double pixel_sigma = 0.7;             ///< px, reprojection residual weight
  double depth_sigma_coeff =
      0.003;  ///< sigma_z = coeff * z^2 + floor (landmark and measurement noise combined)
  double depth_sigma_floor = 0.01;  ///< m
};

/// Keypoints of one frame with their camera-frame 3D point where the depth is valid.
struct VisualObservation {
  std::vector<RgbKeypoint> keypoints;
  std::vector<std::optional<Eigen::Vector3d>> p_cam;
};

/// Builds the observation by back-projecting keypoints with the depth image.
VisualObservation make_observation(const RgbdFrame& frame, const BackprojectionConfig& cfg);

struct PnPCorrespondence {
  Eigen::Vector3d point;  ///< 3D point in the object frame
  Eigen::Vector2d pixel;
};
struct PnPResult {
  bool ok = false;
  gtsam::Pose3 T_cam_obj;    ///< maps object-frame points to the camera frame
  std::vector<int> inliers;  ///< indices into the correspondence list
};
/// PnP with RANSAC followed by Levenberg-Marquardt refinement on the inliers.
PnPResult solve_pnp(const std::vector<PnPCorrespondence>& corr, const CameraIntrinsics& K,
                    const gtsam::Pose3* init_T_cam_obj, double reprojection_px, int iterations);
/// Geometric verification of a visual loop candidate: PnP-RANSAC on the descriptor matches,
/// accepted only with at least `min_inliers` matches and inliers (spec 9.7: 30). Returns T_cam_obj,
/// or nullopt.
std::optional<gtsam::Pose3> verify_loop_pnp(const std::vector<PnPCorrespondence>& corr,
                                            const CameraIntrinsics& K, int min_inliers,
                                            double reprojection_px, int iterations);

/// One matched landmark for the RGB-D refinement: map point, observed pixel and (optionally) the
/// depth measured at that pixel.
struct RgbdMatch {
  Eigen::Vector3d point;  ///< landmark in the object (odometry) frame
  Eigen::Vector2d pixel;
  std::optional<double> depth;  ///< measured z-depth at the keypoint
};
/// Gauss-Newton refinement of T_cam_obj minimizing reprojection error plus, where available, the
/// difference between predicted and measured depth. PnP alone constrains the optical axis poorly;
/// the depth term fixes that axis, which is the direction of travel in a corridor.
gtsam::Pose3 refine_rgbd(const std::vector<RgbdMatch>& matches, const CameraIntrinsics& K,
                         const gtsam::Pose3& init, double pixel_sigma, double depth_sigma_coeff,
                         double depth_sigma_floor, int iterations);

/// Hamming distance between two 256-bit descriptors.
int hamming_distance(const Descriptor& a, const Descriptor& b);

/// What the last tracked frame did with each of its keypoints.
enum class KeypointUse : uint8_t {
  Unmatched,  ///< no landmark matched it (new keyframes turn the strongest ones into landmarks)
  Rejected,   ///< matched, but an outlier of PnP RANSAC, or the frame was lost
  Inlier,     ///< matched and used to estimate the pose
};

/// Local-map visual tracker: guided matching by projection, PnP RANSAC, keyframe landmark map.
class VisualTracker {
 public:
  explicit VisualTracker(const VisualTrackerConfig& config);
  ~VisualTracker();

  void reset();
  bool initialized() const;
  size_t map_size() const;

  struct TrackResult {
    bool ok = false;
    gtsam::Pose3 T_odom_cam;
    int matches = 0;
    int inliers = 0;
    bool want_keyframe = false;
  };
  /// Starts the map from this frame at pose `T_odom_cam` (first frame, or after a reset).
  void initialize(const CameraIntrinsics& K, const VisualObservation& obs,
                  const gtsam::Pose3& T_odom_cam);
  /// Tracks a frame against the local map starting from the predicted pose.
  TrackResult track(const CameraIntrinsics& K, const VisualObservation& obs,
                    const gtsam::Pose3& predicted_T_odom_cam, bool widen_search);
  /// Adds landmarks for the keypoints not matched in the last `track` call and prunes old ones.
  void insert_keyframe(const VisualObservation& obs, const gtsam::Pose3& T_odom_cam);
  /// Per keypoint of the last frame given to `initialize` or `track` (empty after `reset`).
  const std::vector<KeypointUse>& last_keypoint_use() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace strataslam
