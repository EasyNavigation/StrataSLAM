#pragma once

#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "strataslam/backend/backend.hpp"
#include "strataslam/common/logging.hpp"
#include "strataslam/config/config.hpp"
#include "strataslam/floors/floor_manager.hpp"
#include "strataslam/frontend/lidar_frontend.hpp"
#include "strataslam/frontend/visual_frontend.hpp"
#include "strataslam/place_recognition/bow.hpp"
#include "strataslam/io/map_io.hpp"
#include "strataslam/place_recognition/scan_context.hpp"
#include "strataslam/sensors/measurements.hpp"

namespace strataslam {

/// Tunables that are not part of the user profile.
struct SystemTuning {
  BackendConfig backend;
  LidarFrontendConfig lidar;
  VisualFrontendConfig visual;
  /// Visual vocabulary (trained with `strataslam vocab train`). Without it visual loop closure is off.
  std::shared_ptr<const BowVocabulary> vocabulary;
  double bow_min_score = 0.04;          ///< candidate generation threshold
  int bow_max_candidates = 4;
  double bow_exclude_recent_s = 10.0;   ///< same-session keyframes this recent are not candidates
  int visual_loop_min_inliers = 30;     ///< PnP inliers required to accept a visual loop
  double max_relevel_deg = 3.0;         ///< largest attitude correction from a ground plane at a frontend restart
  double max_relevel_imu_deg = 10.0;    ///< largest one from the IMU at rest
  FloorManagerConfig floors;            ///< thresholds not exposed in the profile (floor_height_prior etc. come from it)
  int pending_keyframes_for_new_floor = 8;  ///< unlocalized keyframes after a loss before a new floor is opened
  ScanContextConfig scan_context;
  double loop_min_fitness = 0.55;        ///< GICP inlier fraction required to accept a LiDAR loop
  double loop_max_mean_error = 0.12;     ///< m^2 mean squared correspondence error (small_gicp units)
  int loop_cooldown_keyframes = 8;       ///< minimum keyframes between accepted loops
  int loop_submap_radius = 2;            ///< neighbours merged on each side of the candidate
};

struct KeyframeInfo {
  KeyframeId id = 0;
  Timestamp stamp = 0;
  SessionId session = 0;
  FloorId floor = kUnassignedFloor;
  gtsam::Pose3 T_odom_base;  ///< pose from the frontend when the keyframe was created
};

/// The online SLAM pipeline: sensor callbacks in, floors/keyframes/loops out. ROS-free.
class SlamSystem {
 public:
  SlamSystem(const Config& config, const Extrinsics& extrinsics, const SystemTuning& tuning = {},
             std::shared_ptr<Logger> logger = std::make_shared<NullLogger>());
  ~SlamSystem();

  void add_imu(const ImuSample& s);
  void add_baro(const BaroSample& s);
  /// A sample of the fixed base station (`barometer.reference_topic`): the rover's height becomes differential.
  void add_baro_reference(const BaroSample& s);
  void add_odom(const OdomMeasurement& s);
  void add_rgbd(const RgbdFrame& f);
  void add_lidar(const LidarScan& s);

  const std::vector<KeyframeInfo>& keyframes() const;
  const std::vector<AcceptedLoop>& loops() const;
  const std::vector<TransitionRecord>& transitions() const;
  FloorId current_floor() const;
  SessionId current_session() const;
  TrackingQuality tracking_quality() const;
  std::optional<gtsam::Pose3> current_pose_odom() const;  ///< latest frontend pose
  /// What the camera tracking did with each keypoint of the last RGB-D frame (empty before its map starts);
  /// nullopt when the camera is not the odometry source (its keypoints then only serve loop closures).
  std::optional<std::vector<KeypointUse>> keypoint_use() const;
  /// Altitude of a floor's level above floor 0 (m) as the floor manager measured it (barometer, IMU or
  /// odometry): the height between floors also when the odometry did not see the change (elevator, a
  /// tracking loss), unlike the odometry frame. nullopt for an unknown floor.
  std::optional<double> floor_altitude(FloorId floor) const;
  /// Filtered barometric height since start-up (m), once a barometer has reported.
  std::optional<double> baro_height() const;
  const Backend& backend() const;

  /// Candidates that place recognition returned from a floor other than the current one. The
  /// per-floor databases make this impossible by construction; the counter exists so a test can
  /// check that the first gating layer never has to be rescued by the later ones.
  size_t cross_floor_candidates_seen() const;
  struct FloorStatus {
    FloorState state = FloorState::OnFloor;
    FloorId floor = 0;            ///< -1 while a transition is in progress or the floor is pending
    bool pending = false;         ///< a session whose floor is not known yet
    size_t num_floors = 1;
    double altitude = 0;          ///< m above floor 0
    SessionId session = 0;
    std::optional<TransitionType> last_transition;
  };
  FloorStatus floor_status() const;
  /// T_floor_odom of the active floor (for the TF `floor_<k> -> odom`), if a floor graph exists.
  std::optional<gtsam::Pose3> floor_from_odom() const;
  /// Optimized keyframe poses (stamp, T_floor_base) of a floor, in insertion order.
  std::vector<std::pair<Timestamp, gtsam::Pose3>> floor_path(FloorId floor) const;
  /// Voxelized cloud of a floor in its own frame, rebuilt from the optimized keyframe poses.
  std::vector<Eigen::Vector3f> floor_cloud(FloorId floor, double voxel) const;
  /// Operator request: close the current session and start a new one on the same floor.
  void new_session();
  /// Operator request: the robot is now on the next floor (`~/mark_floor_change`).
  void mark_floor_change(TransitionType type = TransitionType::Unknown);

  /// Closes all floors (batch optimization) and returns the map in the on-disk model.
  MapData finalize(const std::string& profile_yaml = "");

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace strataslam
