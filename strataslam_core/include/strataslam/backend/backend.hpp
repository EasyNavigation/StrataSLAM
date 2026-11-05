#pragma once

#include <map>
#include <memory>
#include <vector>

#include "strataslam/backend/floor_graph.hpp"
#include "strataslam/common/logging.hpp"
#include "strataslam/frontend/frontend.hpp"

namespace strataslam {

struct KeyframeInput {
  KeyframeId id = 0;
  Timestamp stamp = 0;
  SessionId session = 0;
  FloorId floor = kUnassignedFloor;       ///< -1 while a floor change is in progress
  gtsam::Pose3 T_odom_base;               ///< pose in the odometry frame
  TrackingState tracking = TrackingState::Good;
  std::vector<Eigen::Vector3f> cloud;     ///< downsampled scan in the sensor frame (may be empty)
  gtsam::Pose3 T_base_sensor;             ///< sensor extrinsic used for `cloud`
  std::optional<double> baro_height;      ///< filtered barometric height at the keyframe (m)
  std::optional<Eigen::Vector3d> up_body; ///< measured up direction when the sensor was quasi-static
  std::optional<Eigen::Vector3d> plane_up;///< planar mode: the base stays on the floor's plane, with this up
};

struct AcceptedLoop {
  KeyframeId from, to;
  FloorId from_floor, to_floor;           ///< floors the backend assigned (equal unless gating is off)
};

/// Owns one FloorGraph per floor. There is no code path that adds a loop between different floors
/// (three independent layers, each covered by a test):
///   1. candidate search only indexes keyframes of the current floor (see `floor_of`/pipeline),
///   2. `verify_loop_allowed` rejects pairs with different or unassigned floors,
///   3. `FloorGraph::add_loop` refuses a factor whose endpoints are not both nodes of that graph
///      (graph membership, independent of the floor labels layer 2 compares).
class Backend {
 public:
  Backend(const BackendConfig& config, std::shared_ptr<Logger> logger = std::make_shared<NullLogger>());

  /// Inserts a keyframe. Keyframes with floor -1 are registered (they belong to transitions) but
  /// join no graph. Creates the floor graph on first use, with its origin at this keyframe.
  Result<void> add_keyframe(const KeyframeInput& kf, std::optional<KeyframeId> odom_from,
                            const gtsam::Pose3& odom_rel, std::optional<KeyframeId> bridge_from);
  void add_gravity(KeyframeId id, const Eigen::Vector3d& up_body, double sigma_rad);

  /// Assigns keyframes that were registered without a floor (a session whose floor was pending) to
  /// `floor`, in order, using the recorded odometry between them. `bridge_from` links the first
  /// one to an earlier keyframe of that floor with a weak factor, or, if `bridge_is_odometry` and both
  /// are in the same session, with the odometry between them (motion that was tracked continuously).
  Result<void> promote_to_floor(const std::vector<KeyframeId>& ids, FloorId floor, std::optional<KeyframeId> bridge_from,
                                bool bridge_is_odometry = false);

  /// Keyframes tracked through a ramp or stairs (chronological `ids`, no floor yet) are placed on the
  /// floor they arrived at, chained backwards from `next`, the first keyframe of that floor (already in
  /// its graph), so that the floor frame keeps its origin on the floor and the climb is in the map.
  Result<void> promote_transit_before(const std::vector<KeyframeId>& ids, FloorId floor, KeyframeId next);

  /// Layer 2: floors must be equal and assigned.
  Result<void> verify_loop_allowed(KeyframeId a, KeyframeId b) const;
  /// Layer 3 + insertion. `T_from_to` is the measured relative pose between the base frames.
  Result<void> add_loop(KeyframeId from, KeyframeId to, const gtsam::Pose3& T_from_to);

  bool has_keyframe(KeyframeId id) const { return registry_.count(id) > 0; }
  FloorId floor_of(KeyframeId id) const;
  std::vector<FloorId> floors() const;
  /// Read-only: every insertion goes through the backend, which applies the floor gate (audit A4A5-05: a mutable
  /// FloorGraph& let a caller insert a keyframe of another floor and then close a loop to it).
  const FloorGraph& floor(FloorId id) const { return *graphs_.at(graph_key(id)); }
  /// Pose in the odometry frame of the (optimized) keyframe, or the recorded odometry for -1.
  gtsam::Pose3 pose_odom(KeyframeId id) const;
  const std::vector<AcceptedLoop>& loops() const { return loops_; }
  const std::vector<Eigen::Vector3f>* cloud(KeyframeId id) const;
  const KeyframeInput& keyframe(KeyframeId id) const { return registry_.at(id); }
  std::vector<KeyframeId> keyframe_ids() const { return order_; }

  void optimize_all();

 private:
  FloorId graph_key(FloorId f) const { return cfg_.debug_disable_floor_gating ? 0 : f; }
  void insert_into_graph(const KeyframeInput& kf, std::optional<KeyframeId> odom_from, const gtsam::Pose3& odom_rel,
                         std::optional<KeyframeId> bridge_from);

  BackendConfig cfg_;
  std::shared_ptr<Logger> log_;
  std::map<FloorId, std::unique_ptr<FloorGraph>> graphs_;
  std::map<KeyframeId, KeyframeInput> registry_;
  std::vector<KeyframeId> order_;
  std::vector<AcceptedLoop> loops_;
};

}  // namespace strataslam
