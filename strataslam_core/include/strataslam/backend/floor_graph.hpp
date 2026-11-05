#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "strataslam/common/result.hpp"
#include "strataslam/common/types.hpp"
#include "strataslam/geometry/pose4.hpp"
#include "strataslam/io/map_io.hpp"

namespace strataslam {

struct BackendConfig {
  double odom_sigma_trans = 0.02;  ///< m, per-keyframe odometry
  double odom_sigma_rot = 0.004;   ///< rad
  double degraded_scale = 10.0;    ///< noise multiplier when tracking is Degraded
  double loop_sigma_trans = 0.05;
  double loop_sigma_rot = 0.01;
  double loop_cauchy_k = 1.0;       ///< robust kernel parameter (whitened units)
  double bridge_sigma_trans = 1.0;  ///< weak link between consecutive sessions of a floor
  double bridge_sigma_rot = 0.2;
  double baro_sigma = 0.10;            ///< m, per consecutive-keyframe barometric height difference
  double gravity_sigma = 0.02;         ///< rad, attitude prior from a quasi-static IMU window
  double plane_sigma = 0.01;           ///< m, planar mode: height of the base in the floor frame
  double plane_attitude_sigma = 0.01;  ///< rad, planar mode: roll and pitch of the base
  double mahalanobis_chi2 = 22.46;     ///< chi2(6 dof) at 0.999
  /// Debug only, never exposed in the user YAML: removes the floor gate and the consistency
  /// check so that the negative control (scenario S04) can demonstrate the collapse.
  bool debug_disable_floor_gating = false;
};

/// Pose graph of one floor (submap): keyframe poses in the floor frame, iSAM2 incremental
/// optimization and batch optimization when the floor is closed.
class FloorGraph {
 public:
  FloorGraph(FloorId id, const gtsam::Pose3& T_odom_floor, const BackendConfig& config);

  FloorId id() const { return id_; }
  const gtsam::Pose3& T_odom_floor() const { return T_odom_floor_; }
  Pose4 origin() const;  ///< floor frame origin in the odometry frame as Pose4

  struct Node {
    KeyframeId id;
    Timestamp stamp;
    SessionId session;
    gtsam::Pose3 initial;  ///< T_floor_base from odometry
  };

  /// `bridge_from` = previous keyframe of an earlier session to link with a weak factor.
  void add_keyframe(KeyframeId id, Timestamp stamp, SessionId session,
                    const gtsam::Pose3& T_floor_base, std::optional<KeyframeId> odom_from,
                    const gtsam::Pose3& odom_rel, double noise_scale,
                    std::optional<KeyframeId> bridge_from,
                    std::optional<double> baro_dz = std::nullopt);
  /// Gravity-attitude prior on one keyframe: `up_body` is the measured unit up direction.
  void add_gravity(KeyframeId id, const Eigen::Vector3d& up_body, double sigma_rad);
  /// Planar mode: the keyframe's base stays on the floor's plane (height 0 in the floor frame) with
  /// `up_body` up.
  void add_plane(KeyframeId id, const Eigen::Vector3d& up_body);
  /// Loop closure measurement T_from_to between two keyframes of this floor.
  Result<void> add_loop(KeyframeId from, KeyframeId to, const gtsam::Pose3& T_from_to,
                        bool check_consistency);
  /// Squared Mahalanobis distance of a loop measurement against the current estimate.
  double loop_mahalanobis2(KeyframeId from, KeyframeId to, const gtsam::Pose3& T_from_to) const;

  bool has(KeyframeId id) const { return nodes_.count(id) > 0; }
  size_t size() const { return nodes_.size(); }
  gtsam::Pose3 pose(KeyframeId id) const;
  std::vector<KeyframeId> keyframe_ids() const;  ///< chronological order
  void sort_by_stamp();                          ///< after keyframes were inserted out of order
  const Node& node(KeyframeId id) const { return nodes_.at(id); }

  /// Full Levenberg-Marquardt optimization of the stored graph; refreshes the estimates.
  void optimize_batch();

  std::vector<EdgeRecord> edge_records() const { return edges_; }

 private:
  void rebuild_isam();

  FloorId id_;
  gtsam::Pose3 T_odom_floor_;
  BackendConfig cfg_;
  std::unique_ptr<gtsam::ISAM2> isam_;
  gtsam::NonlinearFactorGraph graph_;  // all factors (for batch optimization / marginals)
  gtsam::Values estimate_;
  std::map<KeyframeId, Node> nodes_;
  std::vector<KeyframeId> order_;
  std::vector<EdgeRecord> edges_;
};

}  // namespace strataslam
