#pragma once

#include <Eigen/Core>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "strataslam/common/result.hpp"
#include "strataslam/fusion/floor_stacking.hpp"
#include "strataslam/geometry/pose4.hpp"
#include "strataslam/io/map_io.hpp"

namespace strataslam {

enum class Formulation {
  Lie4,     ///< one Pose4 = SE(2)xR node per floor (the method of the paper)
  Se3Soft,  ///< Pose3 nodes with a soft roll/pitch prior (ablation)
  Euler,    ///< (x, y, z, yaw) with direct addition, no angle-wrap handling (ablation)
};
const char* to_string(Formulation f);
bool parse_formulation(const std::string& name, Formulation* out);

/// The same structural points seen from several floors (elevator door frame, shaft corners) or
/// points marked by the user. `points[floor]` lists them in that floor's frame; entry i of every
/// floor is the same physical point.
struct AnchorSet {
  std::string name;
  std::map<FloorId, std::vector<Eigen::Vector3d>> points;
  /// false: only x and y are compared (a structure replicated on every floor sits at another
  /// height).
  bool use_z = false;
  double sigma = 0.03;  ///< m, marking noise
};

/// Relative pose between base frames of two keyframes of different floors found by place
/// recognition inside a verified overlap zone (never discovered automatically).
struct CrossFloorMatch {
  FloorId floor_a = 0, floor_b = 0;
  KeyframeId kf_a = 0, kf_b = 0;
  Pose4 relative;  ///< pose of kf_b's base in kf_a's base frame
  Eigen::Matrix4d covariance = Eigen::Matrix4d::Identity() * 0.01;
};

/// The user states that two floors are the same level (for example, a drift of the odometry in z
/// split one level in two): their frames are constrained to the same height.
struct SameLevel {
  FloorId a = 0, b = 0;
  double sigma = 0.10;  ///< m
};

/// Two floors that could be the same level: their structure is compatible and their heights differ
/// by less than the drift budget. Reported, never applied; the user (or a height sensor) decides.
struct LevelCandidate {
  FloorId a = 0, b = 0;
  double dz = 0;     ///< m, height difference chained through the transitions
  double score = 0;  ///< structural overlap (floor_stacking)
  bool structure_ambiguous = false;
};

struct FusionOptions {
  Formulation formulation = Formulation::Lie4;
  std::optional<double> floor_height_prior;
  /// m. The prior is the user's statement that floors are about this far apart. 0.10 m lets it
  /// correct the height bias of a LiDAR climbing a stairwell (~0.2 m) while a barometer (0.05 m) or
  /// good odometry still dominate it.
  double floor_height_sigma = 0.10;
  std::vector<AnchorSet> anchors;
  std::vector<CrossFloorMatch> matches;
  /// Register the structure of every pair of floors against each other (floor_stacking.hpp) and use
  /// the unambiguous results as extra constraints. A pair whose structure does not decide the
  /// alignment contributes nothing.
  std::vector<SameLevel> same_level;
  /// m: floors closer in height than this are listed as candidates for being one level (0 = no
  /// search).
  double same_level_max_dz = 2.0;
  double same_level_min_score = 0.3;
  bool auto_stacking = true;
  StackingOptions stacking;
  /// Drift of the odometry that was integrated along a transition: sigma grows with the distance
  /// travelled (m per m, rad per m). Without it a long ramp would be trusted as much as a short
  /// one.
  double transition_drift_per_m = 0.03;
  double transition_yaw_drift_per_m = 0.0005;
  bool use_gnc = true;
  double anchor_outlier_bound = 0.10;  ///< m, inlier bound of the robust anchor alignment
  int max_iterations = 100;
};

struct ConstraintReport {
  std::string kind;  ///< transition, baro_height, floor_height_prior, cross_floor_match, anchor
  std::string label;
  FloorId a = 0, b = 0;
  bool accepted = true;
  double weight = 1.0;
  double residual = 0.0;  ///< whitened error norm at the solution
};

struct FusionResult {
  Formulation formulation = Formulation::Lie4;
  std::map<FloorId, Pose4> floor_to_map;  ///< T_map_floor, floor 0 = identity
  std::map<FloorId, Eigen::Matrix4d>
      covariance;  ///< marginal covariance in the Pose4 tangent space (phi, rho)
  std::vector<ConstraintReport> constraints;
  std::vector<std::string> warnings;
  std::vector<LevelCandidate> same_level_candidates;
  bool converged = false;
  int iterations =
      0;  ///< Levenberg-Marquardt iterations; -1 with GNC, whose inner iterations are not exposed
  double seconds = 0;  ///< the whole of fuse(), structural stacking included
  double optimize_seconds =
      0;  ///< graph construction and optimization only: what differs between formulations
  double final_error = 0;
  bool finite() const;
  std::string report_yaml() const;
  std::string transforms_yaml() const;
};

Result<FusionResult> fuse(const MapData& map, const FusionOptions& options);

/// Floors a and b checked as `fuse` checks every pair (ADR-0031): b's structure registered on a's
/// around `a_to_b` (pose of b's frame in a's, e.g. from the odometry), allowing an ambiguous match.
/// A candidate when the structure matches with at least `same_level_min_score` and the heights are
/// at most `same_level_max_dz` apart. Reported, never applied; `why` says why a pair is not one.
std::optional<LevelCandidate> same_level_candidate(FloorId a, const PointCloud& cloud_a, FloorId b,
                                                   const PointCloud& cloud_b, const Pose4& a_to_b,
                                                   const FusionOptions& options = FusionOptions(),
                                                   std::string* why = nullptr);

/// Writes `<dir>/floor_transforms.yaml`, `fusion_report.yaml` and `global_cloud.ply`.
Result<void> write_fusion_outputs(const std::string& dir, const MapData& map,
                                  const FusionResult& result);

/// Reads anchors from YAML:
///   anchors: [ {name, use_z, sigma, points: {0: [[x,y,z],...], 1: [[...]]}} ]
Result<std::vector<AnchorSet>> load_anchors(const std::string& path);

/// Floor frame pose of a keyframe as Pose4 (yaw + translation; roll and pitch dropped).
Pose4 keyframe_pose4(const KeyframeRecord& k);

}  // namespace strataslam
