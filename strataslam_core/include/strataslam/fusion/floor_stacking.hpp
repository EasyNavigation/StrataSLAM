#pragma once

#include <Eigen/Core>
#include <optional>
#include <string>

#include "strataslam/geometry/pose4.hpp"
#include "strataslam/io/point_cloud.hpp"

namespace strataslam {

/// Floors of a building share structure along the vertical (walls, shafts, columns, the footprint
/// itself), so two floor maps can be registered against each other. This is the general way to
/// remove the drift that the odometry accumulates along a ramp or a stairwell; it assumes nothing
/// about the shape of the building and it refuses to answer when the structure does not decide the
/// alignment (floors that do not stack, repetitive layouts, too little structure).
struct StackingOptions {
  double cell = 0.4;   ///< m, raster resolution of the first, coarse search
  double z_min = 0.4;  ///< m above the floor frame: points between z_min and z_max are structure
  double z_max = 3.0;  ///< (floor, furniture and ceilings are left out)
  double search_radius = 25.0;   ///< m around the initial relative position
  double yaw_window_deg = 30.0;  ///< +- around the initial relative heading
  double yaw_step_deg = 1.5;
  int min_cells = 150;      ///< occupied cells needed on each floor
  double min_score = 0.35;  ///< fraction of the smaller structure that lies on the other one
  double peak_ratio =
      1.25;  ///< best score over the best score elsewhere: below it the answer is ambiguous
  double exclusion_m = 3.0;      ///< "elsewhere": farther than this ...
  double exclusion_deg = 6.0;    ///< ... or turned by more than this from the best pose
  bool allow_ambiguous = false;  ///< return the best pose flagged `ambiguous` instead of refusing
                                 ///< (to report candidates)
};

struct StackingMatch {
  Pose4 a_to_b;      ///< pose of floor b's frame in floor a's frame (the height is the initial one)
  double score = 0;  ///< fraction of the smaller structure explained by the other floor
  double runner_up = 0;    ///< best score at a distinct pose
  bool ambiguous = false;  ///< only with `allow_ambiguous`: another pose scores almost as well
  int cells_a = 0, cells_b = 0;
  Eigen::Matrix4d covariance =
      Eigen::Matrix4d::Identity();  ///< (yaw, x, y, z); z carries no information
};

/// Registers floor b's structure on floor a's around `initial_a_to_b` (relative pose, e.g. chained
/// through the transitions). Returns nothing, and says why in `why`, when the match is not
/// trustworthy.
std::optional<StackingMatch> match_floor_structure(
    const PointCloud& a, const PointCloud& b, const Pose4& initial_a_to_b,
    const StackingOptions& options = StackingOptions(), std::string* why = nullptr);

}  // namespace strataslam
