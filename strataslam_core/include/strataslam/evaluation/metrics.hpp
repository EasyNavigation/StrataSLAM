#pragma once

#include <gtsam/geometry/Pose3.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "strataslam/common/types.hpp"
#include "strataslam/geometry/pose4.hpp"

namespace strataslam::eval {

/// A stamp in nanoseconds as TUM seconds with all nine decimals ("1305031102.175304000"): written
/// with 12 significant digits a 10-digit epoch kept two decimals, and evo matched 10 ms off (+4.5 %
/// of ATE on TUM fr1; audit AUD-049).
std::string tum_stamp(int64_t stamp_ns);

struct TimedPose {
  Timestamp stamp = 0;
  gtsam::Pose3 pose;
};

/// Linear/slerp interpolation in a sorted trajectory; false if `stamp` is outside it.
bool interpolate_pose(const std::vector<TimedPose>& traj, Timestamp stamp, gtsam::Pose3* out);

/// Rigid alignment minimizing sum |T p_i - q_i|^2 (Umeyama without scale).
gtsam::Pose3 align_se3(const std::vector<Eigen::Vector3d>& p,
                       const std::vector<Eigen::Vector3d>& q);
/// Same with the transform restricted to Pose4 (yaw about z + 3D translation).
Pose4 align_4dof(const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Vector3d>& q);

enum class AlignMode { None, SE3, FourDof };

struct AteResult {
  double rmse = 0, mean = 0, max = 0;
  size_t n = 0;
};

/// Absolute trajectory error of `est` against `gt` (gt is interpolated at the estimate stamps).
AteResult ate(const std::vector<TimedPose>& est, const std::vector<TimedPose>& gt, AlignMode mode);

struct RpeResult {
  double trans_rmse = 0;  ///< m
  double rot_rmse_deg = 0;
  size_t n = 0;
};
/// Relative pose error over a fixed time interval `delta_ns` (no global alignment needed).
RpeResult rpe(const std::vector<TimedPose>& est, const std::vector<TimedPose>& gt,
              int64_t delta_ns);

struct LoopRecord {
  KeyframeId from = 0, to = 0;
  FloorId from_floor = -1, to_floor = -1;  ///< ground-truth floors of the two keyframes
};
/// Accepted loop closures between keyframes whose true floor differs / accepted loops total.
/// Loops with an undefined (-1) true floor on either side count as cross-floor. 0 if none.
double collapse_rate(const std::vector<LoopRecord>& accepted);

/// |mean z of the keyframes of each floor - true level height| (both in the map frame).
std::map<FloorId, double> floor_height_errors(const std::map<FloorId, std::vector<double>>& est_z,
                                              const std::map<FloorId, double>& true_z);

struct FusionError {
  double trans_m = 0;
  double rot_deg = 0;
};
/// Difference between estimated and true floor transforms (Pose4 each), per floor.
std::map<FloorId, FusionError> fusion_errors(const std::map<FloorId, Pose4>& est,
                                             const std::map<FloorId, Pose4>& truth);

/// Named scalar metrics rendered for reports.
struct Report {
  std::string title;
  std::vector<std::pair<std::string, double>> values;
  std::string to_markdown() const;
  std::string to_json() const;
};

}  // namespace strataslam::eval
