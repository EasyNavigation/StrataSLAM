#pragma once

#include <Eigen/Core>
#include <vector>

#include "strataslam/geometry/pose4.hpp"

namespace strataslam {

struct PointPair {
  Eigen::Vector3d src;  ///< point in the source frame
  Eigen::Vector3d dst;  ///< the same physical point in the destination frame
};

/// Closed-form Pose4 (yaw about z + translation) minimizing sum w_i |X src_i - dst_i|^2: the
/// Umeyama alignment restricted to rotations about z. With `use_z = false` only x and y enter the
/// fit (the translation along z is then the mean height difference, flagged by the caller).
Pose4 align_pose4_closed_form(const std::vector<PointPair>& pairs,
                              const std::vector<double>* weights = nullptr, bool use_z = true);

struct RobustAlignment {
  Pose4 transform;
  std::vector<bool> inlier;
  size_t n_inliers = 0;
  bool converged = false;
};

/// Robust alignment with graduated non-convexity and a truncated least-squares loss (Yang et al.,
/// "Graduated Non-Convexity for Robust Spatial Perception", RA-L 2020, the estimator behind
/// TEASER's robust rotation/translation steps). `noise_bound` is the inlier residual bound (m).
RobustAlignment align_pose4_gnc(const std::vector<PointPair>& pairs, double noise_bound,
                                bool use_z = true);

}  // namespace strataslam
