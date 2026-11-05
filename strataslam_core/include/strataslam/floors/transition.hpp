#pragma once

#include <array>
#include <optional>
#include <string>

#include <Eigen/Core>

#include "strataslam/common/types.hpp"

namespace strataslam {

enum class TransitionType { Stairs, Ramp, Elevator, Aerial, Unknown };

const char* to_string(TransitionType type);
/// Returns false if `name` is not a known transition type.
bool parse_transition_type(const std::string& name, TransitionType* out);

struct TransitionRecord {
  TransitionType type = TransitionType::Unknown;
  FloorId from = 0;
  FloorId to = 0;
  KeyframeId kf_before = 0;
  KeyframeId kf_after = 0;
  std::optional<double> delta_z;  ///< m, if measurable
  Eigen::Matrix4d covariance = Eigen::Matrix4d::Zero();  ///< tangent space of Pose4
  bool continuous_tracking = false;
  /// Pose4 (yaw, x, y, z) of the base at `kf_after` relative to the base at `kf_before`, when the
  /// odometry was continuous (z comes from the barometer for elevators).
  std::optional<std::array<double, 4>> relative;

  bool operator==(const TransitionRecord& o) const {
    return type == o.type && from == o.from && to == o.to && kf_before == o.kf_before &&
           kf_after == o.kf_after && delta_z == o.delta_z && covariance == o.covariance &&
           continuous_tracking == o.continuous_tracking && relative == o.relative;
  }
};

}  // namespace strataslam
