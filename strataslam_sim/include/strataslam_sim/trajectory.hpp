#pragma once

#include <functional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>

#include "strataslam_sim/building.hpp"

namespace strataslam_sim {

struct TrajectorySample {
  double t = 0;                     ///< s
  Eigen::Vector3d p = Eigen::Vector3d::Zero();            ///< base position, world
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();        ///< base orientation (base -> world)
  double yaw = 0;                   ///< unwrapped heading (rad)
  Eigen::Vector3d v = Eigen::Vector3d::Zero(), a = Eigen::Vector3d::Zero();  ///< world velocity / acceleration
  Eigen::Vector3d omega_body = Eigen::Vector3d::Zero();   ///< angular velocity in the base frame
  int true_floor = 0;               ///< -1 while changing floors
  double cabin_z = 0;               ///< elevator cabin height (level-0 floor = 0)
  bool door_closed = false;
};

/// Densely sampled ground-truth trajectory (200 Hz by default) with floor labels and cabin state.
class Trajectory {
 public:
  Trajectory(std::vector<TrajectorySample> samples, double dt) : samples_(std::move(samples)), dt_(dt) {}
  const std::vector<TrajectorySample>& samples() const { return samples_; }
  double dt() const { return dt_; }
  double duration() const { return samples_.empty() ? 0.0 : samples_.back().t; }

  /// Interpolated pose (lerp / slerp); `t` is clamped to [0, duration].
  gtsam::Pose3 pose_at(double t) const;
  /// Nearest-sample state (floor label, cabin, door) with interpolated cabin height.
  TrajectorySample state_at(double t) const;
  /// TUM format lines "t x y z qx qy qz qw".
  /// TUM text of the trajectory; stamps are `t + t0_s` (pass the simulator's start stamp to get the time base of its
  /// sensor messages).
  std::string to_tum(double t0_s = 0.0) const;

 private:
  std::vector<TrajectorySample> samples_;
  double dt_;
};

struct TrajectoryConfig {
  double dt = 0.005;
  double walk_speed = 1.0;        ///< m/s
  double walk_accel = 0.8;        ///< m/s^2 when starting/stopping
  double turn_rate = 0.8;         ///< rad/s for turns in place
  double sensor_height = 1.2;     ///< base height above the walking surface (handheld)
  double wobble_deg = 0.4;        ///< hand-held roll/pitch oscillation amplitude
  double elevator_speed = 1.0;    ///< m/s
  double elevator_accel = 0.5;    ///< m/s^2
  double fillet = 0.8;            ///< corner radius when walking polylines
  uint64_t seed = 42;
};

/// Scripted trajectory generator. All primitives start and end at rest.
class TrajectoryBuilder {
 public:
  TrajectoryBuilder(const Building& building, const TrajectoryConfig& config, int start_level,
                    const Eigen::Vector2d& start_xy, double start_yaw);

  /// Walk along a polyline (rounded corners) on the current level; turns in place first if needed.
  TrajectoryBuilder& walk(const std::vector<Eigen::Vector2d>& points);
  /// Walk along a polyline keeping a fixed heading (e.g. backing into an elevator).
  TrajectoryBuilder& walk_facing(const std::vector<Eigen::Vector2d>& points, double yaw);
  TrajectoryBuilder& turn_in_place(double target_yaw);
  TrajectoryBuilder& hold(double seconds);
  /// Climb the U-shaped stairs from `level` up (or down from level+1); start next to the annex door.
  TrajectoryBuilder& stairs(int level, bool up);
  TrajectoryBuilder& ramp(int level, bool up);
  /// Ride the elevator with the robot standing at rest inside the cabin.
  TrajectoryBuilder& elevator(int from_level, int to_level);
  /// Vertical flight to `target_level` (plus `agl` meters) at the current xy, rotating at `yaw_rate`.
  TrajectoryBuilder& fly_to_level(int target_level, double agl, double yaw_rate = 0.3);
  /// Declare that the cabin currently waits at `level` (before the first ride).
  TrajectoryBuilder& set_cabin_level(int level);

  Eigen::Vector2d xy() const { return {x_, y_}; }
  double yaw() const { return yaw_; }
  int level() const { return level_; }

  Trajectory build() const;

 private:
  struct PathPoint { Eigen::Vector2d p; double yaw; };
  void emit(const Eigen::Vector3d& p, double yaw, int label, double cabin_z, bool door_closed);
  void walk_path(const std::vector<Eigen::Vector2d>& points, const std::function<double(double, double)>& surface,
                 const std::function<int(double, double, double)>& label);
  Eigen::Matrix3d attitude(double yaw, double t, double wobble_scale) const;

  const Building& building_;
  TrajectoryConfig cfg_;
  int level_;
  double x_, y_, yaw_;
  double surface_z_;      ///< walking-surface height at the current pose
  double t_ = 0;
  int cabin_level_ = 0;
  double cabin_z_;
  double wobble_scale_ = 1.0;
  double agl_ = 0;
  bool aerial_ = false;
  std::vector<TrajectorySample> samples_;
};

}  // namespace strataslam_sim
