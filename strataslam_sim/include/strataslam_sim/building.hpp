#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "strataslam/sensors/measurements.hpp"

namespace strataslam_sim {

/// Oriented box. `R` maps box-local axes to world.
struct Box {
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d half = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  bool dynamic = false;  ///< belongs to the elevator cabin (moves with it)
  bool door = false;     ///< cabin door leaf: only present while the door is closed
  int level = 0;
};

/// Slab-method ray/box intersection. Returns true and the entry distance `t` in (t_min, t_max).
bool ray_box(const Eigen::Vector3d& origin, const Eigen::Vector3d& dir, const Box& box, double t_min,
             double t_max, double* t);
bool point_in_box(const Eigen::Vector3d& p, const Box& box, double margin = 0.0);

struct Landmark {
  Eigen::Vector3d position;  ///< world position with the cabin at height 0 if `dynamic`
  strataslam::Descriptor descriptor;
  bool dynamic = false;
  bool door = false;     ///< visible only while the cabin door is closed
  int level = 0;
  float max_dist = 0;    ///< fine texture: only observable closer than this (0 = always)
};

struct BuildingConfig {
  int num_floors = 2;
  double floor_height = 3.0;       ///< m, floor top to floor top
  bool identical_floors = true;    ///< same geometry and texture on every floor (aliasing)
  bool with_stairs = true;
  bool with_elevator = true;
  bool with_ramp = false;
  double ramp_length = 7.5;        ///< m, horizontal length of each ramp flight (1.5 m rise: 7.5 -> 11.3 deg)
  bool with_atrium = false;
  double landmark_density = 6.0;   ///< coarse landmarks per m^2 (observable at any range)
  double fine_density = 25.0;      ///< fine-texture landmarks per m^2, observable within `fine_range`
  double fine_range = 3.0;         ///< m
  uint64_t seed = 42;
};

/// Anchor points (structural features seen from every floor), world coordinates at level 0.
struct Anchors {
  std::vector<Eigen::Vector3d> elevator_door_corners;
  std::vector<Eigen::Vector3d> elevator_shaft_corners;  ///< inner corners of the shaft, floor and head height
  std::vector<Eigen::Vector3d> stair_shaft_corners;     ///< inner corners of the stairwell, floor and head height
  std::vector<Eigen::Vector3d> building_corners;        ///< inner corners of the outer walls, floor and head height
};

/// Procedural multi-floor building: footprint 32x22 m with a ring corridor, plus annexes on the
/// south (stairs, elevator) and east (ramp) sides. Layout details in building.cpp.
class Building {
 public:
  explicit Building(const BuildingConfig& config);

  const BuildingConfig& config() const { return config_; }
  const std::vector<Box>& boxes() const { return boxes_; }
  const std::vector<Landmark>& landmarks() const { return landmarks_; }
  const Anchors& anchors() const { return anchors_; }

  double ramp_x1() const { return 32.4 + config_.ramp_length; }          ///< end of the flights
  double ramp_land_x1() const { return ramp_x1() + 3.1; }                  ///< end of the landing
  double ramp_slope() const { return kStairRise / config_.ramp_length; }
  double level_height(int level) const { return level * config_.floor_height; }
  /// Walking surface height of the free path at (x, y) on the stairs/ramp annexes; nullopt elsewhere.
  std::optional<double> stairs_surface(int level, double x, double y) const;
  std::optional<double> ramp_surface(int level, double x, double y) const;

  /// First hit distance along a unit-length ray, honoring cabin height and door state.
  std::optional<double> raycast(const Eigen::Vector3d& origin, const Eigen::Vector3d& dir,
                                double max_range, double cabin_z, bool door_closed) const;
  /// Occlusion test: is `target` visible from `origin`?
  bool visible(const Eigen::Vector3d& origin, const Eigen::Vector3d& target, double cabin_z,
               bool door_closed) const;

  Eigen::Vector3d landmark_position(const Landmark& lm, double cabin_z) const {
    return lm.dynamic ? Eigen::Vector3d(lm.position + Eigen::Vector3d(0, 0, cabin_z)) : lm.position;
  }

  // Layout constants shared with the trajectory builder (world frame, level 0).
  static constexpr double kSizeX = 32.0, kSizeY = 22.0, kSlab = 0.2;
  static constexpr double kStairRise = 1.5, kStairTread = 0.27;

 private:
  void build_level(int level);
  void add_box(const Eigen::Vector3d& lo, const Eigen::Vector3d& hi, int level, bool dynamic = false,
               bool door = false);
  void add_oriented(const Eigen::Vector3d& center, const Eigen::Vector3d& half,
                    const Eigen::Matrix3d& R, int level);
  void generate_landmarks();

  BuildingConfig config_;
  std::vector<Box> boxes_;
  std::vector<Landmark> landmarks_;
  Anchors anchors_;
  std::vector<uint64_t> box_key_;  ///< texture key per box (shared across floors if identical)
  std::map<size_t, size_t> roof_copy_of_;  ///< roof box -> the slab box it copies (identical floors)
};

/// Snapshot of the geometry for fast ray queries (cabin position and door state fixed), optionally
/// restricted to the boxes that can matter within `radius` of `center`.
class RayScene {
 public:
  RayScene(const Building& building, double cabin_z, bool door_closed,
           const Eigen::Vector3d& center = Eigen::Vector3d::Zero(), double radius = 1e9);
  /// Distance to the first hit along a unit ray, or nullopt.
  std::optional<double> raycast(const Eigen::Vector3d& origin, const Eigen::Vector3d& dir,
                                double max_range) const;
  bool visible(const Eigen::Vector3d& origin, const Eigen::Vector3d& target) const;
  size_t size() const { return aabbs_.size() + obbs_.size(); }

 private:
  struct Aabb { Eigen::Vector3d lo, hi; };
  std::vector<Aabb> aabbs_;
  std::vector<Box> obbs_;
};

}  // namespace strataslam_sim
