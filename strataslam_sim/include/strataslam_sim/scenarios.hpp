#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "strataslam/evaluation/metrics.hpp"
#include "strataslam/fusion/fusion.hpp"
#include "strataslam/pipeline/slam_system.hpp"
#include "strataslam_sim/simulator.hpp"

namespace strataslam_sim {

/// One integration scenario (S01..S12): building, scripted trajectory, sensors, SLAM profile.
struct ScenarioSpec {
  std::string id;
  std::string description;
  std::shared_ptr<Building> building;
  std::shared_ptr<Trajectory> trajectory;
  SimConfig sim;
  strataslam::Config slam;
  strataslam::SystemTuning tuning;
  /// Optional per-scenario evaluation customisation (e.g. fusion with anchors).
  bool run_fusion = false;
};

/// Builds a scenario. Unknown ids throw std::invalid_argument.
ScenarioSpec make_scenario(const std::string& id, uint64_t seed = 42);

struct ScenarioResult {
  strataslam::MapData map;
  std::vector<strataslam::KeyframeInfo> keyframes;
  std::vector<strataslam::AcceptedLoop> loops;
  std::vector<strataslam::TransitionRecord> transitions;
  strataslam::eval::Report report;
  std::optional<strataslam::FusionResult> fusion;  ///< offline floor fusion, for multi-floor scenarios that need it
  size_t cross_floor_candidates = 0;   ///< candidates from another floor that layer 1 let through (must be 0)
  std::vector<std::string> failures;   ///< acceptance criteria that were not met
  bool passed() const { return failures.empty(); }
};

/// Feeds the simulated sensor stream into a SlamSystem, evaluates, and applies the acceptance
/// criteria of the scenario (docs/decisions.md and the development spec, section 12.4).
ScenarioResult run_scenario(const ScenarioSpec& spec);

/// Visual vocabulary for the simulator. The synthetic descriptors are random 256-bit codes, so the
/// vocabulary is trained on random descriptors with its own seed (never on the test building).
std::shared_ptr<const strataslam::BowVocabulary> sim_vocabulary();

/// Runs the offline fusion of a finished scenario map and measures it against the ground truth:
/// per-floor 4-DoF transform error and floor height error in the map frame.
struct FusedMetrics {
  std::map<strataslam::FloorId, strataslam::eval::FusionError> transform_error;
  std::map<strataslam::FloorId, double> height_error;  ///< |mean z of the floor's keyframes - truth| in the map frame
  double worst_translation = 0, worst_rotation_deg = 0, worst_height = 0;
};
FusedMetrics evaluate_fusion(const ScenarioSpec& spec, const Simulator& sim, const ScenarioResult& result,
                             const strataslam::FusionResult& fusion);

/// Anchors as a user marks them on the estimated floor clouds (ADR-0034). Each structural corner is
/// placed where it appears in that floor's map: expressed relative to the keyframe of that floor that
/// saw it from closest (ground truth) and carried into the map by that keyframe's *estimated* pose, so
/// the map's own distortion is in the mark; then `sigma` of marking error. A corner no keyframe saw
/// within `max_view` m on some floor is not marked at all. The ground truth only tells which physical
/// corner is clicked (the user knows the building); no floor frame is fitted to it.
struct AnchorChoice {
  bool elevator = true;  ///< elevator door frame and shaft corners
  bool stairs = false;   ///< stairwell corners
  bool building = true;  ///< inner corners of the outer walls
};
std::vector<strataslam::AnchorSet> make_anchors(const Simulator& sim, const ScenarioResult& result, uint64_t seed, double sigma = 0.05,
                                              AnchorChoice choice = AnchorChoice(), double max_view = 6.0);

/// Ground-truth trajectory samples of the base (every `step` s), for evaluation.
std::vector<strataslam::eval::TimedPose> ground_truth(const Simulator& sim, double step = 0.05);

}  // namespace strataslam_sim
