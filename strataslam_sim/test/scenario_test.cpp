#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "strataslam_sim/scenarios.hpp"

using namespace strataslam_sim;

namespace {

/// Runs a scenario, prints and (if STRATASLAM_REPORT_DIR is set) saves its Markdown report.
void check(const std::string& id) {
  const ScenarioSpec spec = make_scenario(id);
  const ScenarioResult r = run_scenario(spec);
  std::string md = r.report.to_markdown();
  for (const auto& f : r.failures) md += "\n**FAILED:** " + f + "\n";
  std::cout << md << std::endl;
  if (const char* dir = std::getenv("STRATASLAM_REPORT_DIR")) std::ofstream(std::string(dir) + "/" + id + ".md") << md;
  for (const auto& f : r.failures) ADD_FAILURE() << id << ": " << f;
}

}  // namespace

TEST(Scenario, S01_OneFloorCameraLoop) { check("S01"); }
TEST(Scenario, S02_OneFloorLidarLoop) { check("S02"); }
TEST(Scenario, S03_IdenticalFloorsStairs) { check("S03"); }
TEST(Scenario, S04_NegativeControlCollapsesWithoutGate) { check("S04"); }
TEST(Scenario, S05_ElevatorBarometer) { check("S05"); }
TEST(Scenario, S06_ElevatorImu) { check("S06"); }
TEST(Scenario, S07_ElevatorWithoutSensorsAnchorsFusion) { check("S07"); }
TEST(Scenario, S08_Ramp) { check("S08"); }
TEST(Scenario, S09_AtriumFlight) { check("S09"); }
TEST(Scenario, S10_CombinedCameraLidar) { check("S10"); }
TEST(Scenario, S11_FullPipelineThreeFloors) { check("S11"); }
TEST(Scenario, S12_CameraBlackout) { check("S12"); }
TEST(Scenario, S13_GentleRamp) { check("S13"); }
TEST(Scenario, S14_CameraBlackoutWithWheelOdometry) { check("S14"); }

TEST(Scenario, S02_MapSurvivesDiskRoundTrip) {
  const ScenarioSpec spec = make_scenario("S02");
  const ScenarioResult r = run_scenario(spec);
  const std::string dir = ::testing::TempDir() + "/strataslam_s02_map";
  ASSERT_TRUE(strataslam::write_map(dir, r.map).ok());
  const auto back = strataslam::read_map(dir);
  ASSERT_TRUE(back.ok()) << back.error().message;
  EXPECT_TRUE(back.value() == r.map);
  ASSERT_EQ(r.map.floors.size(), 1u);
  EXPECT_GT(r.map.floors[0].floor_cloud->points.size(), 5000u);
  EXPECT_FALSE(r.map.floors[0].lidar.empty());
}

namespace {
ScenarioResult run_with_threads(int threads) {
  ScenarioSpec spec = make_scenario("S02");
  spec.slam.runtime.threads = threads;
  return run_scenario(spec);
}
double max_translation_diff(const ScenarioResult& a, const ScenarioResult& b) {
  EXPECT_EQ(a.keyframes.size(), b.keyframes.size());
  double worst = 0;
  for (size_t i = 0; i < std::min(a.keyframes.size(), b.keyframes.size()); ++i)
    worst = std::max(worst, (a.keyframes[i].T_odom_base.translation() - b.keyframes[i].T_odom_base.translation()).norm());
  return worst;
}
}  // namespace

// Spec 12.9: same seed + single thread is bit-identical; multi-thread agrees to < 1e-6 m. The LiDAR registration
// groups its sums in fixed chunks whatever the number of threads (ADR-0048), so every thread count gives the same bits.
TEST(Scenario, DeterministicSingleThread) {
  const ScenarioResult a = run_with_threads(1), b = run_with_threads(1);
  EXPECT_EQ(max_translation_diff(a, b), 0.0);
  EXPECT_EQ(a.loops.size(), b.loops.size());
}

TEST(Scenario, MultiThreadIsIdenticalToSingleThread) {
  const ScenarioResult a = run_with_threads(1);
  for (int threads : {4, 7}) {
    const ScenarioResult b = run_with_threads(threads);
    EXPECT_EQ(max_translation_diff(a, b), 0.0) << threads << " threads";
    EXPECT_EQ(a.loops.size(), b.loops.size()) << threads << " threads";
    EXPECT_EQ(a.map.floors.size(), b.map.floors.size()) << threads << " threads";
  }
}

// `multifloor.enabled: false` keeps a single level: the ramp of S08 is then terrain.
TEST(Scenario, MultifloorDisabledKeepsOneFloor) {
  ScenarioSpec spec = make_scenario("S08");
  spec.slam.multifloor.enabled = false;
  const ScenarioResult r = run_scenario(spec);
  EXPECT_EQ(r.map.floors.size(), 1u);
  EXPECT_TRUE(r.transitions.empty());
}

TEST(Scenario, MultifloorDisabledKeepsOneFloorThroughAnUnrelocalizedLoss) {
  // S12's camera blackout without a vocabulary: the new session cannot be relocalized. With one level it is a
  // session of that level, not an unobservable floor change.
  ScenarioSpec spec = make_scenario("S12");
  spec.slam.multifloor.enabled = false;
  spec.tuning.vocabulary.reset();
  const ScenarioResult r = run_scenario(spec);
  EXPECT_EQ(r.map.floors.size(), 1u);
  EXPECT_TRUE(r.transitions.empty());
  ASSERT_FALSE(r.map.floors.empty());
  EXPECT_GE(r.map.floors[0].metadata.sessions.size(), 2u);  // the loss still opened a session
}

TEST(Scenario, PlanarKeepsTheBaseOnOnePlane) {
  // S02 (one floor, LiDAR) on a ground robot in planar mode: one floor, every keyframe at the plane's height and
  // level, and the scenario's own criteria (ATE, loops) still met.
  ScenarioSpec spec = make_scenario("S02");
  spec.slam.robot.platform = strataslam::Platform::Ground;
  spec.slam.multifloor.mode = strataslam::LevelMode::Planar;
  spec.slam.multifloor.enabled = false;
  const ScenarioResult r = run_scenario(spec);
  for (const auto& f : r.failures) ADD_FAILURE() << f;
  ASSERT_EQ(r.map.floors.size(), 1u);
  double worst_z = 0, worst_tilt_deg = 0;
  for (const auto& k : r.map.floors[0].keyframes) {
    worst_z = std::max(worst_z, std::abs(k.t.z()));
    const Eigen::Vector3d up = k.q.toRotationMatrix().transpose() * Eigen::Vector3d::UnitZ();
    worst_tilt_deg = std::max(worst_tilt_deg, std::acos(std::clamp(up.z(), -1.0, 1.0)) * 180.0 / M_PI);
  }
  EXPECT_LT(worst_z, 0.03);
  EXPECT_LT(worst_tilt_deg, 1.0);
}

// Layer 1 of the floor gate on the LiDAR path: S03's two identical floors walked with a 32-beam LiDAR instead of the
// camera, so that Scan Context, not BoW, finds the loops. A single Scan Context database for both floors would hand the
// loop search candidates of the other floor; none may come up, and the map keeps its two floors (audit AUD-034).
TEST(Scenario, LidarLoopCandidatesNeverComeFromAnotherFloor) {
  ScenarioSpec spec = make_scenario("S03");
  spec.sim.camera = false;
  spec.sim.lidar = true;
  spec.sim.lidar_beams = 32;  // a 16-beam sensor loses height in the stairwell (ADR-0015)
  spec.slam.sensors.camera.enabled = false;
  spec.slam.sensors.lidar.enabled = true;
  const ScenarioResult r = run_scenario(spec);
  EXPECT_EQ(r.cross_floor_candidates, 0u);
  EXPECT_EQ(r.map.floors.size(), 2u);
  EXPECT_GT(r.loops.size(), 0u);
  for (const auto& l : r.loops) EXPECT_EQ(l.from_floor, l.to_floor);
}
