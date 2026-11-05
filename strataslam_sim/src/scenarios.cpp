#include "strataslam_sim/scenarios.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "strataslam_sim/rng.hpp"

namespace strataslam_sim {

using strataslam::eval::AlignMode;
using strataslam::eval::TimedPose;
using Eigen::Vector2d;

namespace {

strataslam::Config base_config() {
  strataslam::Config c;
  c.runtime.seed = 42;
  c.multifloor.enabled = true;
  c.multifloor.floor_height_prior = 3.0;
  return c;
}

std::shared_ptr<Trajectory> share(Trajectory t) { return std::make_shared<Trajectory>(std::move(t)); }

void enable_camera(ScenarioSpec& s) {
  s.sim.camera = true;
  s.slam.sensors.camera.enabled = true;
  s.slam.sensors.camera.rgb_topic = "/camera/color/image_raw";
  s.slam.sensors.camera.depth_topic = "/camera/aligned_depth_to_color/image_raw";
  s.slam.sensors.camera.info_topic = "/camera/color/camera_info";
  s.tuning.vocabulary = sim_vocabulary();
}

void enable_imu(ScenarioSpec& s) {
  s.sim.imu = true;
  s.slam.sensors.imu.enabled = true;
  s.slam.sensors.imu.topic = "/imu";
}

void enable_baro(ScenarioSpec& s) {
  s.sim.baro = true;
  s.slam.sensors.barometer.enabled = true;
  s.slam.sensors.barometer.topic = "/pressure";
}

}  // namespace

std::shared_ptr<const strataslam::BowVocabulary> sim_vocabulary() {
  static const std::shared_ptr<const strataslam::BowVocabulary> vocab = [] {
    Rng rng(0xB0A5);
    std::vector<strataslam::Descriptor> d(60000);
    for (auto& desc : d)
      for (size_t w = 0; w < 4; ++w) {
        const uint64_t v = rng.next();
        for (int b = 0; b < 8; ++b) desc[w * 8 + b] = static_cast<uint8_t>(v >> (8 * b));
      }
    strataslam::BowConfig cfg;
    cfg.branching = 10;
    cfg.levels = 4;
    return std::make_shared<const strataslam::BowVocabulary>(strataslam::BowVocabulary::train(d, cfg));
  }();
  return vocab;
}

std::vector<TimedPose> ground_truth(const Simulator& sim, double step) {
  std::vector<TimedPose> gt;
  const double T = sim.trajectory().duration();
  for (double t = 0; t <= T; t += step) gt.push_back({sim.stamp_at(t), sim.trajectory().pose_at(t)});
  return gt;
}

ScenarioSpec make_scenario(const std::string& id, uint64_t seed) {
  ScenarioSpec s;
  s.id = id;
  s.slam = base_config();
  s.sim.seed = seed;
  s.sim.camera = false;
  s.sim.lidar = false;
  s.sim.imu = false;
  TrajectoryConfig tc;
  tc.seed = seed;

  BuildingConfig bc;
  bc.seed = seed;

  if (id == "S01") {
    s.description = "One floor, closed loop, RGB-D camera";
    bc.num_floors = 1;
    bc.with_stairs = bc.with_elevator = false;
    s.building = std::make_shared<Building>(bc);
    TrajectoryBuilder tb(*s.building, tc, 0, {3, 3}, 0.0);
    tb.walk({{29, 3}, {29, 19}, {3, 19}, {3, 3}, {16, 3}});
    s.trajectory = share(tb.build());
    enable_camera(s);
  } else if (id == "S02") {
    s.description = "One floor, closed loop, LiDAR";
    bc.num_floors = 1;
    bc.with_stairs = bc.with_elevator = false;
    s.building = std::make_shared<Building>(bc);
    TrajectoryBuilder tb(*s.building, tc, 0, {3, 3}, 0.0);
    tb.walk({{29, 3}, {29, 19}, {3, 19}, {3, 3}, {16, 3}});
    s.trajectory = share(tb.build());
    s.sim.lidar = true;
    s.slam.sensors.lidar.enabled = true;
  } else if (id == "S03" || id == "S04") {
    s.description = id == "S03" ? "Two identical floors, stairs, camera, no IMU" : "S03 with the floor gate disabled (negative control)";
    bc.num_floors = 2;
    bc.identical_floors = true;
    bc.with_elevator = false;
    s.building = std::make_shared<Building>(bc);
    TrajectoryBuilder tb(*s.building, tc, 0, {3, 3}, 0.0);
    tb.walk({{29, 3}, {29, 19}, {3, 19}, {3, 3}, {11.0, 3}, {11.0, 0.8}}).stairs(0, true);
    // Same ring on floor 1 (identical geometry and texture), starting from the stair exit.
    tb.walk({{13.2, 3}, {29, 3}, {29, 19}, {3, 19}, {3, 3}, {13.2, 3}});
    s.trajectory = share(tb.build());
    enable_camera(s);
    s.tuning.backend.debug_disable_floor_gating = (id == "S04");
  } else if (id == "S08") {
    s.description = "Ramp between floors, camera";
    bc.num_floors = 2;
    bc.with_ramp = true;
    bc.with_stairs = bc.with_elevator = false;
    s.building = std::make_shared<Building>(bc);
    TrajectoryBuilder tb(*s.building, tc, 0, {3, 3}, 0.0);
    tb.walk({{29, 3}, {29, 9.1}}).ramp(0, true).walk({{29, 12}, {29, 19}, {3, 19}, {3, 12}});
    s.trajectory = share(tb.build());
    enable_camera(s);
  } else if (id == "S13") {
    s.description = "Gentle ramp (2.9 degrees, 30 m flights) between floors, camera";
    bc.num_floors = 2;
    bc.with_ramp = true;
    bc.ramp_length = 30.0;
    bc.with_stairs = bc.with_elevator = false;
    s.building = std::make_shared<Building>(bc);
    TrajectoryBuilder tb(*s.building, tc, 0, {3, 3}, 0.0);
    tb.walk({{29, 3}, {29, 9.1}}).ramp(0, true).walk({{29, 12}, {29, 19}, {3, 19}, {3, 12}});
    s.trajectory = share(tb.build());
    enable_camera(s);
  } else if (id == "S10") {
    s.description = "One floor, closed loop, camera + LiDAR combined";
    bc.num_floors = 1;
    bc.with_stairs = bc.with_elevator = false;
    s.building = std::make_shared<Building>(bc);
    TrajectoryBuilder tb(*s.building, tc, 0, {3, 3}, 0.0);
    tb.walk({{29, 3}, {29, 19}, {3, 19}, {3, 3}, {16, 3}});
    s.trajectory = share(tb.build());
    enable_camera(s);
    s.sim.lidar = true;
    s.slam.sensors.lidar.enabled = true;
  } else if (id == "S07") {
    s.description = "Elevator without IMU or barometer, camera; floors differ in appearance; anchors + floor height prior";
    bc.num_floors = 2;
    bc.identical_floors = false;
    bc.with_stairs = false;
    s.building = std::make_shared<Building>(bc);
    TrajectoryBuilder tb(*s.building, tc, 0, {3, 3}, 0.0);
    tb.walk({{29, 3}, {29, 19}, {3, 19}, {3, 3}, {21.6, 3}});
    tb.walk_facing({{21.6, 0.5}, {21.6, -1.1}}, M_PI / 2).elevator(0, 1);
    tb.walk({{21.6, 0.5}, {21.6, 3}, {29, 3}, {29, 19}, {3, 19}, {3, 3}, {21.6, 3}});
    s.trajectory = share(tb.build());
    enable_camera(s);
    s.slam.multifloor.tracking_loss_timeout = 1.5;
    s.run_fusion = true;
  } else if (id == "S11") {
    s.description = "Three identical floors, stairs then elevator, camera + LiDAR + IMU, full pipeline with fusion";
    bc.num_floors = 3;
    bc.identical_floors = true;
    s.building = std::make_shared<Building>(bc);
    TrajectoryBuilder tb(*s.building, tc, 0, {3, 3}, 0.0);
    tb.walk({{29, 3}, {29, 19}, {3, 19}, {3, 3}, {11.0, 3}, {11.0, 0.8}}).stairs(0, true);
    tb.walk({{13.2, 3}, {29, 3}, {29, 19}, {3, 19}, {3, 3}, {21.6, 3}});
    tb.set_cabin_level(1);
    tb.walk_facing({{21.6, 0.5}, {21.6, -1.1}}, M_PI / 2).elevator(1, 2);
    tb.walk({{21.6, 0.5}, {21.6, 3}, {29, 3}, {29, 19}, {3, 19}, {3, 3}, {21.6, 3}});
    s.trajectory = share(tb.build());
    enable_camera(s);
    enable_imu(s);
    s.sim.lidar = true;
    s.sim.lidar_beams = 32;  // a 16-beam sensor loses height in the stairwell (ADR-0015)
    s.slam.sensors.lidar.enabled = true;
    s.run_fusion = true;
  } else if (id == "S12" || id == "S14") {
    s.description = id == "S12" ? "One floor, 1 s camera blackout mid-floor"
                                : "S12 with wheel odometry: the blackout is carried through, no new session";
    bc.num_floors = 1;
    bc.with_stairs = bc.with_elevator = false;
    s.building = std::make_shared<Building>(bc);
    TrajectoryBuilder tb(*s.building, tc, 0, {3, 3}, 0.0);
    tb.walk({{29, 3}, {29, 19}, {3, 19}, {3, 3}, {16, 3}});
    s.trajectory = share(tb.build());
    enable_camera(s);
    s.sim.camera_blackouts = {{30.0, 31.0}};
    s.slam.multifloor.tracking_loss_timeout = 0.5;
    if (id == "S14") {
      s.sim.external_odom = true;
      s.slam.sensors.external_odometry.enabled = true;
      s.slam.sensors.external_odometry.topic = "/odom";
    }
  } else if (id == "S05" || id == "S06") {
    s.description = id == "S05" ? "Two identical floors, elevator, camera + barometer" : "Two identical floors, elevator, camera + IMU";
    bc.num_floors = 2;
    bc.identical_floors = true;
    bc.with_stairs = false;
    s.building = std::make_shared<Building>(bc);
    TrajectoryBuilder tb(*s.building, tc, 0, {3, 3}, 0.0);
    // The robot backs into the cabin (facing the lobby) and walks out forwards: no turn inside the
    // cabin, where the door opening replaces the whole view anyway.
    tb.walk({{29, 3}, {29, 19}, {3, 19}, {3, 3}, {21.6, 3}});
    tb.walk_facing({{21.6, 0.5}, {21.6, -1.1}}, M_PI / 2).elevator(0, 1);
    tb.walk({{21.6, 0.5}, {21.6, 3}, {29, 3}, {29, 19}, {3, 19}, {3, 3}, {21.6, 3}});
    s.trajectory = share(tb.build());
    enable_camera(s);
    if (id == "S05") enable_baro(s); else enable_imu(s);
  } else if (id == "S09") {
    s.description = "Vertical flight through the atrium, camera + IMU, aerial platform";
    bc.num_floors = 3;
    bc.with_atrium = true;
    bc.with_stairs = bc.with_elevator = false;
    s.building = std::make_shared<Building>(bc);
    tc.wobble_deg = 0.3;
    TrajectoryBuilder tb(*s.building, tc, 0, {16, 11}, 0.0);
    tb.fly_to_level(0, 1.5).hold(3.0).fly_to_level(2, 1.5, 0.25).hold(6.0);
    s.trajectory = share(tb.build());
    enable_camera(s);
    enable_imu(s);
    s.slam.robot.platform = strataslam::Platform::Aerial;
    s.slam.multifloor.floor_height_prior = 3.0;
  } else {
    throw std::invalid_argument("unknown scenario " + id);
  }
  return s;
}

namespace {
class StderrLogger final : public strataslam::Logger {
 public:
  void log(strataslam::LogLevel level, const std::string& m) override {
    static const char* names[] = {"debug", "info", "warn", "error"};
    std::cerr << "[" << names[static_cast<int>(level)] << "] " << m << std::endl;
  }
};
}  // namespace

namespace {
/// Physical level a floor map was built on: the true level most of its keyframes were on.
std::optional<int> level_of_floor(const strataslam::FloorData& f, const Simulator& sim) {
  std::map<int, int> votes;
  for (const auto& k : f.keyframes) {
    const int t = sim.true_floor(k.stamp_ns);
    if (t >= 0) ++votes[t];
  }
  std::optional<int> level;
  int best = -1;
  for (const auto& [lv, n] : votes) if (n > best) { best = n; level = lv; }
  return level;
}

/// Pose4 placing a floor frame in the building: 4-DoF alignment of the floor's estimated keyframes
/// to the ground-truth positions (independent of the odometry error at the keyframe that defines the
/// frame's origin). Only the keyframes that were really on the floor's level enter (ADR-0034): a
/// keyframe recorded in an elevator cabin or on a stair is a labelling error, which floor_accuracy
/// and the collapse rate measure; in the reference placement it would drag the height (an
/// unobservable elevator ride leaves keyframes metres away from the level in the arrival floor).
strataslam::Pose4 floor_frame_in_building(const strataslam::FloorData& f, const Simulator& sim) {
  const auto level = level_of_floor(f, sim);
  std::vector<Eigen::Vector3d> p, q;
  for (const auto& k : f.keyframes) {
    if (level && sim.true_floor(k.stamp_ns) != *level) continue;
    p.push_back(k.t);
    q.push_back(sim.gt_pose(k.stamp_ns).translation());
  }
  if (p.size() < 3) {  // too few on the level: fall back to every keyframe
    p.clear();
    q.clear();
    for (const auto& k : f.keyframes) {
      p.push_back(k.t);
      q.push_back(sim.gt_pose(k.stamp_ns).translation());
    }
  }
  return strataslam::eval::align_4dof(p, q);
}
}  // namespace

std::vector<strataslam::AnchorSet> make_anchors(const Simulator& sim, const ScenarioResult& r, uint64_t seed, double sigma, AnchorChoice choice,
                                              double max_view) {
  const double H = sim.building().config().floor_height;
  // The physical level of each floor map: the user knows on which floor of the building they mapped.
  std::map<strataslam::FloorId, int> level_of;
  for (const auto& f : r.map.floors)
    if (const auto lv = level_of_floor(f, sim)) level_of[f.metadata.id] = *lv;
  // Where corner `c` (level-0 coordinates) appears in floor f's map, or nothing if no keyframe of f saw it.
  const auto mark_in = [&](const strataslam::FloorData& f, const Eigen::Vector3d& c) -> std::optional<Eigen::Vector3d> {
    if (!level_of.count(f.metadata.id)) return std::nullopt;
    const int level = level_of.at(f.metadata.id);
    const Eigen::Vector3d world = c + Eigen::Vector3d(0, 0, level * H);
    const strataslam::KeyframeRecord* best = nullptr;
    double best_d = max_view;
    for (const auto& k : f.keyframes) {
      if (sim.true_floor(k.stamp_ns) != level) continue;
      const double d = (sim.gt_pose(k.stamp_ns).translation() - world).head<2>().norm();
      if (d < best_d) { best_d = d; best = &k; }
    }
    if (!best) return std::nullopt;
    const gtsam::Pose3 T_est(gtsam::Rot3(best->q), best->t);
    return T_est.transformFrom(sim.gt_pose(best->stamp_ns).transformTo(world));
  };
  std::vector<strataslam::AnchorSet> sets;
  const auto build = [&](const std::string& name, const std::vector<Eigen::Vector3d>& corners, uint64_t salt) {
    strataslam::AnchorSet set;
    set.name = name;
    set.use_z = false;
    set.sigma = sigma;
    Rng rng(seed ^ salt);
    for (const auto& c : corners) {
      std::map<strataslam::FloorId, Eigen::Vector3d> marks;
      for (const auto& f : r.map.floors)
        if (const auto m = mark_in(f, c)) marks[f.metadata.id] = *m;
      if (marks.size() != r.map.floors.size()) continue;  // a correspondence needs the corner on every floor
      for (const auto& [fid, m] : marks) set.points[fid].push_back(m + sigma * Eigen::Vector3d(rng.normal(), rng.normal(), rng.normal()));
    }
    if (!set.points.empty()) sets.push_back(set);
  };
  const auto& a = sim.building().anchors();
  if (choice.elevator) {
    std::vector<Eigen::Vector3d> corners = a.elevator_door_corners;
    corners.insert(corners.end(), a.elevator_shaft_corners.begin(), a.elevator_shaft_corners.end());
    build("elevator_shaft", corners, 0xA7C4);
  }
  if (choice.stairs) build("stair_shaft", a.stair_shaft_corners, 0x57A1);
  if (choice.building) build("building_corners", a.building_corners, 0xB01D);
  return sets;
}

FusedMetrics evaluate_fusion(const ScenarioSpec& spec, const Simulator& sim, const ScenarioResult& r, const strataslam::FusionResult& fu) {
  FusedMetrics m;
  (void)spec;
  if (r.map.floors.empty()) return m;
  strataslam::FloorId root = r.map.floors.front().metadata.id;
  for (const auto& f : r.map.floors) root = std::min(root, f.metadata.id);
  std::map<strataslam::FloorId, strataslam::Pose4> truth;
  strataslam::Pose4 root_in_bldg;
  std::map<strataslam::FloorId, strataslam::Pose4> in_bldg;
  for (const auto& f : r.map.floors) in_bldg[f.metadata.id] = floor_frame_in_building(f, sim);
  root_in_bldg = in_bldg.at(root);
  for (const auto& [id, X] : in_bldg) truth[id] = root_in_bldg.inverse() * X;
  m.transform_error = strataslam::eval::fusion_errors(fu.floor_to_map, truth);
  for (const auto& [id, e] : m.transform_error) {
    if (id == root) continue;
    m.worst_translation = std::max(m.worst_translation, e.trans_m);
    m.worst_rotation_deg = std::max(m.worst_rotation_deg, e.rot_deg);
  }
  // Height: z of every floor's keyframes in the map frame against the ground truth (relative to the
  // ground-truth height of the root floor's first keyframe).
  double z0 = 0;
  for (const auto& f : r.map.floors) {
    if (f.metadata.id != root || f.keyframes.empty()) continue;
    z0 = sim.gt_pose(f.keyframes.front().stamp_ns).z();
  }
  for (const auto& f : r.map.floors) {
    const auto it = fu.floor_to_map.find(f.metadata.id);
    if (it == fu.floor_to_map.end() || f.keyframes.empty()) continue;
    const auto level = level_of_floor(f, sim);  // same keyframe selection as floor_frame_in_building
    double est = 0, tru = 0;
    size_t n = 0;
    for (const auto& k : f.keyframes) {
      if (level && sim.true_floor(k.stamp_ns) != *level) continue;
      est += it->second.act(k.t).z();
      tru += sim.gt_pose(k.stamp_ns).z() - z0;
      ++n;
    }
    if (n == 0) continue;
    const double e = std::abs(est - tru) / n;
    m.height_error[f.metadata.id] = e;
    m.worst_height = std::max(m.worst_height, e);
  }
  return m;
}

namespace {

/// Minimum fraction of the keyframes taken on a floor that must end in the map for the ATE to count.
constexpr double kMinMapCoverage = 0.90;

/// A named value of a scenario report; throws if the name is missing (never read values by position).
double report_value(const strataslam::eval::Report& report, const std::string& name) {
  for (const auto& [key, value] : report.values)
    if (key == name) return value;
  throw std::logic_error("scenario report has no value '" + name + "'");
}

/// Shared measurements of a finished run, used by the acceptance criteria and reported.
struct Measures {
  double collapse = 0, pooled_ate = 0, worst_floor_ate = 0, worst_height_err = 0;
  double floor_accuracy = 1, floor_accuracy_all = 1;
  double map_coverage = 1;  ///< keyframes on a floor (not mid-climb) that ended in the map / all such keyframes
  size_t inter_session_loops = 0;
};

/// Feeds the simulated sensor stream, in time order, into a SlamSystem.
ScenarioResult run_slam(const ScenarioSpec& spec, const Simulator& sim) {
  std::shared_ptr<strataslam::Logger> logger = std::make_shared<strataslam::NullLogger>();
  if (std::getenv("STRATASLAM_VERBOSE") || strataslam::any_debug_channel()) logger = std::make_shared<StderrLogger>();
  strataslam::SlamSystem slam(spec.slam, spec.sim.extrinsics, spec.tuning, logger);
  using T = SensorEvent::Type;
  for (const auto& ev : sim.events()) {
    switch (ev.type) {
      case T::Imu: slam.add_imu(sim.imu(ev.index)); break;
      case T::Baro: slam.add_baro(sim.baro(ev.index)); break;
      case T::Odom: slam.add_odom(sim.odom(ev.index)); break;
      case T::Rgbd: slam.add_rgbd(sim.rgbd(ev.index)); break;
      case T::Lidar: slam.add_lidar(sim.lidar(ev.index)); break;
    }
  }
  ScenarioResult r;
  // A simulated run has no profile file: profile.yaml says where its configuration comes from (audit A6-04: it was
  // empty).
  r.map = slam.finalize("# generated by the simulator: scenario " + spec.id + ", seed " + std::to_string(spec.sim.seed) +
                        "; the configuration is the scenario's (strataslam_sim/src/scenarios.cpp, make_scenario)\n");
  r.keyframes = slam.keyframes();
  r.loops = slam.loops();
  r.transitions = slam.transitions();

  r.cross_floor_candidates = slam.cross_floor_candidates_seen();
  return r;
}

Measures measure(const ScenarioSpec& spec, const Simulator& sim, const ScenarioResult& r) {
  Measures m;
  const auto gt = ground_truth(sim);
  std::map<strataslam::KeyframeId, int> true_floor_of;  // ground-truth floor of every keyframe
  std::map<strataslam::KeyframeId, strataslam::SessionId> session_of;
  for (const auto& k : r.keyframes) {
    true_floor_of[k.id] = sim.true_floor(k.stamp);
    session_of[k.id] = k.session;
  }
  std::vector<strataslam::eval::LoopRecord> loop_records;
  for (const auto& l : r.loops) {
    loop_records.push_back({l.from, l.to, true_floor_of[l.from], true_floor_of[l.to]});
    if (session_of[l.from] != session_of[l.to]) ++m.inter_session_loops;
  }
  m.collapse = strataslam::eval::collapse_rate(loop_records);

  // Per-floor ATE (each floor aligned on its own), pooled over keyframes.
  double sum_sq = 0;
  size_t n_all = 0;
  for (const auto& f : r.map.floors) {
    std::vector<TimedPose> est;
    for (const auto& k : f.keyframes) est.push_back({k.stamp_ns, gtsam::Pose3(gtsam::Rot3(k.q.toRotationMatrix()), k.t)});
    std::sort(est.begin(), est.end(), [](const TimedPose& a, const TimedPose& b) { return a.stamp < b.stamp; });
    const auto a = strataslam::eval::ate(est, gt, AlignMode::SE3);
    sum_sq += a.rmse * a.rmse * a.n;
    n_all += a.n;
    m.worst_floor_ate = std::max(m.worst_floor_ate, a.rmse);
  }
  m.pooled_ate = n_all ? std::sqrt(sum_sq / n_all) : 0.0;

  // Coverage of the ATE: the ATE only sees keyframes that ended in the map, so a run that drops most of
  // its keyframes (a transition that never closes, a session never relocalized) could report a tiny
  // ATE on what is left. Keyframes taken mid-climb have no floor and are left out on purpose (ADR-0025).
  {
    std::set<strataslam::KeyframeId> in_map;
    for (const auto& f : r.map.floors)
      for (const auto& k : f.keyframes) in_map.insert(k.id);
    size_t on_floor = 0, kept = 0;
    for (const auto& k : r.keyframes) {
      if (true_floor_of[k.id] < 0) continue;
      ++on_floor;
      kept += in_map.count(k.id);
    }
    m.map_coverage = on_floor ? double(kept) / on_floor : 1.0;
  }

  // Height of each floor frame against the true level (floor id -> level by majority of true labels).
  const double H = spec.building->config().floor_height;
  for (const auto& f : r.map.floors) {
    std::map<int, int> votes;
    for (const auto& k : f.keyframes) {
      const int tf = true_floor_of[k.id];
      if (tf >= 0) ++votes[tf];
    }
    if (votes.empty() || !f.metadata.altitude_m) continue;
    int level = 0, best = -1;
    for (const auto& [lv, n] : votes) if (n > best) { best = n; level = lv; }
    m.worst_height_err = std::max(m.worst_height_err, std::abs(*f.metadata.altitude_m - level * H));
  }

  // Estimated floor id (visit order) -> true level, by majority of the true labels of its keyframes.
  std::map<strataslam::FloorId, int> level_of_floor;
  for (const auto& f : r.map.floors) {
    std::map<int, int> votes;
    for (const auto& k : f.keyframes) {
      const int tf = true_floor_of[k.id];
      if (tf >= 0) ++votes[tf];
    }
    int level = -1, best = -1;
    for (const auto& [lv, n] : votes) if (n > best) { best = n; level = lv; }
    level_of_floor[f.metadata.id] = level;
  }
  // Fraction of keyframes with the right floor (ADR-0035). `floor_accuracy`, the acceptance metric,
  // ignores the +-2 s around each change of the true label (the manager needs a moment to notice that
  // a transition began or ended); `floor_accuracy_all` counts every keyframe and is reported too.
  std::vector<double> label_changes;
  {
    const auto& smp = spec.trajectory->samples();
    for (size_t i = 1; i < smp.size(); ++i)
      if (smp[i].true_floor != smp[i - 1].true_floor) label_changes.push_back(smp[i].t);
  }
  size_t floor_ok = 0, floor_n = 0, all_ok = 0;
  for (const auto& k : r.keyframes) {
    const int truth = true_floor_of[k.id];
    // A keyframe taken mid-climb has no true floor: it may be on either floor of the climb (ADR-0025).
    const bool ok = truth < 0 || (k.floor >= 0 && level_of_floor.count(k.floor) && level_of_floor[k.floor] == truth);
    all_ok += ok;
    const double tk = sim.time_of(k.stamp);
    bool near_change = false;
    for (double tc2 : label_changes) near_change = near_change || std::abs(tk - tc2) < 2.0;
    if (near_change) continue;
    ++floor_n;
    floor_ok += ok;
  }
  m.floor_accuracy = floor_n ? double(floor_ok) / floor_n : 1.0;
  m.floor_accuracy_all = r.keyframes.empty() ? 1.0 : double(all_ok) / r.keyframes.size();

  return m;
}

/// The acceptance criteria of every scenario (development spec 12.4; the ADRs cited where a criterion
/// was made precise). Each unmet criterion is appended to `r.failures`.
void apply_acceptance_criteria(const ScenarioSpec& spec, const Simulator& sim, const Measures& m, ScenarioResult& r) {
  const auto fail = [&](const std::string& msg) { r.failures.push_back(msg); };
  // Layer 1 of the floor gate in every scenario, not only S03: the place databases (BoW and Scan Context) are per
  // floor, so no candidate of another floor may ever come up. Only the negative control (S04, gate off) sees them.
  // Before, a single Scan Context database for all floors passed every LiDAR scenario (audit AUD-034).
  if (!spec.tuning.backend.debug_disable_floor_gating && r.cross_floor_candidates != 0)
    fail(std::to_string(r.cross_floor_candidates) + " loop candidates from another floor reached the loop search (layer 1 of the gate)");
  const auto check_ate = [&](double limit) {
    if (!(m.pooled_ate <= limit)) fail("ATE RMSE " + std::to_string(m.pooled_ate) + " > " + std::to_string(limit) + " m");
    if (!(m.map_coverage >= kMinMapCoverage))
      fail("map coverage " + std::to_string(m.map_coverage) + " < " + std::to_string(kMinMapCoverage) + ": the ATE ignores keyframes left out of the map");
  };
  const auto expect_transition = [&](strataslam::TransitionType t) {
    bool found = false;
    std::string got;
    for (const auto& tr : r.transitions) {
      found = found || tr.type == t;
      got += std::string(got.empty() ? "" : ", ") + strataslam::to_string(tr.type);
    }
    if (!found) fail(std::string("no ") + strataslam::to_string(t) + " transition recorded (got: " + (got.empty() ? "none" : got) + ")");
  };
  if (spec.id == "S01" || spec.id == "S02") {
    check_ate(0.10);
    if (r.loops.empty()) fail("no loop closure accepted");
  } else if (spec.id == "S07" || spec.id == "S11") {
    strataslam::FusionOptions fo;
    fo.floor_height_prior = spec.slam.multifloor.floor_height_prior;
    // Anchors marked on the estimated clouds with 5 cm of marking error (ADR-0034).
    AnchorChoice choice;
    choice.stairs = spec.id == "S11";
    fo.anchors = make_anchors(sim, r, spec.sim.seed, 0.05, choice);
    strataslam::FusionOptions no_anchors = fo;
    no_anchors.anchors.clear();
    const auto fused = strataslam::fuse(r.map, fo);
    if (strataslam::debug_channel("FUSION")) {
      for (const auto& set : fo.anchors) {
        std::cerr << "[debug] anchor set " << set.name << ":";
        for (const auto& [fid, pts] : set.points) std::cerr << " floor " << fid << " " << pts.size() << " points";
        std::cerr << "\n";
      }
      if (fused.ok()) {
        for (const auto& w : fused.value().warnings) std::cerr << "[debug] fusion warning: " << w << "\n";
        for (const auto& c : fused.value().constraints)
          std::cerr << "[debug] constraint " << c.kind << " " << c.label << " " << c.a << "->" << c.b << " accepted " << c.accepted << " weight " << c.weight << " residual " << c.residual << "\n";
      }
    }
    if (!fused.ok()) {
      fail("fusion failed: " + fused.error().message);
    } else {
      r.fusion = fused.value();
      const FusedMetrics fm = evaluate_fusion(spec, sim, r, fused.value());
      for (const auto& [fid, e] : fm.transform_error) {
        r.report.values.push_back({"fused_floor" + std::to_string(fid) + "_trans_m", e.trans_m});
        r.report.values.push_back({"fused_floor" + std::to_string(fid) + "_rot_deg", e.rot_deg});
      }
      r.report.values.push_back({"fused_trans_err_m", fm.worst_translation});
      r.report.values.push_back({"fused_rot_err_deg", fm.worst_rotation_deg});
      r.report.values.push_back({"fused_height_err_m", fm.worst_height});
      if (spec.id == "S07") {  // the same map with the elevator shaft as the only anchors, for transparency
        strataslam::FusionOptions elevator_only = fo;
        AnchorChoice c;
        c.building = false;
        elevator_only.anchors = make_anchors(sim, r, spec.sim.seed, 0.05, c);
        const auto e = strataslam::fuse(r.map, elevator_only);
        if (e.ok()) {
          const FusedMetrics em = evaluate_fusion(spec, sim, r, e.value());
          r.report.values.push_back({"elevator_anchors_only_trans_err_m", em.worst_translation});
          r.report.values.push_back({"elevator_anchors_only_rot_err_deg", em.worst_rotation_deg});
        }
      }
      if (spec.id == "S11") {  // the same map fused from the transitions alone, reported for transparency
        const auto bare = strataslam::fuse(r.map, no_anchors);
        if (bare.ok()) {
          const FusedMetrics bm = evaluate_fusion(spec, sim, r, bare.value());
          r.report.values.push_back({"without_anchors_trans_err_m", bm.worst_translation});
          r.report.values.push_back({"without_anchors_rot_err_deg", bm.worst_rotation_deg});
        }
      }
      if (fm.worst_translation > 0.10) fail("4-DoF translation error " + std::to_string(fm.worst_translation) + " > 0.10 m");
      if (fm.worst_rotation_deg > 1.0) fail("4-DoF rotation error " + std::to_string(fm.worst_rotation_deg) + " > 1 deg");
      if (spec.id == "S11") {
        if (fm.worst_height > 0.15) fail("fused floor height error " + std::to_string(fm.worst_height) + " > 0.15 m");
        if (m.collapse != 0.0) fail("collapse rate " + std::to_string(m.collapse) + " != 0");
        if (r.map.floors.size() != 3) fail("expected 3 submaps, got " + std::to_string(r.map.floors.size()));
      } else {
        bool unknown = false;
        for (const auto& t : r.transitions) unknown = unknown || t.type == strataslam::TransitionType::Unknown;
        if (!unknown) fail("no Unknown transition recorded");
        if (r.map.floors.size() != 2) fail("expected 2 sessions/floors, got " + std::to_string(r.map.floors.size()));
      }
    }
  } else if (spec.id == "S10") {
    check_ate(0.08);
    if (r.loops.empty()) fail("no loop closure accepted");
    // Not worse than the best single modality by more than 10 %.
    const auto isolated = [&](bool camera) {
      ScenarioSpec v = spec;
      v.id = "S10-variant";
      v.sim.camera = camera;
      v.sim.lidar = !camera;
      v.slam.sensors.camera.enabled = camera;
      v.slam.sensors.lidar.enabled = !camera;
      return report_value(run_scenario(v).report, "ate_rmse_m");
    };
    const double cam_only = isolated(true), lidar_only = isolated(false);
    r.report.values.push_back({"ate_camera_only_m", cam_only});
    r.report.values.push_back({"ate_lidar_only_m", lidar_only});
    const double best = std::min(cam_only, lidar_only);
    if (m.pooled_ate > 1.10 * best) fail("combined ATE " + std::to_string(m.pooled_ate) + " is more than 10 % worse than the best single modality " + std::to_string(best));
  } else if (spec.id == "S03") {
    if (m.collapse != 0.0) fail("collapse rate " + std::to_string(m.collapse) + " != 0");
    if (m.worst_height_err > 0.15) fail("floor height error " + std::to_string(m.worst_height_err) + " > 0.15 m");
    check_ate(0.15);
    if (r.map.floors.size() != 2) fail("expected 2 submaps, got " + std::to_string(r.map.floors.size()));
    expect_transition(strataslam::TransitionType::Stairs);
    if (r.cross_floor_candidates != 0) fail("gating layer 1 returned cross-floor candidates");
  } else if (spec.id == "S04") {
    // Negative control: without the gate, aliased candidates become accepted loops across floors.
    if (!(m.collapse > 0.0)) fail("negative control did not m.collapse: the test cannot detect the problem");
  } else if (spec.id == "S08" || spec.id == "S13") {
    expect_transition(strataslam::TransitionType::Ramp);
    check_ate(0.15);
    if (r.map.floors.size() != 2) fail("expected 2 submaps, got " + std::to_string(r.map.floors.size()));
    // A 2.9 degree ramp climbs slower than the calm speed: the whole 3 m must still be measured.
    // The relative pose of a ramp is in the gravity-aligned frames: its height is the climb, not a
    // component of the displacement in the tilted body frame of the keyframe at the foot of the ramp.
    for (const auto& tr : r.transitions)
      if (tr.type == strataslam::TransitionType::Ramp && tr.relative && tr.delta_z && std::abs((*tr.relative)[3] - *tr.delta_z) > 0.5)
        fail("ramp relative height " + std::to_string((*tr.relative)[3]) + " m against a climb of " + std::to_string(*tr.delta_z) + " m");
    if (spec.id == "S13" && m.worst_height_err > 0.25) fail("floor height error " + std::to_string(m.worst_height_err) + " > 0.25 m");
  } else if (spec.id == "S05" || spec.id == "S06") {
    expect_transition(strataslam::TransitionType::Elevator);
    const double limit = spec.id == "S05" ? 0.25 : 0.50;
    if (m.worst_height_err > limit) fail("floor height error " + std::to_string(m.worst_height_err) + " > " + std::to_string(limit) + " m");
    if (spec.id == "S05" && m.collapse != 0.0) fail("collapse rate " + std::to_string(m.collapse) + " != 0");
    if (r.map.floors.size() != 2) fail("expected 2 submaps, got " + std::to_string(r.map.floors.size()));
  } else if (spec.id == "S09") {
    expect_transition(strataslam::TransitionType::Aerial);
    if (m.floor_accuracy < 0.95) fail("floor accuracy " + std::to_string(m.floor_accuracy) + " < 0.95");
  } else if (spec.id == "S14") {
    check_ate(0.15);
    if (r.map.floors.size() != 1) fail("a blackout must not create a floor");
    if (r.map.floors.empty() || r.map.floors[0].metadata.sessions.size() != 1)
      fail("the wheel odometry must carry the pose through the blackout: no new session");
  } else if (spec.id == "S12") {
    check_ate(0.15);
    if (m.inter_session_loops < 1) fail("the new session was not reconnected by a loop closure");
    if (r.map.floors.size() != 1) fail("a blackout must not create a floor");
    if (r.map.floors.empty() || r.map.floors[0].metadata.sessions.size() < 2) fail("no new session was opened after the blackout");
  }
}

}  // namespace

ScenarioResult run_scenario(const ScenarioSpec& spec) {
  Simulator sim(spec.building, spec.trajectory, spec.sim);
  ScenarioResult r = run_slam(spec, sim);
  const Measures m = measure(spec, sim, r);
  r.report.title = spec.id + " - " + spec.description;
  r.report.values = {{"floor_accuracy", m.floor_accuracy}, {"floor_accuracy_all", m.floor_accuracy_all}, {"ate_rmse_m", m.pooled_ate},
                     {"ate_worst_floor_m", m.worst_floor_ate}, {"map_coverage", m.map_coverage}, {"keyframes", double(r.keyframes.size())},
                     {"loops_accepted", double(r.loops.size())}, {"inter_session_loops", double(m.inter_session_loops)},
                     {"collapse_rate", m.collapse}, {"floor_height_err_m", m.worst_height_err},
                     {"floors", double(r.map.floors.size())}, {"transitions", double(r.transitions.size())},
                     {"cross_floor_candidates", double(r.cross_floor_candidates)}};
  apply_acceptance_criteria(spec, sim, m, r);
  return r;
}

}  // namespace strataslam_sim
