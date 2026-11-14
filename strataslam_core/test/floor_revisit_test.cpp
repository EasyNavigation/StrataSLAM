// A floor reached again by other stairs (ADR-0073) through the whole pipeline: the odometry
// measures the second flight longer than the first, the floor manager opens a new level near the
// first one, and the place recognition finds the arrival on that first level and puts it back
// there, also when the run ends before the transition closed (ADR-0074).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "strataslam/common/logging.hpp"
#include "strataslam/config/config.hpp"
#include "strataslam/pipeline/slam_system.hpp"

using namespace strataslam;
using Eigen::Vector3d;

namespace {

struct Messages final : Logger {
  std::vector<std::string> lines;
  void log(LogLevel, const std::string& m) override { lines.push_back(m); }
  bool said(const std::string& text) const {
    return std::any_of(lines.begin(), lines.end(),
                       [&](const std::string& m) { return m.find(text) != std::string::npos; });
  }
};

/// Random samples on o + u a + v b (u, v in [0, 1]), raised by `lift(x)` (a ramp).
void face(std::vector<Vector3d>& pts, std::mt19937& rng, const Vector3d& o, const Vector3d& a,
          const Vector3d& b, double per_m2, const std::function<double(double)>& lift = {}) {
  std::uniform_real_distribution<double> u(0, 1);
  const int n = static_cast<int>(a.norm() * b.norm() * per_m2);
  for (int i = 0; i < n; ++i) {
    Vector3d p = o + u(rng) * a + u(rng) * b;
    if (lift) p.z() += lift(p.x());
    pts.push_back(p);
  }
}

constexpr double kDensity = 6;      // points per m^2
constexpr double kSlope = 1 / 3.0;  // both ramps
constexpr double kRampA = 9.0;      // m, 3 m up
constexpr double kCorridorEnd = 21.0;
/// m, how much lower room B is than room A: over the floor manager's level tolerance (0.6 m without
/// a floor height prior) and under the change that makes a level (1.5 m), so a new level near floor
/// 0 for the place to decide.
constexpr double kDrop = 1.0;
constexpr double kRampB = (3.0 + kDrop) / kSlope;  // m, down to room B
constexpr double kFootB = kCorridorEnd + kRampB;
constexpr double kRoomB = kFootB;  // the room at the foot of ramp B starts here

/// A room `depth` x 8 m with pillars, its door at local x = 0 opening onto the foot of a ramp up
/// +x: the ramp's first `ramp` metres with its side walls. In local coordinates; `place` puts each
/// point. `other`: a room of another shape (deeper, its pillars elsewhere).
void room_with_ramp_foot(std::vector<Vector3d>& w, std::mt19937& rng, double ramp,
                         const std::function<Vector3d(const Vector3d&)>& place,
                         bool other = false) {
  std::vector<Vector3d> pts;
  const double d = other ? 12.0 : 8.0, hw = other ? 2.5 : 4.0;
  face(pts, rng, {-d, -hw, 0}, {d, 0, 0}, {0, 2 * hw, 0}, kDensity);  // floor
  face(pts, rng, {-d, -hw, 0}, {0, 2 * hw, 0}, {0, 0, 3}, kDensity);  // back wall
  face(pts, rng, {-d, -hw, 0}, {d, 0, 0}, {0, 0, 3}, kDensity);       // side walls
  face(pts, rng, {-d, hw, 0}, {d, 0, 0}, {0, 0, 3}, kDensity);
  face(pts, rng, {0, -hw, 0}, {0, hw - 1.5, 0}, {0, 0, 3}, kDensity);  // the door's wall
  face(pts, rng, {0, 1.5, 0}, {0, hw - 1.5, 0}, {0, 0, 3}, kDensity);
  const auto pillars =
      other ? std::vector<std::pair<double, double>>{{-1.5, -1.8}, {-4.0, 1.2}}
            : std::vector<std::pair<double, double>>{{-2.5, 2.0}, {-5.5, -2.5}, {-6.5, 2.5}};
  for (const auto& [x, y] : pillars) {
    face(pts, rng, {x, y, 0}, {0.6, 0, 0}, {0, 0, 3}, 2 * kDensity);
    face(pts, rng, {x, y, 0}, {0, 0.6, 0}, {0, 0, 3}, 2 * kDensity);
  }
  const auto up = [](double x) { return kSlope * x; };
  face(pts, rng, {0, -1.5, 0}, {ramp, 0, 0}, {0, 3, 0}, kDensity, up);
  face(pts, rng, {0, -1.5, 0}, {ramp, 0, 0}, {0, 0, 3}, kDensity, up);
  face(pts, rng, {0, 1.5, 0}, {ramp, 0, 0}, {0, 0, 3}, kDensity, up);
  for (const auto& p : pts) w.push_back(place(p));
}

/// Room A at the origin, ramp A up 3 m to a corridor, ramp B down 3 + kDrop m from it to room B:
/// room B and the foot of ramp B are room A and the foot of ramp A turned round, kDrop lower. The
/// odometry measures both flights right; the building is what a staircase the odometry reads kDrop
/// too long (ADR-0073: Campus4, prueba_00) makes of a real one: the place is floor 0, the stairs
/// say a level below it.
std::vector<Vector3d> building(bool same_room) {
  std::vector<Vector3d> w;
  std::mt19937 rng(7);
  room_with_ramp_foot(w, rng, kRampA, [](const Vector3d& p) { return p; });
  // The corridor of the upper floor, with a few boxes.
  face(w, rng, {kRampA, -1.5, 3}, {kCorridorEnd - kRampA, 0, 0}, {0, 3, 0}, kDensity);
  face(w, rng, {kRampA, -1.5, 3}, {kCorridorEnd - kRampA, 0, 0}, {0, 0, 3}, kDensity);
  face(w, rng, {kRampA, 1.5, 3}, {kCorridorEnd - kRampA, 0, 0}, {0, 0, 3}, kDensity);
  for (const double x : {12.0, 15.5, 19.0})
    face(w, rng, {x, 0.8, 3}, {0.7, 0, 0}, {0, 0, 1.5}, 2 * kDensity);
  // Ramp B above room B's ramp foot (which room_with_ramp_foot adds).
  const auto down = [](double x) { return 3.0 - kSlope * (x - kCorridorEnd); };
  const double above = kRampB - kRampA;
  face(w, rng, {kCorridorEnd, -1.5, 0}, {above, 0, 0}, {0, 3, 0}, kDensity, down);
  face(w, rng, {kCorridorEnd, -1.5, 0}, {above, 0, 0}, {0, 0, 3}, kDensity, down);
  face(w, rng, {kCorridorEnd, 1.5, 0}, {above, 0, 0}, {0, 0, 3}, kDensity, down);
  room_with_ramp_foot(
      w, rng, kRampA,
      [](const Vector3d& p) { return Vector3d(kRoomB - p.x(), -p.y(), p.z() - kDrop); },
      !same_room);
  return w;
}

/// Where a room unlike any other stands, far from the route (reached only by a jump: a ride the
/// odometry did not follow).
constexpr double kFar = 100.0;

/// Height of the walking surface at x along the route.
double surface(double x) {
  if (x <= 0 || x > kFar - 20) return 0;
  if (x <= kRampA) return kSlope * x;
  if (x <= kCorridorEnd) return 3;
  if (x <= kFootB) return 3 - kSlope * (x - kCorridorEnd);
  return -kDrop;
}

struct Walk {
  std::shared_ptr<Messages> messages = std::make_shared<Messages>();
  std::unique_ptr<SlamSystem> slam;
  std::vector<Vector3d> world;
  double noise = 0;  ///< m, standard deviation of each point's range noise
  std::mt19937 rng{11};
  double t = 0;
  double x = -0.5, yaw = M_PI;  // in room A, facing its back wall
  double carry = 1.2;           ///< m, the sensor above the walking surface
  bool imu_sends = false;       ///< an IMU at rest in the vertical (100 Hz) before each scan
  double last_yaw = M_PI;

  /// `same_room`: room B is room A turned round (the place of floor 0), or another room. `imu`:
  /// the profile declares an IMU that measures floor changes, sending while `imu_sends`.
  explicit Walk(bool same_room = true, bool imu = false) : world(building(same_room)) {
    Config cfg;
    cfg.robot.platform = Platform::Handheld;
    cfg.sensors.lidar.enabled = true;
    cfg.sensors.imu.enabled = imu;
    imu_sends = imu;
    cfg.fusion.odometry_from = OdometryFrom::Lidar;
    cfg.multifloor.floor_height_prior.reset();  // as Elevator-VIGS is mapped: storeys differ
    slam = std::make_unique<SlamSystem>(cfg, Extrinsics(), SystemTuning(), messages);
    // A hall at kFar, 14 x 7 m, its pillars where no room of the route has them.
    std::mt19937 rng(13);
    face(world, rng, {kFar - 14, -3.5, 0}, {14, 0, 0}, {0, 7, 0}, kDensity);
    face(world, rng, {kFar - 14, -3.5, 0}, {14, 0, 0}, {0, 0, 3}, kDensity);
    face(world, rng, {kFar - 14, 3.5, 0}, {14, 0, 0}, {0, 0, 3}, kDensity);
    face(world, rng, {kFar - 14, -3.5, 0}, {0, 7, 0}, {0, 0, 3}, kDensity);
    face(world, rng, {kFar, -3.5, 0}, {0, 7, 0}, {0, 0, 3}, kDensity);
    for (const auto& [px, py] :
         std::vector<std::pair<double, double>>{{-3, 1.5}, {-6.5, -2}, {-10, 0.5}, {-12, -1}}) {
      face(world, rng, {kFar + px, py, 0}, {0.5, 0, 0}, {0, 0, 3}, 2 * kDensity);
      face(world, rng, {kFar + px, py, 0}, {0, 0.5, 0}, {0, 0, 3}, 2 * kDensity);
    }
  }
  /// A scan (5 Hz) of the world within 12 m, the sensor 1.2 m above the walking surface.
  void scan() {
    const gtsam::Pose3 T(gtsam::Rot3::Yaw(yaw), gtsam::Point3(x, 0, surface(x) + carry));
    LidarScan s;
    s.stamp = 1'000'000'000LL + static_cast<Timestamp>(std::llround(t * 1e9));
    if (imu_sends) {
      for (int k = 19; k >= 0; --k) {  // the 0.2 s since the last scan, turning as the scan did
        ImuSample m;
        m.stamp = s.stamp - k * 10'000'000LL;
        m.accel = Vector3d(0, 0, 9.80665);
        m.gyro = Vector3d(0, 0, std::remainder(yaw - last_yaw, 2 * M_PI) / 0.2);
        slam->add_imu(m);
      }
    }
    last_yaw = yaw;
    std::normal_distribution<double> n(0.0, 1.0);
    for (const auto& p : world)
      if ((p - T.translation()).norm() < 12.0) {
        Vector3d q = T.transformTo(p);
        if (noise > 0) q *= 1.0 + noise * n(rng) / q.norm();
        s.points.push_back(q.cast<float>());
      }
    slam->add_lidar(s);
    t += 0.2;
  }
  /// The sensor picked up or put down to `to` m above the floor at 1.5 m/s, straight up.
  void lift_to(double to) {
    while (std::abs(to - carry) > 1e-9) {
      carry = std::abs(to - carry) < 0.3 ? to : carry + (to > carry ? 0.3 : -0.3);
      scan();
    }
  }
  void stand(double seconds) {
    for (double e = 0; e < seconds; e += 0.2) scan();
  }
  /// To `to` along the route at 1 m/s horizontally, facing the way it goes.
  void walk_to(double to) {
    yaw = to < x ? M_PI : 0.0;
    const double step = to < x ? -0.2 : 0.2;
    while (std::abs(to - x) > 1e-9) {
      x = std::abs(to - x) < 0.2 ? to : x + step;
      scan();
    }
  }
  /// Half a turn on the spot, at 0.75 rad/s.
  void turn_round() {
    const double to = yaw + M_PI;
    while (yaw < to - 1e-9) {
      yaw = std::min(to, yaw + 0.15);
      scan();
    }
    yaw = std::remainder(yaw, 2 * M_PI);
  }
  /// Room A, out and back (its first keyframes are the place recognition's: the most recent ones
  /// are not candidates), up ramp A and along the corridor, then down ramp B.
  void to_the_foot_of_ramp_b() {
    stand(2);
    walk_to(-7);
    turn_round();
    walk_to(kRampA + 4);
    stand(6);
    walk_to(kFootB);
  }
};

}  // namespace

TEST(FloorRevisit, AFloorReachedAgainByLongerStairsIsPutBackOnItByThePlace) {
  Walk w;
  w.to_the_foot_of_ramp_b();
  w.walk_to(kRoomB + 3);
  w.stand(8);
  ASSERT_EQ(w.slam->transitions().size(), 2u);
  EXPECT_EQ(w.slam->transitions()[0].to, 1);
  EXPECT_NEAR(*w.slam->transitions()[0].delta_z, 3.0, 0.3);
  EXPECT_NEAR(*w.slam->transitions()[1].delta_z, -3.0 - kDrop, 0.3)
      << "the stairs measured as they are";
  EXPECT_TRUE(w.messages->said("floor 0 reached again"));
  EXPECT_FALSE(w.messages->said("over the same floor")) << "the ramps are no device picked up";
  EXPECT_EQ(w.slam->transitions()[1].to, 0) << "back on floor 0, not on a level below it";
  EXPECT_EQ(w.slam->current_floor(), 0);
  EXPECT_EQ(w.slam->floor_status().num_floors, 2u);
}

TEST(FloorRevisit, AtTheEndOfTheRunTheOpenTransitionReachesTheFloorItsPlaceIsOn) {
  Walk w;
  w.to_the_foot_of_ramp_b();
  w.walk_to(kRoomB + 1.5);  // the run ends before the transition closed
  ASSERT_EQ(w.slam->transitions().size(), 1u);
  const MapData map = w.slam->finalize();
  EXPECT_TRUE(w.messages->said("end of the run with a floor transition open"));
  EXPECT_TRUE(w.messages->said("floor 0 reached again"));
  ASSERT_EQ(w.slam->transitions().size(), 2u);
  EXPECT_EQ(w.slam->transitions()[1].to, 0);
  EXPECT_EQ(map.floors.size(), 2u);
  // The descent's keyframes joined floor 0, the floor it reached (before ADR-0074 they stayed on
  // the withdrawn level, with no transition).
  const auto& kfs = w.slam->keyframes();
  ASSERT_FALSE(kfs.empty());
  EXPECT_EQ(kfs.back().floor, 0);
}

// The same stairs to a room that is not room A: the place does not say floor 0, and the level the
// stairs measured stays.
TEST(FloorRevisit, ANewLevelNearAFloorStaysWhenItsPlaceIsAnother) {
  Walk w(/*same_room=*/false);
  w.to_the_foot_of_ramp_b();
  w.walk_to(kRoomB + 3);
  w.stand(8);
  ASSERT_EQ(w.slam->transitions().size(), 2u);
  EXPECT_TRUE(w.messages->said("from floor 0: not seen as the same place"));
  EXPECT_FALSE(w.messages->said("reached again"));
  EXPECT_EQ(w.slam->transitions()[1].to, 2);
  EXPECT_EQ(w.slam->floor_status().num_floors, 3u);
}

// The TF floor_<k> -> odom carries the graph's correction: the robot drawn from its odometry stands
// on its latest optimized keyframe. The floor's anchor (the odometry when the floor was opened)
// leaves it off by the drift the loop closures corrected; with a 3 cm range noise the scans drift,
// and walking up and down room A closes loops. The Summit's elevator bag was 4.5 m off with the
// anchor, in the live view as in RViz. The correction is level: the keyframe's own roll and pitch
// (a robot on a ramp) would tilt the whole odometry frame about it, metres of height tens of metres
// away (the Summit's prueba_03).
TEST(FloorRevisit, TheFloorToOdometryTransformCarriesTheCorrectionOfTheGraph) {
  Walk w;
  w.noise = 0.03;
  w.stand(2);
  for (int k = 0; k < 3; ++k) {
    w.walk_to(-7);
    w.turn_round();
    w.walk_to(-0.5);
    w.turn_round();
  }
  int loops = 0;
  for (const auto& m : w.messages->lines)
    loops += m.find("loop closure accepted") != std::string::npos;
  ASSERT_GT(loops, 0);
  const auto corrected = w.slam->floor_from_odom();
  const auto anchor = w.slam->floor_anchor_from_odom();
  ASSERT_TRUE(corrected && anchor);
  const auto path = w.slam->floor_path(0);
  const auto& kfs = w.slam->keyframes();
  ASSERT_FALSE(path.empty());
  ASSERT_EQ(kfs.back().stamp, path.back().first);
  const gtsam::Pose3 drawn = *corrected * kfs.back().T_odom_base;
  const gtsam::Pose3 anchored = *anchor * kfs.back().T_odom_base;
  // The loops moved the latest keyframe (1.7 mm here): the anchor misses it, the TF does not.
  EXPECT_GT((anchored.translation() - path.back().second.translation()).norm(), 5e-4);
  EXPECT_LT((drawn.translation() - path.back().second.translation()).norm(), 1e-6);
  // Heading and position only: the keyframe's roll and pitch would tilt the whole odometry frame.
  EXPECT_LT(
      (corrected->rotation().matrix() * Eigen::Vector3d::UnitZ() - Eigen::Vector3d::UnitZ()).norm(),
      1e-9);
}

// A hand-held sensor picked up from the floor (0.3 m to 1.3 m above it in 0.7 s) opens a
// transition by its height; the LiDAR sees the same floor under it before and after, so it is
// told for what it is and the transition ends on floor 0 at once, not when the walk calms down
// (a carried IMU never does: the portable rig spent 120 s of its floor 0 in a transition).
TEST(FloorRevisit, ASensorPickedUpFromTheFloorIsNoFloorChange) {
  Walk w;
  w.carry = 0.3;
  w.stand(3);
  w.lift_to(1.3);
  w.stand(1.0);
  EXPECT_TRUE(w.messages->said("the sensor was picked up"));
  EXPECT_EQ(w.slam->floor_status().state, FloorState::OnFloor);
  EXPECT_EQ(w.slam->current_floor(), 0);
  w.walk_to(-7);
  w.stand(2);
  w.lift_to(0.3);  // put down again
  w.stand(1.0);
  EXPECT_TRUE(w.messages->said("the sensor was put down"));
  EXPECT_EQ(w.slam->floor_status().state, FloorState::OnFloor);
  EXPECT_TRUE(w.slam->transitions().empty());
  EXPECT_EQ(w.slam->floor_status().num_floors, 1u);
}

// A live view's floor cloud may be built from part of each keyframe's points (`max_points`): fewer
// points, the same floor (its extent), and the saved map's cloud keeps them all.
TEST(FloorRevisit, AViewsFloorCloudIsBuiltFromFewerPointsAndCoversTheSameFloor) {
  Walk w;
  w.stand(2);
  w.walk_to(-7);
  const auto full = w.slam->floor_cloud(0, 0.1);
  const auto view = w.slam->floor_cloud(0, 0.1, 2000);
  ASSERT_GT(full.size(), 2000u);
  EXPECT_LT(view.size(), full.size());
  EXPECT_GT(view.size(), 500u);
  auto extent = [](const std::vector<Eigen::Vector3f>& c) {
    Eigen::Vector3f lo = c.front(), hi = c.front();
    for (const auto& p : c) {
      lo = lo.cwiseMin(p);
      hi = hi.cwiseMax(p);
    }
    return std::make_pair(lo, hi);
  };
  const auto [lo_f, hi_f] = extent(full);
  const auto [lo_v, hi_v] = extent(view);
  EXPECT_LT((lo_f - lo_v).norm(), 1.0f);
  EXPECT_LT((hi_f - hi_v).norm(), 1.0f);
}

// The portable rig (2026-11-13): its camera and IMU stopped 20 s before an elevator ride, and the
// session lost in the cabin was kept on the floor it left, as a height sensor that saw no change
// keeps it. A sensor that fell silent sees none, and says so: the loss is an unobservable change,
// and a place relocalization does not find is a new floor. While the IMU sends, the session stays.
TEST(FloorRevisit, ASessionLostAfterTheImuFellSilentIsNotKeptOnItsFloor) {
  for (const bool still_sends : {true, false}) {
    Walk w(/*same_room=*/true, /*imu=*/true);
    w.stand(2);
    w.walk_to(-7);
    w.turn_round();
    w.walk_to(-1);
    w.imu_sends = still_sends;
    w.stand(3);
    w.x = 3 * kFar;  // a cabin the LiDAR sees nothing of: the odometry is lost
    w.stand(3);
    w.x = kFar - 1;  // and out of it, in a room it never saw
    w.stand(3);
    w.walk_to(kFar - 9);
    w.stand(2);
    w.slam->finalize();  // the session not relocalized is decided at the end of the run
    SCOPED_TRACE(still_sends ? "the IMU sends" : "the IMU fell silent");
    EXPECT_EQ(w.messages->said("IMU: no data for"), !still_sends);
    if (still_sends) {
      EXPECT_TRUE(w.messages->said("session kept on its floor"));
      EXPECT_EQ(w.slam->floor_status().num_floors, 1u);
    } else {
      EXPECT_TRUE(w.messages->said("opened a new floor (unobservable change)"));
      EXPECT_EQ(w.slam->floor_status().num_floors, 2u);
    }
  }
}
