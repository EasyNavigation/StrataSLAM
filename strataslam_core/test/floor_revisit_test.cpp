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

/// Height of the walking surface at x along the route.
double surface(double x) {
  if (x <= 0) return 0;
  if (x <= kRampA) return kSlope * x;
  if (x <= kCorridorEnd) return 3;
  if (x <= kFootB) return 3 - kSlope * (x - kCorridorEnd);
  return -kDrop;
}

struct Walk {
  std::shared_ptr<Messages> messages = std::make_shared<Messages>();
  std::unique_ptr<SlamSystem> slam;
  std::vector<Vector3d> world;
  double t = 0;
  double x = -0.5, yaw = M_PI;  // in room A, facing its back wall

  /// `same_room`: room B is room A turned round (the place of floor 0), or another room.
  explicit Walk(bool same_room = true) : world(building(same_room)) {
    Config cfg;
    cfg.robot.platform = Platform::Handheld;
    cfg.sensors.lidar.enabled = true;
    cfg.sensors.imu.enabled = false;
    cfg.fusion.odometry_from = OdometryFrom::Lidar;
    cfg.multifloor.floor_height_prior.reset();  // as Elevator-VIGS is mapped: storeys differ
    slam = std::make_unique<SlamSystem>(cfg, Extrinsics(), SystemTuning(), messages);
  }
  /// A scan (5 Hz) of the world within 12 m, the sensor 1.2 m above the walking surface.
  void scan() {
    const gtsam::Pose3 T(gtsam::Rot3::Yaw(yaw), gtsam::Point3(x, 0, surface(x) + 1.2));
    LidarScan s;
    s.stamp = 1'000'000'000LL + static_cast<Timestamp>(std::llround(t * 1e9));
    for (const auto& p : world)
      if ((p - T.translation()).norm() < 12.0) s.points.push_back(T.transformTo(p).cast<float>());
    slam->add_lidar(s);
    t += 0.2;
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
