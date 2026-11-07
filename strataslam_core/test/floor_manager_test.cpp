#include "strataslam/floors/floor_manager.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace strataslam;
using Eigen::Vector3d;

namespace {

constexpr double kDt = 0.1;  // 10 Hz samples

/// Feeds signals into a manager and records events.
struct Rig {
  FloorManager mgr;
  Timestamp t = 1'000'000'000'000LL;
  Vector3d pos{0, 0, 1.2};
  double baro = 0.0;
  bool use_baro = false;
  std::vector<FloorEvent> events;
  std::vector<FloorId> floors;  // floor() after each sample

  explicit Rig(FloorManagerConfig c = {}) : mgr(c) {}

  void step(const Vector3d& p, std::optional<double> baro_h = std::nullopt,
            std::optional<double> az = std::nullopt, bool ok = true) {
    FloorSample s;
    s.stamp = t;
    s.tracking_ok = ok;
    s.position = p;
    s.baro_height = baro_h;
    s.imu_vertical_accel = az;
    const FloorEvent e = mgr.update(s);
    if (e.kind != FloorEvent::Kind::None) events.push_back(e);
    floors.push_back(mgr.floor());
    pos = p;
    t += static_cast<Timestamp>(kDt * 1e9);
  }
  void stand(double seconds) {
    for (int i = 0; i < static_cast<int>(seconds / kDt); ++i)
      step(pos, use_baro ? std::optional<double>(baro) : std::nullopt);
  }
  /// Walks horizontally for `dist` m at 1 m/s on the current height.
  void walk(double dist) {
    const int n = static_cast<int>(dist / (1.0 * kDt));
    for (int i = 0; i < n; ++i)
      step(pos + Vector3d(1.0 * kDt, 0, 0), use_baro ? std::optional<double>(baro) : std::nullopt);
  }
  /// Climbs `rise` m while walking `run` m horizontally; `stepped` mimics a stair (tread/riser).
  void climb(double rise, double run, bool stepped, double tread = 0.27) {
    const int n = static_cast<int>(run / (1.0 * kDt));
    const double z0 = pos.z(), x0 = pos.x();
    for (int i = 1; i <= n; ++i) {
      const double x = x0 + run * i / n;
      double z = z0 + rise * i / n;
      if (stepped) {
        const double riser = tread * rise / run;
        const double k = std::floor((x - x0) / tread);
        const double frac = (x - x0) / tread - k;
        z = z0 +
            riser * (k + std::min(1.0, frac * 2.5));  // rise during the first 40 % of each tread
      }
      step({x, pos.y(), z}, use_baro ? std::optional<double>(baro) : std::nullopt);
    }
  }
  /// The transition's end (not the end of the odometry's part before a ride the IMU takes over).
  const FloorEvent* ended() const {
    for (const auto& e : events)
      if (e.kind == FloorEvent::Kind::TransitionEnded && !e.ride_follows) return &e;
    return nullptr;
  }
  int count(FloorEvent::Kind k) const {
    int n = 0;
    for (const auto& e : events) n += e.kind == k;
    return n;
  }
};

}  // namespace

TEST(FloorManager, StairsWithHorizontalAdvanceAreClassifiedAndOpenFloorOne) {
  Rig r;
  r.stand(3);
  r.walk(2);
  r.climb(3.0, 4.05 * 2 + 1.0, /*stepped=*/true);  // two flights' worth of run and the landing
  r.stand(4);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Stairs);
  EXPECT_EQ(e->from, 0);
  EXPECT_EQ(e->to, 1);
  EXPECT_NEAR(e->delta_z, 3.0, 0.1);
  EXPECT_TRUE(e->continuous_tracking);
  EXPECT_EQ(r.mgr.floor(), 1);
  EXPECT_EQ(r.mgr.num_floors(), 2u);
  EXPECT_NEAR(r.mgr.level_altitude(1), 3.0, 0.1);
}

TEST(FloorManager, ConstantSixDegreeRampIsARamp) {
  Rig r;
  r.stand(3);
  r.climb(3.0, 3.0 / std::tan(6.0 * M_PI / 180.0), /*stepped=*/false);
  r.stand(4);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Ramp);
  EXPECT_EQ(e->to, 1);
}

TEST(FloorManager, ElevatorIsDetectedFromTheBarometerWhileOdometryIsStatic) {
  Rig r;
  r.use_baro = true;
  r.stand(3);
  // The odometry sees the cabin standing still while the barometric height rises 3 m (smooth ride).
  for (int i = 0; i < 50; ++i) {
    const double s = i / 49.0;
    r.baro = 3.0 * (0.5 - 0.5 * std::cos(M_PI * s));
    r.step(r.pos, r.baro);
  }
  r.baro = 3.0;
  r.stand(4);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_EQ(e->to, 1);
  EXPECT_NEAR(e->delta_z, 3.0, 0.05);
  EXPECT_TRUE(e->needs_frontend_reset);
  EXPECT_NEAR(e->reset_dz, 3.0, 0.05);
  EXPECT_NEAR(r.mgr.altitude(), 3.0, 0.1);
}

TEST(FloorManager, ElevatorWithoutHeightSensorsIsInvisibleToTheManager) {
  Rig r;
  r.stand(3);
  r.stand(10);  // static odometry, no barometer, no IMU: nothing to see
  EXPECT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 0);
  EXPECT_EQ(r.mgr.floor(), 0);
}

TEST(FloorManager, GoingDownAndUpAgainReusesTheExistingLevels) {
  Rig r;
  r.stand(2);
  r.climb(3.0, 8.0, true);
  r.stand(4);
  ASSERT_EQ(r.mgr.floor(), 1);
  r.climb(-3.0, 8.0, true);
  r.stand(4);
  EXPECT_EQ(r.mgr.floor(), 0);
  r.climb(3.0, 8.0, true);
  r.stand(4);
  EXPECT_EQ(r.mgr.floor(), 1);
  EXPECT_EQ(r.mgr.num_floors(), 2u);  // no new floors were created
  int ended = 0;
  for (const auto& e : r.events) ended += e.kind == FloorEvent::Kind::TransitionEnded;
  EXPECT_EQ(ended, 3);
}

TEST(FloorManager, KeyframesDuringATransitionHaveNoFloor) {
  Rig r;
  r.stand(2);
  const size_t before = r.floors.size();
  r.climb(3.0, 10.0, true);
  size_t unassigned = 0;
  for (size_t i = before; i < r.floors.size(); ++i) unassigned += r.floors[i] == kUnassignedFloor;
  EXPECT_GT(unassigned, 60u);
  EXPECT_EQ(r.floors.front(), 0);
  r.stand(4);
  EXPECT_EQ(r.floors.back(), 1);
}

TEST(FloorManager, MidLandingIsNotMistakenForAFloor) {
  Rig r;
  r.stand(2);
  r.climb(1.5, 2.2, true);  // first flight
  r.walk(2.5);              // the mid landing: calm vertical speed for 2.5 s
  EXPECT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 0);
  r.climb(1.5, 2.2, true);  // second flight
  r.stand(4);
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 1);
  EXPECT_EQ(r.ended()->to, 1);
  EXPECT_NEAR(r.ended()->delta_z, 3.0, 0.1);
}

TEST(FloorManager, GentleRampSlowerThanTheCalmSpeedStillMeasuresTheWholeClimb) {
  // 2.9 degrees at 1 m/s climbs 0.05 m/s, below calm_vz: without the slope criterion the reference
  // follows the climb, the transition opens late and closes at the landing with a short dz.
  Rig r;
  r.stand(3);
  const double run = 1.5 / std::tan(2.9 * M_PI / 180.0);
  r.climb(1.5, run, false);
  r.walk(3.0);  // mid landing
  EXPECT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 0);
  r.climb(1.5, run, false);
  r.stand(4);
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 1);
  EXPECT_EQ(r.ended()->type, TransitionType::Ramp);
  EXPECT_EQ(r.ended()->to, 1);
  EXPECT_NEAR(r.ended()->delta_z, 3.0, 0.1);
  EXPECT_NEAR(r.mgr.level_altitude(1), 3.0, 0.1);
}

TEST(FloorManager, RampWhoseSlopeFlickersAroundTheThresholdIsOneTransition) {
  // Summit prueba_00: a ramp of about 3 degrees whose odometry z alternates 1.5 and 3.5 degrees
  // every 3 m. The short slope window drops below calm_slope on the flat parts; the long one keeps
  // it open.
  Rig r;
  r.stand(3);
  for (int i = 0; i < 12; ++i) {
    r.climb(3.0 * std::tan(1.5 * M_PI / 180.0), 3.0, false);
    r.climb(3.0 * std::tan(3.5 * M_PI / 180.0), 3.0, false);
    EXPECT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 0)
        << "closed in the middle of the ramp, pair " << i;
  }
  r.stand(6);
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 1);
  EXPECT_NEAR(r.ended()->delta_z,
              12 * 3.0 * (std::tan(1.5 * M_PI / 180.0) + std::tan(3.5 * M_PI / 180.0)), 0.25);
}

TEST(FloorManager, GentleSlopeThatReturnsToTheSameLevelIsNotAFloor) {
  Rig r;
  r.stand(2);
  r.climb(0.9, 0.9 / std::tan(2.0 * M_PI / 180.0), false);
  r.walk(5.0);
  r.climb(-0.9, 0.9 / std::tan(2.0 * M_PI / 180.0), false);
  r.stand(5);
  EXPECT_EQ(r.mgr.floor(), 0);
  EXPECT_EQ(r.mgr.num_floors(), 1u);
}

TEST(FloorManager, ForcedCloseOfAnInvalidHeightStaysOnTheFloor) {
  // Outdoor terrain: it rises 1.2 m (not a whole number of floors) in 120 m of ups (0.05) and downs
  // (0.03) that never let the odometry stay calm for a plateau.
  Rig r;
  r.stand(2);
  for (int i = 0; i < 14; ++i) {
    r.climb(0.25, 5.0, false);
    r.climb(-0.15, 5.0, false);
  }
  r.stand(4);
  EXPECT_EQ(r.mgr.floor(), 0);
  EXPECT_EQ(r.mgr.num_floors(), 1u);
  const FloorEvent* e = r.ended();
  if (e) {
    EXPECT_EQ(e->to, e->from);
  }
}

TEST(FloorManager, ASlowSustainedClimbIsNotFollowedByTheFloorReference) {
  // Summit prueba_00: right after a ramp, a second one rises 0.03 m per metre travelled (1.9 m in
  // 63 m): under the 0.04 of the short slope window. A reference that tracks the altitude with a 5
  // s time constant lags only 0.15 m behind it, under the 0.30 m that opens a transition, so the
  // climb would pass for drift.
  FloorManagerConfig cfg;
  cfg.floor_height_prior = std::nullopt;
  Rig r(cfg);
  r.stand(3);
  r.climb(3.0, 40.0, false);
  r.walk(20.0);
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 1);
  r.climb(1.9, 63.0, false);
  r.walk(40.0);
  EXPECT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 2);
  EXPECT_EQ(r.mgr.floor(), 2);
}

TEST(FloorManager, ALongPlateauAfterAClimbThatIsNotAWholeFloorIsALevel) {
  // 1.5 m up a ramp, then 60 m on a level: a mezzanine, not a multiple of the 3 m prior.
  Rig r;
  r.stand(3);
  r.climb(1.5, 15.0, false);
  r.walk(60.0);
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 1);
  EXPECT_EQ(r.ended()->to, 1);
  EXPECT_NEAR(r.ended()->delta_z, 1.5, 0.15);
  EXPECT_EQ(r.mgr.floor(), 1);
}

TEST(FloorManager, AShortLandingIsNotAPlateau) {
  Rig r;
  r.stand(3);
  r.climb(1.5, 15.0, false);
  r.walk(8.0);  // a landing of 8 s
  EXPECT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 0);
  r.climb(1.5, 15.0, false);
  r.stand(6);
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 1);
  EXPECT_NEAR(r.ended()->delta_z, 3.0, 0.15);
}

TEST(FloorManager, ALevelNearAnExistingOneIsNotANewLevel) {
  Rig r;
  r.stand(3);
  r.climb(1.5, 15.0, false);
  r.walk(60.0);
  ASSERT_EQ(r.mgr.floor(), 1);
  r.climb(-1.4, 15.0, false);  // back down to within the plateau tolerance of floor 0
  r.walk(60.0);
  EXPECT_EQ(r.mgr.floor(), 0);
  EXPECT_EQ(r.mgr.num_floors(), 2u);
}

TEST(FloorManager, SmallExcursionsDoNotChangeFloor) {
  Rig r;
  r.stand(2);
  r.climb(0.5, 2.0, false);
  r.climb(-0.5, 2.0, false);
  r.stand(5);
  // At most a close on the same floor (which puts the keyframes taken meanwhile back on it), never
  // a change.
  for (const auto& e : r.events) {
    if (e.kind == FloorEvent::Kind::TransitionEnded) {
      EXPECT_EQ(e.from, e.to);
    }
  }
  EXPECT_EQ(r.mgr.floor(), 0);
  EXPECT_EQ(r.mgr.num_floors(), 1u);
}

TEST(FloorManager, AerialPlatformVerticalFlightIsAerial) {
  FloorManagerConfig c;
  c.aerial_platform = true;
  Rig r(c);
  r.stand(2);
  for (int i = 0; i < 70; ++i)
    r.step(Vector3d(0, 0, 1.5 + 3.0 * (0.5 - 0.5 * std::cos(M_PI * i / 69.0))));
  r.stand(4);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Aerial);
  EXPECT_EQ(e->to, 1);
}

TEST(FloorManager, TrackingLossIsReportedAfterTheTimeout) {
  Rig r;
  r.stand(2);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, std::nullopt, /*ok=*/false);
  EXPECT_GE(r.count(FloorEvent::Kind::TrackingLost), 1);
}

TEST(FloorManager, ManualChangeOpensTheNextFloor) {
  Rig r;
  r.stand(2);
  const FloorEvent e = r.mgr.mark_floor_change(TransitionType::Elevator);
  EXPECT_EQ(e.kind, FloorEvent::Kind::TransitionEnded);
  EXPECT_EQ(e.to, 1);
  EXPECT_EQ(r.mgr.floor(), 1);
  EXPECT_NEAR(e.delta_z, 3.0, 1e-9);
  EXPECT_TRUE(e.needs_frontend_reset);
  EXPECT_NEAR(r.mgr.altitude(), 3.0, 1e-9);
}

TEST(FloorManager, WithoutAPriorTheMinimumVerticalChangeDecides) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  c.min_vertical_change = 1.5;
  Rig r(c);
  r.stand(2);
  r.climb(3.4, 9.0, true);
  r.stand(8);  // settle time is longer without a prior
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 1);
  EXPECT_EQ(r.mgr.floor(), 1);
}

// The run ends on the stairs, before the height settles: `finish` decides the transition now (audit
// AUD-022: an open transition at the end left its keyframes out of the map). A whole floor climbed
// closes on the new level; a climb that is not a whole floor stays on its floor; with nothing open
// there is nothing to say.
TEST(FloorManager, FinishDecidesATransitionStillOpenAtTheEndOfTheRun) {
  {
    Rig r;
    r.stand(3);
    r.walk(2);
    r.climb(3.0, 4.05 * 2 + 1.0, /*stepped=*/true);  // the run stops at the top of the stairs
    ASSERT_EQ(r.mgr.state(), FloorState::InTransition);
    const FloorEvent e = r.mgr.finish();
    ASSERT_EQ(e.kind, FloorEvent::Kind::TransitionEnded);
    EXPECT_EQ(e.from, 0);
    EXPECT_EQ(e.to, 1);
    EXPECT_NEAR(e.delta_z, 3.0, 0.15);
    EXPECT_EQ(r.mgr.floor(), 1);
  }
  {
    Rig r;
    r.stand(3);
    r.walk(2);
    r.climb(1.0, 4.0, /*stepped=*/false);  // a slope of 1 m, not a floor
    ASSERT_EQ(r.mgr.state(), FloorState::InTransition);
    const FloorEvent e = r.mgr.finish();
    ASSERT_EQ(e.kind, FloorEvent::Kind::TransitionEnded);
    EXPECT_EQ(e.to, 0);
    EXPECT_EQ(r.mgr.floor(), 0);
  }
  Rig idle;
  idle.stand(3);
  EXPECT_EQ(idle.mgr.finish().kind, FloorEvent::Kind::None);
}

// A climb that comes back down to where it started ends on its own floor, and says so: the pipeline
// puts the keyframes taken meanwhile on that floor instead of leaving them out.
TEST(FloorManager, AClimbThatComesBackDownEndsOnItsOwnFloor) {
  Rig r;
  r.stand(3);
  r.walk(2);
  r.climb(0.6, 3.0, /*stepped=*/false);
  r.climb(-0.6, 3.0, /*stepped=*/false);
  r.walk(1);
  r.stand(8);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->from, 0);
  EXPECT_EQ(e->to, 0);
  EXPECT_EQ(e->type, TransitionType::Unknown);
  EXPECT_EQ(r.mgr.floor(), 0);
}

// The odometry drifts up 0.3 m while walking 30 m on the ground floor (1 cm per metre, below what
// reads as a climb), then the stairs climb 3 m: the first floor is 3 m above the ground floor, as
// the stairs measured, not 3.3 m (audit AUD-007: S03, seed 7014).
TEST(FloorManager, ANewLevelIsPlacedByTheClimbNotByTheDriftedAltitude) {
  Rig r;
  r.stand(3);
  for (int i = 0; i < 300; ++i)
    r.step(r.pos + Vector3d(0.1, 0, 0.001));  // 30 m at 1 m/s, +0.3 m of drift
  r.stand(2);
  r.climb(3.0, 4.05 * 2 + 1.0, /*stepped=*/true);
  r.stand(6);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  ASSERT_EQ(e->to, 1);
  EXPECT_NEAR(r.mgr.level_altitude(1), 3.0, 0.1);
  // Coming back down the same stairs lands on the ground floor.
  r.climb(-3.0, 4.05 * 2 + 1.0, /*stepped=*/true);
  r.stand(6);
  EXPECT_EQ(r.mgr.floor(), 0);
  EXPECT_EQ(r.mgr.num_floors(), 2u);
}

// Without a floor height prior, `min_vertical_change` (1.5 m) decides what is a floor: a 1 m climb
// that settles is a raised part of the same floor, a 2 m one is a new level (audit AUD-031: the
// default could change unseen).
TEST(FloorManager, WithoutAHeightPriorTheMinimumVerticalChangeDecides) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  {
    Rig r(c);
    r.stand(3);
    r.walk(2);
    r.climb(1.0, 4.0, /*stepped=*/true);
    r.walk(3);
    r.stand(8);
    // No level is opened (the manager may still be waiting: a change that is not a floor closes as
    // terrain when max_transition_s runs out).
    EXPECT_EQ(r.mgr.num_floors(), 1u);
    for (const auto& e : r.events) {
      if (e.kind == FloorEvent::Kind::TransitionEnded) {
        EXPECT_EQ(e.to, 0);
      }
    }
  }
  {
    Rig r(c);
    r.stand(3);
    r.walk(2);
    r.climb(2.0, 6.0, /*stepped=*/true);
    r.walk(3);
    r.stand(8);
    EXPECT_EQ(r.mgr.floor(), 1);
    EXPECT_NEAR(r.mgr.level_altitude(1), 2.0, 0.15);
  }
}

namespace {

/// An elevator ride as the IMU feels it: `a` m/s^2 up for `t_acc` s, cruise for `t_cruise` s, `a`
/// down for `t_acc` s, sampled while the odometry stays at `p` (`ok`: tracking or lost).
void ride(Rig& r, const Vector3d& p, double a, double t_acc, double t_cruise, bool ok) {
  const auto steps = [](double s) { return static_cast<int>(std::lround(s / kDt)); };
  for (int i = 0; i < steps(t_acc); ++i) r.step(p, std::nullopt, a, ok);
  for (int i = 0; i < steps(t_cruise); ++i) r.step(p, std::nullopt, 0.0, ok);
  for (int i = 0; i < steps(t_acc); ++i) r.step(p, std::nullopt, -a, ok);
}

}  // namespace

// Elevator-VIGS Mall-E: the closing doors made the LiDAR odometry jump 0.33 m before it was lost in
// the cabin, which opened the transition as one the odometry measures; the 4.5 m ride was then
// left to the odometry, which came back 1.5 m up. The IMU, accelerating as for an elevator while
// the odometry is lost, takes the transition, and the ride ends when the cabin stops, although the
// person walks out at once (a hand-held odometry is never calm for long).
TEST(FloorManager, AnElevatorWhoseDoorsMadeTheOdometryJumpIsMeasuredByTheImu) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  const Vector3d before = r.pos, doors = before + Vector3d(0, 0, 0.35);
  for (int i = 0; i < 3; ++i)
    r.step(before + Vector3d(0, 0, 0.35 * (i + 1) / 3), std::nullopt, 0.0);
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 1);
  for (int i = 0; i < 20; ++i) r.step(doors, std::nullopt, 0.0, false);  // lost, the cabin waits
  ride(r, doors, 0.5, 2.0, 2.5, false);                                  // 4.5 m up
  for (int i = 0; i < 10; ++i) r.step(doors, std::nullopt, 0.0, false);  // the doors open
  // Back with a jump of its own, walking out: the height and the IMU bob, at rates that one second
  // does not average out (Mall-E: the odometry's speed up to 0.14 m/s, the IMU's 1 s mean up to
  // 0.19 m/s^2), so the odometry is never calm for long.
  const Vector3d out = doors + Vector3d(0, 0, 1.2);
  for (int i = 0; i < 200; ++i) {
    const double t = i * kDt;
    r.step(out + Vector3d(t, 0, 0.05 * std::sin(2 * M_PI * 0.7 * t)), std::nullopt,
           0.3 * std::sin(2 * M_PI * 1.3 * t), true);
    if (i == 30) {
      EXPECT_EQ(r.mgr.floor(), 1) << "the ride ends when the cabin stops";
    }
  }
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_EQ(e->to, 1);
  EXPECT_NEAR(e->delta_z, 4.5, 0.2);
  EXPECT_EQ(r.mgr.floor(), 1);
}

// A ride longer than the 12 s of vertical acceleration kept on a floor (15 floors in Elevator-VIGS
// Residential2): it is decided after it has stopped, and integrated from its start.
TEST(FloorManager, ALongElevatorRideIsIntegratedFromItsStart) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  ride(r, r.pos, 0.5, 2.0, 12.0, true);  // 14 m in 16 s, the odometry seeing a still cabin
  for (int i = 0; i < 40; ++i) r.step(r.pos, std::nullopt, 0.0);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_NEAR(e->delta_z, 14.0, 0.3);
}

// Elevator-VIGS Office2-E: the closing doors made the odometry jump 0.2 m, under depart_dz, before
// it was lost; its altitude, frozen there, kept the IMU from opening the 3.3 m ride.
TEST(FloorManager, AnOdometryLostAfterAJumpUnderTheThresholdStillLetsTheImuOpenTheRide) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  const Vector3d before = r.pos, doors = before + Vector3d(0, 0, 0.2);
  for (int i = 0; i < 3; ++i) r.step(before + Vector3d(0, 0, 0.2 * (i + 1) / 3), std::nullopt, 0.0);
  EXPECT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 0);
  for (int i = 0; i < 20; ++i) r.step(doors, std::nullopt, 0.0, false);
  ride(r, doors, 0.5, 2.0, 1.0, false);  // 3 m up
  for (int i = 0; i < 30; ++i) r.step(doors, std::nullopt, 0.0, false);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_NEAR(e->delta_z, 3.0, 0.2);
}

// While the odometry tracks, a vertical move it sees too (a rig lowered and raised again, a step,
// the bob of the gait) gives 1 s means of vertical acceleration over the departure threshold for
// as long as a cabin's push (Elevator-VIGS Office2: 2.5 s; Station1's cabin pushes for 1.1 s). It
// is not handed to the IMU: integrated from there, the tilt of the gravity estimate made a 93 m
// "ride" of Campus1, back where the odometry had started.
TEST(FloorManager, AVerticalMoveTheOdometryTracksIsNotARide) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  r.climb(0.4, 2.0, false);  // the odometry drifts up: a transition opens
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 1);
  double z = r.pos.z(), v = 0;
  for (const double a :
       {-0.8, 0.8, 0.8, -0.8}) {  // 0.9 m down and up again, the odometry following
    for (int i = 0; i < 15; ++i) {
      v += a * kDt;
      z += v * kDt;
      r.step({r.pos.x(), r.pos.y(), z}, std::nullopt, a);
    }
  }
  for (int i = 0; i < 300; ++i) r.step(r.pos, std::nullopt, 0.08);  // a tilted gravity estimate
  for (const auto& e : r.events) EXPECT_NE(e.type, TransitionType::Elevator);
}

// Elevator-VIGS Office4: down a flight of stairs the odometry tracks at less than half the speed
// that the IMU's 1 s means give. The steps jolt the IMU as no cabin does: not a ride.
TEST(FloorManager, AStairTheOdometryTracksHalfWayIsNotARideForTheImu) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  r.climb(-0.4, 2.0, false);  // the odometry drifts down: a transition opens
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 1);
  Vector3d p = r.pos;
  int k = 0;
  for (const double a : {-1.0, 1.0}) {
    for (int i = 0; i < 20; ++i, ++k) {
      p += Vector3d(0.6 * kDt, 0, -0.2 * kDt);
      r.step(p, std::nullopt, a + 1.5 * std::sin(2 * M_PI * 2.0 * k * kDt + 0.3));  // the steps
    }
  }
  for (int i = 0; i < 30; ++i) r.step(r.pos + Vector3d(kDt, 0, 0), std::nullopt, 0.0);
  for (int i = 0; i < 100; ++i) r.step(r.pos, std::nullopt, 0.0);
  for (const auto& e : r.events) EXPECT_NE(e.type, TransitionType::Elevator);
}

// Elevator-VIGS Office4: the odometry opened a transition 50 s before the ride (the rig lifted)
// and was lost on the way to the cabin. When the IMU takes it over, what came before ends on the
// floor it left, and the ride goes on to its own end.
TEST(FloorManager, ARideTheImuTakesOverEndsTheOdometrysTransitionOnTheFloorItLeft) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  r.climb(0.4, 2.0, false);  // the odometry drifts up: a transition opens
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 1);
  for (int i = 0; i < 100; ++i) r.step(r.pos + Vector3d(kDt, 0, 0), std::nullopt, 0.0);  // walking
  ride(r, r.pos, 0.6, 2.5, 7.5, false);  // 15 m up, lost in the cabin
  for (int i = 0; i < 40; ++i) r.step(r.pos, std::nullopt, 0.0);
  std::vector<const FloorEvent*> ends;
  for (const auto& e : r.events)
    if (e.kind == FloorEvent::Kind::TransitionEnded) ends.push_back(&e);
  ASSERT_EQ(ends.size(), 2u);
  EXPECT_TRUE(ends[0]->ride_follows);
  EXPECT_EQ(ends[0]->from, 0);
  EXPECT_EQ(ends[0]->to, 0);
  EXPECT_FALSE(ends[1]->ride_follows);
  EXPECT_EQ(ends[1]->type, TransitionType::Elevator);
  EXPECT_EQ(ends[1]->from, 0);
  EXPECT_EQ(ends[1]->to, 1);
  EXPECT_NEAR(ends[1]->delta_z, 15.0, 0.5);
}

// Elevator-VIGS Station2: a short push, mostly over before the IMU took the ride from the odometry.
// Counted from the onset it is a cabin's, and the brake that matches it ends the ride at rest.
TEST(FloorManager, AShortPushIsCountedFromTheOnsetAndTheRideEndsWhereTheCabinBrakes) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  r.climb(0.4, 2.0, false);  // the odometry drifts up: a transition opens
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 1);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  const auto steps = [](double sec) { return static_cast<int>(std::lround(sec / kDt)); };
  for (int i = 0; i < steps(1.6); ++i) r.step(r.pos, std::nullopt, -0.4);  // 0.64 m/s down
  for (int i = 0; i < steps(4.0); ++i) r.step(r.pos, std::nullopt, 0.0);
  for (int i = 0; i < steps(1.6); ++i) r.step(r.pos, std::nullopt, 0.4);
  for (int i = 0; i < steps(3.0); ++i) r.step(r.pos, std::nullopt, 0.0);  // at rest, still inside
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr) << "the ride ends where the cabin stops";
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_NEAR(e->delta_z, -0.64 * (1.6 + 4.0), 0.4);
}

// Elevator-VIGS Office3: walked into the cabin and it left at once. The boarding walk, still in the
// odometry's last seconds, is no walk out of a cabin that is moving: the ride goes on to its brake.
TEST(FloorManager, TheWalkIntoACabinThatLeavesAtOnceDoesNotEndTheRide) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  r.climb(0.4, 2.0, false);  // the odometry drifts up: a transition opens
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 1);
  const auto steps = [](double sec) { return static_cast<int>(std::lround(sec / kDt)); };
  Vector3d p = r.pos;
  for (int i = 0; i < steps(3.0); ++i) r.step(p += Vector3d(kDt, 0, 0), std::nullopt, 0.0);
  for (int i = 0; i < steps(1.5); ++i)  // the cabin leaves, the odometry still walking in
    r.step(p += Vector3d(kDt, 0, 0), std::nullopt, -0.8);
  for (int i = 0; i < steps(3.0); ++i)  // and then drifting in the closed cabin
    r.step(p += Vector3d(kDt, 0, 0), std::nullopt, 0.0);
  for (int i = 0; i < steps(2.0); ++i) r.step(p, std::nullopt, 0.0);
  for (int i = 0; i < steps(1.5); ++i) r.step(p, std::nullopt, 0.8);
  for (int i = 0; i < steps(3.0); ++i) r.step(p, std::nullopt, 0.0);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_NEAR(e->delta_z, -1.2 * (1.5 + 5.0), 0.5);
}

// Elevator-VIGS Station3: the brake of a ride whose push the IMU did not see, taken for a push.
// Its speed never comes back to rest, and the walk that follows, longer than any cabin, ends it:
// no ride integrated through the walk.
TEST(FloorManager, ALoneBrakeTakenForAPushIsEndedByAWalkLongerThanACabin) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  r.climb(0.4, 2.0, false);  // the odometry drifts up: a transition opens
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 1);
  const auto steps = [](double sec) { return static_cast<int>(std::lround(sec / kDt)); };
  for (int i = 0; i < steps(1.5); ++i) r.step(r.pos, std::nullopt, -0.5, false);  // the brake
  for (int i = 0; i < steps(2.0); ++i) r.step(r.pos, std::nullopt, 0.0, false);
  Vector3d p = r.pos;
  for (int i = 0; i < steps(45.0); ++i)  // a walk at 1 m/s, the IMU bobbing with the steps
    r.step(p += Vector3d(kDt, 0, 0), std::nullopt, 0.3 * std::sin(2 * M_PI * 1.3 * i * kDt));
  for (const auto& e : r.events) EXPECT_NE(e.type, TransitionType::Elevator) << e.delta_z;
  EXPECT_FALSE(r.mgr.in_unobserved_transition()) << "the transition is the odometry's again";
}

// Elevator-VIGS Residential3: a cabin that starts gently (under the departure threshold) and brakes
// hard. The IMU takes the brake for a push; when it never comes back to rest, the push is found
// back in the history, before the cabin's calm, and the ride measured from it.
TEST(FloorManager, ARideWhosePushWasUnderTheThresholdIsFoundBackFromItsBrake) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  r.climb(0.4, 2.0, false);  // the odometry drifts up: a transition opens
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 1);
  const auto steps = [](double sec) { return static_cast<int>(std::lround(sec / kDt)); };
  for (int i = 0; i < steps(4.0); ++i) r.step(r.pos, std::nullopt, 0.0, false);
  for (int i = 0; i < steps(4.0); ++i) r.step(r.pos, std::nullopt, 0.18, false);  // 0.72 m/s up
  for (int i = 0; i < steps(8.0); ++i) r.step(r.pos, std::nullopt, 0.0, false);
  for (int i = 0; i < steps(1.6); ++i) r.step(r.pos, std::nullopt, -0.45, false);  // the brake
  for (int i = 0; i < steps(2.0); ++i) r.step(r.pos, std::nullopt, 0.0, false);
  Vector3d p = r.pos;
  for (int i = 0; i < steps(45.0); ++i)  // out, at 1 m/s, the IMU bobbing with the steps
    r.step(p += Vector3d(kDt, 0, 0), std::nullopt, 0.3 * std::sin(2 * M_PI * 1.3 * i * kDt));
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_NEAR(e->delta_z, 0.5 * 0.72 * 4.0 + 0.72 * 8.0 + 0.5 * 0.72 * 1.6, 0.6);
}

// Elevator-VIGS Campus6: the odometry moves in the cabin as it brakes. The ride does not end there
// as a walk out of it: it ends when the cabin has stopped.
TEST(FloorManager, ARideIsNotWalkedOutOfWhileTheCabinBrakes) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  r.climb(0.4, 2.0, false);  // the odometry drifts up: a transition opens
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 1);
  const auto steps = [](double sec) { return static_cast<int>(std::lround(sec / kDt)); };
  for (int i = 0; i < steps(2.0); ++i) r.step(r.pos, std::nullopt, 0.6);  // 1.2 m/s up
  Vector3d p = r.pos;
  for (int i = 0; i < steps(5.0); ++i) r.step(p, std::nullopt, 0.0);
  for (int i = 0; i < steps(3.0); ++i) r.step(p += Vector3d(kDt, 0, 0), std::nullopt, 0.0);
  for (int i = 0; i < steps(2.0); ++i)  // braking, the odometry still moving
    r.step(p += Vector3d(kDt, 0, 0), std::nullopt, -0.6);
  for (int i = 0; i < steps(1.0); ++i) r.step(p, std::nullopt, 0.0);
  for (int i = 0; i < steps(6.0); ++i) r.step(p += Vector3d(kDt, 0, 0), std::nullopt, 0.0);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_NEAR(e->delta_z, 1.2 * 2.0 + 1.2 * 8.0, 0.5);
}

// Elevator-VIGS Office3: a jolt taken for a cabin's push, then a walk whose steps jolt the IMU. It
// is no ride once the steps are clear, so the ride that follows is taken from its own start.
TEST(FloorManager, AJoltFollowedBySteppingIsDroppedInTimeForTheRideThatFollows) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  r.climb(0.4, 2.0, false);  // the odometry drifts up: a transition opens
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 1);
  const auto steps = [](double sec) { return static_cast<int>(std::lround(sec / kDt)); };
  for (int i = 0; i < steps(1.5); ++i) r.step(r.pos, std::nullopt, -0.6, false);  // the jolt
  Vector3d p = r.pos;
  for (int i = 0; i < steps(15.0); ++i)  // a walk at 1 m/s, its steps jolting the IMU
    r.step(p += Vector3d(kDt, 0, 0), std::nullopt, 1.5 * std::sin(2 * M_PI * 2.0 * i * kDt + 0.3));
  for (int i = 0; i < steps(3.0); ++i) r.step(p, std::nullopt, 0.0);  // in the cabin
  ride(r, p, 0.6, 2.5, 7.5, false);                                   // 15 m up
  for (int i = 0; i < steps(4.0); ++i) r.step(p, std::nullopt, 0.0);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_NEAR(e->delta_z, 15.0, 0.6);
}

// Elevator-VIGS Office1: the odometry opened the transition before the ride and tracked the cabin
// through it, still. The IMU's 15 m are its own.
TEST(FloorManager, ARideTheOdometryTracksAsAStillCabinIsMeasuredByTheImu) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  r.climb(0.4, 2.0, false);  // the odometry drifts up: a transition opens
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 1);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  ride(r, r.pos, 0.6, 2.5, 7.5, true);  // 1.5 m/s for 10 s: 15 m, the cabin still to the odometry
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_NEAR(e->delta_z, 15.0, 0.5);
}

// Elevator-VIGS Office1-H: a bias and the tilt of the gravity estimate add 0.03 m/s^2 through a
// 33 m ride, and the speed integrated from the onset was still 0.94 m/s when the cabin had
// stopped. The stop is the brake that answers the push; the zero-velocity update at that end
// removes a constant bias from the height.
TEST(FloorManager, ALongRideWithABiasedAccelerometerStopsWhereTheCabinBrakes) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  const double bias = 0.03;
  const auto steps = [](double s) { return static_cast<int>(std::lround(s / kDt)); };
  for (int i = 0; i < steps(2.0); ++i) r.step(r.pos, std::nullopt, 0.6 + bias, false);
  for (int i = 0; i < steps(20.0); ++i) r.step(r.pos, std::nullopt, bias, false);
  for (int i = 0; i < steps(2.0); ++i) r.step(r.pos, std::nullopt, -0.6 + bias, false);
  for (int i = 0; i < steps(3.0); ++i) r.step(r.pos, std::nullopt, bias, false);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr) << "the ride ends when the cabin has braked";
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_NEAR(e->delta_z, 1.2 * 22.0, 0.5);  // 1.2 m/s for 22 s
}

// What the bias estimate has left is larger than the calm threshold (Elevator-VIGS Mall-M: the
// estimate drifted while walking and the cabin at rest read +0.12 m/s^2): the ride still stops
// where the cabin brakes, its accelerations measured from the one before it started.
TEST(FloorManager, ARideStopsWhereTheCabinBrakesWithABiasTheEstimateLeft) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  const double left = 0.13;
  const auto steps = [](double s) { return static_cast<int>(std::lround(s / kDt)); };
  for (int i = 0; i < steps(5.0); ++i) r.step(r.pos, std::nullopt, left);  // in the cabin
  for (int i = 0; i < steps(2.0); ++i) r.step(r.pos, std::nullopt, -0.6 + left, false);
  for (int i = 0; i < steps(15.0); ++i) r.step(r.pos, std::nullopt, left, false);
  for (int i = 0; i < steps(2.0); ++i) r.step(r.pos, std::nullopt, 0.6 + left, false);
  for (int i = 0; i < steps(3.0); ++i) r.step(r.pos, std::nullopt, left, false);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr) << "the ride ends at rest, before anyone walks out";
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_NEAR(e->delta_z, -1.2 * 17.0, 0.5);  // 1.2 m/s for 17 s
}

// Elevator-VIGS Office4: up by elevator, measured by the IMU (14.75 m), and back down the stairs,
// measured by the odometry (15.43 m), to where the walk had started. The level reached by the ride
// is only known to a few percent of it: the stairs end on the first level, not 0.68 m under it.
TEST(FloorManager, BackDownTheStairsFromALevelTheImuMeasuredIsTheLevelItLeft) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  ride(r, r.pos, 0.6, 2.5, 7.5, true);  // 15 m up, the cabin still to the odometry
  for (int i = 0; i < 40; ++i) r.step(r.pos, std::nullopt, 0.0);
  ASSERT_EQ(r.mgr.floor(), 1);
  r.walk(3);
  r.climb(-15.7, 40.0, /*stepped=*/true);  // the stairs, all the way down
  r.walk(3);
  r.stand(8);
  EXPECT_EQ(r.mgr.floor(), 0);
  EXPECT_EQ(r.mgr.num_floors(), 2u);
}

// A cabin that brakes gently, under the pulse threshold, is never seen braking; the ride ends when
// the odometry walks out of it, at a walking pace (Elevator-VIGS Factory-H, where the cabin's
// bumps hid the brake).
TEST(FloorManager, ARideWhoseBrakeIsNotSeenEndsWhenTheOdometryWalksOut) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.5);     // 1.5 m/s
  for (int i = 0; i < 100; ++i) r.step(r.pos, std::nullopt, -0.15);  // a gentle brake: 0 m/s
  for (int i = 0; i < 20; ++i) r.step(r.pos, std::nullopt, 0.0);
  for (int i = 0; i < 60; ++i)  // out, at 1 m/s, the IMU bobbing
    r.step(r.pos + Vector3d(kDt, 0, 0), std::nullopt, 0.3 * std::sin(2 * M_PI * 1.3 * i * kDt));
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_NEAR(e->delta_z, 0.5 * 1.5 * 3.0 + 0.5 * 1.5 * 10.0, 0.5);
}

// A jolt the IMU took for a cabin starting (Elevator-VIGS Factory-H, boarding) is no ride once the
// odometry walks on; the transition goes back to the odometry, and the ride that follows is
// measured from its own start.
TEST(FloorManager, AJoltThatWasNoRideLeavesTheNextRideToItsOwnStart) {
  FloorManagerConfig c;
  c.floor_height_prior.reset();
  Rig r(c);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  r.climb(0.4, 2.0, false);  // the odometry drifts up: a transition opens
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionStarted), 1);
  for (int i = 0; i < 12; ++i) r.step(r.pos, std::nullopt, -0.6, false);  // a jolt, lost
  for (int i = 0; i < 12; ++i) r.step(r.pos, std::nullopt, 0.6, false);
  for (int i = 0; i < 50; ++i)  // walks on, into the cabin
    r.step(r.pos + Vector3d(kDt, 0, 0), std::nullopt, 0.0);
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  ride(r, r.pos, 0.6, 2.5, 5.0, true);  // 11.25 m up, the cabin still to the odometry
  for (int i = 0; i < 30; ++i) r.step(r.pos, std::nullopt, 0.0);
  const FloorEvent* e = nullptr;
  for (const auto& ev : r.events)
    if (ev.kind == FloorEvent::Kind::TransitionEnded && ev.type == TransitionType::Elevator)
      e = &ev;
  ASSERT_NE(e, nullptr);
  EXPECT_NEAR(e->delta_z, 11.25, 0.4);
}
