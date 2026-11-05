#include <gtest/gtest.h>

#include <cmath>

#include "strataslam/floors/floor_manager.hpp"

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

  void step(const Vector3d& p, std::optional<double> baro_h = std::nullopt, std::optional<double> az = std::nullopt, bool ok = true) {
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
    for (int i = 0; i < static_cast<int>(seconds / kDt); ++i) step(pos, use_baro ? std::optional<double>(baro) : std::nullopt);
  }
  /// Walks horizontally for `dist` m at 1 m/s on the current height.
  void walk(double dist) {
    const int n = static_cast<int>(dist / (1.0 * kDt));
    for (int i = 0; i < n; ++i) step(pos + Vector3d(1.0 * kDt, 0, 0), use_baro ? std::optional<double>(baro) : std::nullopt);
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
        z = z0 + riser * (k + std::min(1.0, frac * 2.5));  // rise during the first 40 % of each tread
      }
      step({x, pos.y(), z}, use_baro ? std::optional<double>(baro) : std::nullopt);
    }
  }
  const FloorEvent* ended() const {
    for (const auto& e : events)
      if (e.kind == FloorEvent::Kind::TransitionEnded) return &e;
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
  r.climb(1.5, 2.2, true);   // first flight
  r.walk(2.5);               // the mid landing: calm vertical speed for 2.5 s
  EXPECT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 0);
  r.climb(1.5, 2.2, true);   // second flight
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
  r.walk(3.0);               // mid landing
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
  // Summit prueba_00: a ramp of about 3 degrees whose odometry z alternates 1.5 and 3.5 degrees every
  // 3 m. The short slope window drops below calm_slope on the flat parts; the long one keeps it open.
  Rig r;
  r.stand(3);
  for (int i = 0; i < 12; ++i) {
    r.climb(3.0 * std::tan(1.5 * M_PI / 180.0), 3.0, false);
    r.climb(3.0 * std::tan(3.5 * M_PI / 180.0), 3.0, false);
    EXPECT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 0) << "closed in the middle of the ramp, pair " << i;
  }
  r.stand(6);
  ASSERT_EQ(r.count(FloorEvent::Kind::TransitionEnded), 1);
  EXPECT_NEAR(r.ended()->delta_z, 12 * 3.0 * (std::tan(1.5 * M_PI / 180.0) + std::tan(3.5 * M_PI / 180.0)), 0.25);
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
  // Summit prueba_00: right after a ramp, a second one rises 0.03 m per metre travelled (1.9 m in 63 m): under
  // the 0.04 of the short slope window. A reference that tracks the altitude with a 5 s time constant lags
  // only 0.15 m behind it, under the 0.30 m that opens a transition, so the climb would pass for drift.
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
  // At most a close on the same floor (which puts the keyframes taken meanwhile back on it), never a change.
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
  for (int i = 0; i < 70; ++i) r.step(Vector3d(0, 0, 1.5 + 3.0 * (0.5 - 0.5 * std::cos(M_PI * i / 69.0))));
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

// The run ends on the stairs, before the height settles: `finish` decides the transition now (audit AUD-022: an open
// transition at the end left its keyframes out of the map). A whole floor climbed closes on the new level; a
// climb that is not a whole floor stays on its floor; with nothing open there is nothing to say.
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

// A climb that comes back down to where it started ends on its own floor, and says so: the pipeline puts the
// keyframes taken meanwhile on that floor instead of leaving them out.
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

// The odometry drifts up 0.3 m while walking 30 m on the ground floor (1 cm per metre, below what reads as a climb),
// then the stairs climb 3 m: the first floor is 3 m above the ground floor, as the stairs measured, not 3.3 m
// (audit AUD-007: S03, seed 7014).
TEST(FloorManager, ANewLevelIsPlacedByTheClimbNotByTheDriftedAltitude) {
  Rig r;
  r.stand(3);
  for (int i = 0; i < 300; ++i) r.step(r.pos + Vector3d(0.1, 0, 0.001));  // 30 m at 1 m/s, +0.3 m of drift
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

// Without a floor height prior, `min_vertical_change` (1.5 m) decides what is a floor: a 1 m climb that settles is a
// raised part of the same floor, a 2 m one is a new level (audit AUD-031: the default could change unseen).
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
    // No level is opened (the manager may still be waiting: a change that is not a floor closes as terrain when
    // max_transition_s runs out).
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
