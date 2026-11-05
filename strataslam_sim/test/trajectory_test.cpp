#include <gtest/gtest.h>

#include <cmath>

#include "strataslam_sim/trajectory.hpp"

using namespace strataslam_sim;
using Eigen::Vector2d;

namespace {

BuildingConfig bcfg() {
  BuildingConfig c;
  c.num_floors = 3;
  c.with_ramp = true;
  c.with_atrium = true;
  return c;
}

bool collides(const Building& b, const TrajectorySample& s) {
  for (const auto& box : b.boxes()) {
    if (box.dynamic) continue;
    if (point_in_box(s.p, box, 0.1)) return true;
  }
  return false;
}

}  // namespace

TEST(Trajectory, WalkSpeedAndSmoothness) {
  const Building b(bcfg());
  TrajectoryConfig tc;
  TrajectoryBuilder tb(b, tc, 0, {3, 3}, 0.0);
  tb.walk({{29, 3}, {29, 19}, {3, 19}, {3, 3}});
  const Trajectory tr = tb.build();
  double vmax = 0, amax = 0, wmax = 0;
  int cruise = 0;
  for (const auto& s : tr.samples()) {
    const double v = Vector2d(s.v.x(), s.v.y()).norm();
    vmax = std::max(vmax, v);
    amax = std::max(amax, Vector2d(s.a.x(), s.a.y()).norm());
    wmax = std::max(wmax, s.omega_body.norm());
    if (std::abs(v - 1.0) < 0.02) ++cruise;
    EXPECT_NEAR(s.p.z(), 1.2, 1e-9);
    EXPECT_EQ(s.true_floor, 0);
  }
  EXPECT_NEAR(vmax, 1.0, 0.01);
  EXPECT_GT(cruise, 0.8 * tr.samples().size());
  EXPECT_LT(amax, 2.0);   // centripetal at 1 m/s on 0.8 m fillets is 1.25 m/s^2
  EXPECT_LT(wmax, 1.8);   // rad/s
  // Closed loop: ends where it started.
  EXPECT_LT((tr.samples().back().p - tr.samples().front().p).head<2>().norm(), 0.01);
}

TEST(Trajectory, WalkingNeverCollidesWithGeometry) {
  const Building b(bcfg());
  TrajectoryConfig tc;
  TrajectoryBuilder tb(b, tc, 0, {3, 3}, 0.0);
  tb.walk({{29, 3}, {29, 19}, {3, 19}, {3, 3}}).walk({{11.0, 3}, {11.0, 0.8}});
  const Trajectory tr = tb.build();
  for (const auto& s : tr.samples()) ASSERT_FALSE(collides(b, s)) << s.t << " " << s.p.transpose();
}

TEST(Trajectory, StairsRiseThreeMetersAndLabelTheTransition) {
  const Building b(bcfg());
  TrajectoryConfig tc;
  TrajectoryBuilder tb(b, tc, 0, {11.0, 3}, 0.0);
  tb.walk({{11.0, 0.8}}).stairs(0, true).hold(0.5);
  const Trajectory tr = tb.build();
  const auto& s = tr.samples();
  EXPECT_NEAR(s.front().p.z(), 1.2, 1e-9);
  EXPECT_NEAR(s.back().p.z(), 3.0 + 1.2, 0.02);
  EXPECT_EQ(s.front().true_floor, 0);
  EXPECT_EQ(s.back().true_floor, 1);
  int transition = 0;
  for (const auto& x : s) {
    ASSERT_FALSE(collides(b, x)) << x.t << " " << x.p.transpose();
    if (x.true_floor == -1) ++transition;
  }
  EXPECT_GT(transition, 7.0 / tc.dt);  // ~10 m of U-shaped stairs at 1 m/s
}

TEST(Trajectory, RampSlopeBelowFifteenDegrees) {
  const Building b(bcfg());
  TrajectoryConfig tc;
  TrajectoryBuilder tb(b, tc, 0, {28, 9.1}, 0.0);
  tb.walk({{31.0, 9.1}}).ramp(0, true).hold(0.5);
  const Trajectory tr = tb.build();
  double max_slope = 0;
  for (const auto& x : tr.samples()) {
    const double h = Vector2d(x.v.x(), x.v.y()).norm();
    if (h > 0.5) max_slope = std::max(max_slope, std::atan2(std::abs(x.v.z()), h));
    ASSERT_FALSE(collides(b, x)) << x.t << " " << x.p.transpose();
  }
  EXPECT_LT(max_slope * 180 / M_PI, 15.0);
  EXPECT_GT(max_slope * 180 / M_PI, 8.0);
  EXPECT_NEAR(tr.samples().back().p.z(), 4.2, 0.03);
  EXPECT_EQ(tr.samples().back().true_floor, 1);
}

TEST(Trajectory, ElevatorProfileAndDoor) {
  const Building b(bcfg());
  TrajectoryConfig tc;
  TrajectoryBuilder tb(b, tc, 0, {21.6, 3.0}, 0.0);
  tb.walk({{21.6, 0.5}, {21.6, -1.1}}).turn_in_place(M_PI / 2).elevator(0, 1);
  const Trajectory tr = tb.build();
  double vmax = 0, amax = 0;
  bool door_closed_during_motion = true;
  for (const auto& s : tr.samples()) {
    if (std::abs(s.v.z()) > 0.01) {
      if (!s.door_closed) door_closed_during_motion = false;
    }
    vmax = std::max(vmax, std::abs(s.v.z()));
    if (s.true_floor == -1) amax = std::max(amax, std::abs(s.a.z()));
  }
  EXPECT_NEAR(vmax, 1.0, 0.01);
  EXPECT_NEAR(amax, 0.5, 0.03);
  EXPECT_TRUE(door_closed_during_motion);
  EXPECT_NEAR(tr.samples().back().cabin_z, 3.0, 1e-9);
  EXPECT_NEAR(tr.samples().back().p.z(), 4.2, 1e-9);
  for (const auto& s : tr.samples()) ASSERT_FALSE(collides(b, s)) << s.t << " " << s.p.transpose();
}

TEST(Trajectory, VerticalFlightInTheAtrium) {
  const Building b(bcfg());
  TrajectoryConfig tc;
  tc.wobble_deg = 0.2;
  TrajectoryBuilder tb(b, tc, 0, {16, 11}, 0.0);
  tb.fly_to_level(0, 1.5).hold(1.0).fly_to_level(2, 1.5).hold(1.0);
  const Trajectory tr = tb.build();
  EXPECT_NEAR(tr.samples().back().p.z(), 6.0 + 1.5, 1e-6);
  EXPECT_EQ(tr.samples().back().true_floor, 2);
  EXPECT_EQ(tr.samples().front().true_floor, 0);
  double vmax = 0;
  for (const auto& s : tr.samples()) {
    vmax = std::max(vmax, s.v.z());
    ASSERT_FALSE(collides(b, s));
  }
  EXPECT_NEAR(vmax, 0.5, 0.01);
}

TEST(Trajectory, OrientationStaysNearLevelAndTumExports) {
  const Building b(bcfg());
  TrajectoryConfig tc;
  TrajectoryBuilder tb(b, tc, 0, {3, 3}, 0.0);
  tb.walk({{10, 3}});
  const Trajectory tr = tb.build();
  for (const auto& s : tr.samples()) {
    const Eigen::Vector3d up = s.R.col(2);
    EXPECT_GT(up.z(), std::cos(1.0 * M_PI / 180.0));
  }
  const std::string tum = tr.to_tum();
  EXPECT_EQ(std::count(tum.begin(), tum.end(), '\n'), static_cast<long>(tr.samples().size()));
  const auto pose = tr.pose_at(2.0);
  EXPECT_GT(pose.x(), 3.0);
}
