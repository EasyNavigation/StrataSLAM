#include "strataslam/backend/backend.hpp"

#include <gtest/gtest.h>

#include <random>

using namespace strataslam;
using Eigen::Vector3d;

namespace {

/// Square trajectory of `n` keyframes per side, 0.25 m steps, with a yaw turn of 90 deg per corner.
std::vector<gtsam::Pose3> square_truth(int n) {
  std::vector<gtsam::Pose3> t;
  gtsam::Pose3 p(gtsam::Rot3(), Vector3d(0, 0, 1.2));
  t.push_back(p);
  for (int side = 0; side < 4; ++side) {
    for (int i = 0; i < n; ++i) {
      p = p * gtsam::Pose3(gtsam::Rot3(), Vector3d(0.25, 0, 0));
      t.push_back(p);
    }
    p = p * gtsam::Pose3(gtsam::Rot3::Yaw(M_PI / 2), Vector3d::Zero());
    t.push_back(p);
  }
  return t;
}

KeyframeInput kf(KeyframeId id, FloorId floor, const gtsam::Pose3& pose, SessionId session = 0) {
  KeyframeInput k;
  k.id = id;
  k.stamp = static_cast<Timestamp>(id) * 100000000;
  k.floor = floor;
  k.session = session;
  k.T_odom_base = pose;
  return k;
}

}  // namespace

TEST(Backend, LoopClosureReducesDriftAndOptimizationIsConsistent) {
  const auto truth = square_truth(40);
  std::mt19937 rng(5);
  std::normal_distribution<double> n(0, 1);
  // Noisy odometry accumulated in the odom frame.
  std::vector<gtsam::Pose3> odom{truth[0]};
  for (size_t i = 1; i < truth.size(); ++i) {
    const gtsam::Pose3 d = truth[i - 1].between(truth[i]);
    const gtsam::Pose3 dn(d.rotation() * gtsam::Rot3::Yaw(0.004 * n(rng)),
                          Vector3d(d.translation() + Vector3d(0.01 * n(rng), 0.01 * n(rng), 0.0)));
    odom.push_back(odom.back() * dn);
  }
  Backend be(BackendConfig{});
  for (size_t i = 0; i < truth.size(); ++i) {
    std::optional<KeyframeId> from;
    gtsam::Pose3 rel;
    if (i > 0) {
      from = i - 1;
      rel = odom[i - 1].between(odom[i]);
    }
    ASSERT_TRUE(be.add_keyframe(kf(i, 0, odom[i]), from, rel, std::nullopt).ok());
  }
  const double drift_before = (odom.back().translation() - truth.back().translation()).norm();
  EXPECT_GT(drift_before, 0.05);
  // Loop closure: the last keyframe is back at the first pose (true relative pose is exact).
  ASSERT_TRUE(be.add_loop(0, truth.size() - 1, truth.front().between(truth.back())).ok());
  be.optimize_all();
  const gtsam::Pose3 last = be.pose_odom(truth.size() - 1);
  const gtsam::Pose3 first = be.pose_odom(0);
  const double err_after =
      (first.between(last).translation() - truth.front().between(truth.back()).translation())
          .norm();
  EXPECT_LT(err_after, 0.5 * drift_before);
  EXPECT_EQ(be.loops().size(), 1u);
  EXPECT_EQ(be.floor_of(0), 0);
}

TEST(Backend, InconsistentLoopIsRejectedByMahalanobisCheck) {
  const auto truth = square_truth(20);
  Backend be(BackendConfig{});
  for (size_t i = 0; i < truth.size(); ++i) {
    std::optional<KeyframeId> from;
    gtsam::Pose3 rel;
    if (i > 0) {
      from = i - 1;
      rel = truth[i - 1].between(truth[i]);
    }
    ASSERT_TRUE(be.add_keyframe(kf(i, 0, truth[i]), from, rel, std::nullopt).ok());
  }
  // A loop claiming that the last pose is 3 m away from where odometry puts it.
  const gtsam::Pose3 bad =
      truth.front().between(truth.back()) * gtsam::Pose3(gtsam::Rot3(), Vector3d(3, 0, 0));
  const auto r = be.add_loop(0, truth.size() - 1, bad);
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().message.find("Mahalanobis"), std::string::npos);
  EXPECT_TRUE(be.loops().empty());
  // The exact loop is accepted.
  EXPECT_TRUE(be.add_loop(0, truth.size() - 1, truth.front().between(truth.back())).ok());
}

TEST(BackendGating, LoopBetweenDifferentFloorsIsRejected) {
  Backend be(BackendConfig{});
  const gtsam::Pose3 p0(gtsam::Rot3(), Vector3d(0, 0, 1.2));
  ASSERT_TRUE(be.add_keyframe(kf(0, 0, p0), std::nullopt, {}, std::nullopt).ok());
  ASSERT_TRUE(be.add_keyframe(kf(1, 0, p0 * gtsam::Pose3(gtsam::Rot3(), Vector3d(1, 0, 0))), 0,
                              gtsam::Pose3(gtsam::Rot3(), Vector3d(1, 0, 0)), std::nullopt)
                  .ok());
  // Floor 1 has an identical-looking keyframe at the same xy, 3 m higher.
  const gtsam::Pose3 q0(gtsam::Rot3(), Vector3d(0, 0, 4.2));
  ASSERT_TRUE(be.add_keyframe(kf(2, 1, q0), std::nullopt, {}, std::nullopt).ok());
  const auto r = be.add_loop(0, 2, gtsam::Pose3());
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().message.find("different floors"), std::string::npos);
  EXPECT_TRUE(be.loops().empty());
  EXPECT_EQ(be.floors().size(), 2u);
  EXPECT_FALSE(be.verify_loop_allowed(0, 2).ok());
  EXPECT_TRUE(be.verify_loop_allowed(0, 1).ok());
}

TEST(BackendGating, Layer3FloorGraphRefusesAFactorToANodeItDoesNotHold) {
  // Layer 3 checks graph membership on its own: no floor labels are involved here, so layer 2
  // cannot be what rejects the factor, and the consistency check is off.
  const BackendConfig cfg;
  FloorGraph g0(0, gtsam::Pose3(), cfg), g1(1, gtsam::Pose3(gtsam::Rot3(), Vector3d(0, 0, 3)), cfg);
  const gtsam::Pose3 step(gtsam::Rot3(), Vector3d(1, 0, 0));
  g0.add_keyframe(0, 0, 0, gtsam::Pose3(), std::nullopt, {}, 1.0, std::nullopt);
  g0.add_keyframe(1, 100000000, 0, step, 0, step, 1.0, std::nullopt);
  g1.add_keyframe(2, 200000000, 0, gtsam::Pose3(), std::nullopt, {}, 1.0, std::nullopt);
  const size_t edges_before = g0.edge_records().size();
  for (const auto& [from, to] : {std::pair<KeyframeId, KeyframeId>{0, 2}, {2, 0}, {2, 2}}) {
    const auto r = g0.add_loop(from, to, gtsam::Pose3(), /*check_consistency=*/false);
    ASSERT_FALSE(r.ok()) << from << " -> " << to;
    EXPECT_NE(r.error().message.find("backend assertion"), std::string::npos) << r.error().message;
  }
  EXPECT_EQ(g0.edge_records().size(), edges_before);
  EXPECT_EQ(g1.size(), 1u);
  EXPECT_TRUE(g0.add_loop(0, 1, step, /*check_consistency=*/false).ok());  // same graph: allowed
}

TEST(BackendGating, TransitionKeyframesAreInNoGraphAndCannotCloseLoops) {
  Backend be(BackendConfig{});
  const gtsam::Pose3 p(gtsam::Rot3(), Vector3d(0, 0, 1.2));
  ASSERT_TRUE(be.add_keyframe(kf(0, 0, p), std::nullopt, {}, std::nullopt).ok());
  ASSERT_TRUE(be.add_keyframe(kf(1, -1, p), std::nullopt, {}, std::nullopt).ok());
  EXPECT_EQ(be.floor_of(1), -1);
  EXPECT_EQ(be.floors().size(), 1u);
  EXPECT_EQ(be.floor(0).size(), 1u);  // the transition keyframe joined no graph
  EXPECT_FALSE(be.add_loop(0, 1, gtsam::Pose3()).ok());
  EXPECT_FALSE(be.add_loop(1, 1, gtsam::Pose3()).ok());
}

TEST(BackendGating, DebugFlagRemovesTheGateForTheNegativeControl) {
  BackendConfig cfg;
  cfg.debug_disable_floor_gating = true;
  Backend be(cfg);
  const gtsam::Pose3 p0(gtsam::Rot3(), Vector3d(0, 0, 1.2));
  ASSERT_TRUE(be.add_keyframe(kf(0, 0, p0), std::nullopt, {}, std::nullopt).ok());
  ASSERT_TRUE(be.add_keyframe(kf(1, 1, p0 * gtsam::Pose3(gtsam::Rot3(), Vector3d(0, 0, 3))), 0,
                              gtsam::Pose3(gtsam::Rot3(), Vector3d(0, 0, 3)), std::nullopt)
                  .ok());
  EXPECT_TRUE(be.add_loop(0, 1, gtsam::Pose3()).ok());  // claims the two floors are the same place
  ASSERT_EQ(be.loops().size(), 1u);
  EXPECT_NE(be.loops()[0].from_floor, be.loops()[0].to_floor);  // ground-truth floors still differ
  // With the gate off both floors live in one graph (the robust kernel may still down-weight it).
  EXPECT_EQ(be.floors().size(), 1u);
}

TEST(Backend, NewSessionIsBridgedWithAWeakFactor) {
  Backend be(BackendConfig{});
  const gtsam::Pose3 p(gtsam::Rot3(), Vector3d(0, 0, 1.2));
  ASSERT_TRUE(be.add_keyframe(kf(0, 0, p, 0), std::nullopt, {}, std::nullopt).ok());
  const gtsam::Pose3 p1 = p * gtsam::Pose3(gtsam::Rot3(), Vector3d(1, 0, 0));
  ASSERT_TRUE(be.add_keyframe(kf(1, 0, p1, 1), std::nullopt, {}, 0).ok());
  const auto edges = be.floor(0).edge_records();
  ASSERT_EQ(edges.size(), 1u);
  EXPECT_EQ(edges[0].type, "bridge");
  be.optimize_all();
  EXPECT_NEAR(be.pose_odom(1).x(), 1.0, 0.05);
}

// The barometer and the gravity prior are wired into the floor graph with the right sign: with an
// uncertain (degraded) odometry step, the optimized height follows the barometric height difference
// and the optimized attitude follows the measured up direction. A sign error or a missing factor
// fails these (audit AUD-009..011).
TEST(Backend, BarometricHeightDifferenceSetsTheHeightWhenTheOdometryIsUncertain) {
  BackendConfig cfg;  // odometry 0.02 m (x10 when degraded = 0.2 m), barometer 0.10 m
  Backend b(cfg);
  KeyframeInput a = kf(0, 0, gtsam::Pose3(gtsam::Rot3(), Vector3d(0, 0, 0)));
  a.baro_height = 10.0;
  ASSERT_TRUE(b.add_keyframe(a, std::nullopt, gtsam::Pose3(), std::nullopt).ok());
  // The odometry says the robot moved 1 m forward on the level; the barometer says it went 1 m up.
  const gtsam::Pose3 step(gtsam::Rot3(), Vector3d(1, 0, 0));
  KeyframeInput c = kf(1, 0, step);
  c.tracking = TrackingState::Degraded;
  c.baro_height = 11.0;
  ASSERT_TRUE(b.add_keyframe(c, KeyframeId{0}, step, std::nullopt).ok());
  b.optimize_all();
  const double dz = b.pose_odom(1).z() - b.pose_odom(0).z();
  // Information-weighted: 1 m x (1/0.1^2) / (1/0.1^2 + 1/0.2^2) = 0.8 m.
  EXPECT_NEAR(dz, 0.8, 0.05);
  // A keyframe without a barometric height gets no barometric factor: the odometry alone sets its
  // height.
  Backend nb(cfg);
  KeyframeInput a2 = a, c2 = c;
  c2.baro_height.reset();
  ASSERT_TRUE(nb.add_keyframe(a2, std::nullopt, gtsam::Pose3(), std::nullopt).ok());
  ASSERT_TRUE(nb.add_keyframe(c2, KeyframeId{0}, step, std::nullopt).ok());
  nb.optimize_all();
  EXPECT_NEAR(nb.pose_odom(1).z() - nb.pose_odom(0).z(), 0.0, 1e-6);
}

TEST(Backend, GravityPriorLevelsAKeyframeTheOdometryTilted) {
  BackendConfig cfg;  // odometry 0.004 rad (x10 when degraded = 0.04 rad), gravity 0.02 rad
  Backend b(cfg);
  ASSERT_TRUE(
      b.add_keyframe(kf(0, 0, gtsam::Pose3()), std::nullopt, gtsam::Pose3(), std::nullopt).ok());
  // The odometry says the robot rolled 0.1 rad; the IMU, quasi-static, says it is level.
  const gtsam::Pose3 step(gtsam::Rot3::Roll(0.1), Vector3d(1, 0, 0));
  KeyframeInput c = kf(1, 0, step);
  c.tracking = TrackingState::Degraded;
  c.up_body = Vector3d(0, 0, 1);
  ASSERT_TRUE(b.add_keyframe(c, KeyframeId{0}, step, std::nullopt).ok());
  b.optimize_all();
  const double roll = b.pose_odom(1).rotation().roll();
  // Information-weighted: 0.1 x (1/0.04^2) / (1/0.04^2 + 1/0.02^2) = 0.02 rad.
  EXPECT_NEAR(roll, 0.02, 0.01);
  // The same with the up direction of a body rolled by 0.1 rad: the odometry is confirmed, not
  // pulled.
  Backend t(cfg);
  ASSERT_TRUE(
      t.add_keyframe(kf(0, 0, gtsam::Pose3()), std::nullopt, gtsam::Pose3(), std::nullopt).ok());
  KeyframeInput d = c;
  d.up_body = gtsam::Rot3::Roll(0.1).unrotate(gtsam::Point3(0, 0, 1));
  ASSERT_TRUE(t.add_keyframe(d, KeyframeId{0}, step, std::nullopt).ok());
  t.optimize_all();
  EXPECT_NEAR(t.pose_odom(1).rotation().roll(), 0.1, 1e-3);
}
