#include <gtest/gtest.h>

#include <random>

#include "strataslam/evaluation/metrics.hpp"

using namespace strataslam;
using namespace strataslam::eval;

namespace {

std::vector<TimedPose> line(int n, double speed = 1.0, double dt_s = 0.1) {
  std::vector<TimedPose> t;
  for (int i = 0; i < n; ++i) {
    const double s = i * dt_s;
    // Gentle curve so that alignment is well-conditioned in 3D.
    gtsam::Pose3 p(gtsam::Rot3::Yaw(0.2 * s), Eigen::Vector3d(speed * s, 2.0 * std::sin(0.3 * s), 0.3 * std::cos(0.2 * s)));
    t.push_back({static_cast<Timestamp>(i * dt_s * 1e9), p});
  }
  return t;
}

std::vector<TimedPose> transformed(const std::vector<TimedPose>& in, const gtsam::Pose3& T) {
  auto out = in;
  for (auto& p : out) p.pose = T * p.pose;
  return out;
}

}  // namespace

TEST(Metrics, AteIsZeroForIdenticalAndForRigidlyMovedWithAlignment) {
  const auto gt = line(200);
  EXPECT_NEAR(ate(gt, gt, AlignMode::None).rmse, 0.0, 1e-12);
  const gtsam::Pose3 T(gtsam::Rot3::RzRyRx(0.3, -0.2, 1.1), Eigen::Vector3d(5, -3, 2));
  const auto est = transformed(gt, T.inverse());
  EXPECT_GT(ate(est, gt, AlignMode::None).rmse, 1.0);
  EXPECT_NEAR(ate(est, gt, AlignMode::SE3).rmse, 0.0, 1e-9);
}

TEST(Metrics, AteKnownNoiseLevel) {
  const auto gt = line(200);
  auto est = gt;
  for (size_t i = 0; i < est.size(); ++i) {
    est[i].pose = gtsam::Pose3(est[i].pose.rotation(), Eigen::Vector3d(est[i].pose.translation() + Eigen::Vector3d(0, 0, i % 2 ? 0.05 : -0.05)));
  }
  const auto r = ate(est, gt, AlignMode::SE3);
  EXPECT_NEAR(r.rmse, 0.05, 2e-3);
  EXPECT_NEAR(r.max, 0.05, 2e-3);
  EXPECT_EQ(r.n, 200u);
}

TEST(Metrics, FourDofAlignmentRecoversPose4ExactlyAndIgnoresTilt) {
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> u(-5, 5);
  std::vector<Eigen::Vector3d> p, q;
  const Pose4 X(0.7, Eigen::Vector3d(3, -2, 1.5));
  for (int i = 0; i < 4; ++i) {
    p.emplace_back(u(rng), u(rng), u(rng));
    q.push_back(X.act(p.back()));
  }
  const Pose4 est = align_4dof(p, q);
  EXPECT_LT(std::abs(Pose4::WrapAngle(est.theta() - X.theta())), 1e-9);
  EXPECT_LT((est.translation() - X.translation()).norm(), 1e-9);

  const auto gt = line(100);
  const auto est_traj = transformed(gt, gtsam::Pose3(gtsam::Rot3::Yaw(1.0), Eigen::Vector3d(1, 2, 3)).inverse());
  EXPECT_NEAR(ate(est_traj, gt, AlignMode::FourDof).rmse, 0.0, 1e-9);
}

TEST(Metrics, RpeConstantScaleDrift) {
  const auto gt = line(100);
  std::vector<TimedPose> est = gt;
  for (auto& p : est) p.pose = gtsam::Pose3(p.pose.rotation(), Eigen::Vector3d(p.pose.translation() * 1.1));
  const auto r = rpe(est, gt, static_cast<int64_t>(1e9));
  // Translation over 1 s is ~1 m (+0.1 m per metre travelled): error is ~0.1 m.
  EXPECT_NEAR(r.trans_rmse, 0.1, 0.02);
  EXPECT_GT(r.n, 50u);
  EXPECT_LT(r.rot_rmse_deg, 1e-6);
}

TEST(Metrics, CollapseRate) {
  EXPECT_EQ(collapse_rate({}), 0.0);
  std::vector<LoopRecord> loops{{1, 2, 0, 0}, {3, 4, 1, 1}, {5, 6, 0, 1}};
  EXPECT_NEAR(collapse_rate(loops), 1.0 / 3.0, 1e-12);
  EXPECT_EQ(collapse_rate({{1, 2, 0, 0}, {3, 4, 1, 1}}), 0.0);
  EXPECT_EQ(collapse_rate({{1, 2, 0, -1}}), 1.0);
}

TEST(Metrics, FloorHeightAndFusionErrors) {
  const auto h = floor_height_errors({{0, {1.2, 1.4}}, {1, {4.1, 4.3}}}, {{0, 1.3}, {1, 4.5}});
  EXPECT_NEAR(h.at(0), 0.0, 1e-12);
  EXPECT_NEAR(h.at(1), 0.3, 1e-12);
  const Pose4 t(0.5, Eigen::Vector3d(1, 2, 3));
  const Pose4 e(0.5 + 0.02, Eigen::Vector3d(1.0, 2.0, 3.05));
  const auto f = fusion_errors({{1, e}}, {{1, t}});
  EXPECT_NEAR(f.at(1).rot_deg, 0.02 * 180 / M_PI, 1e-9);
  EXPECT_GT(f.at(1).trans_m, 0.04);
  EXPECT_LT(f.at(1).trans_m, 0.08);
}

TEST(Metrics, ReportFormats) {
  Report r{"S01", {{"ate_rmse", 0.0123}, {"loops", 2}}};
  EXPECT_NE(r.to_markdown().find("| ate_rmse | 0.0123 |"), std::string::npos);
  EXPECT_EQ(r.to_json(), "{\"title\": \"S01\", \"metrics\": {\"ate_rmse\": 0.0123, \"loops\": 2}}");
}

TEST(Metrics, TumStampsKeepEveryNanosecond) {
  EXPECT_EQ(strataslam::eval::tum_stamp(1305031102175304000LL), "1305031102.175304000");
  EXPECT_EQ(strataslam::eval::tum_stamp(5), "0.000000005");
  EXPECT_EQ(strataslam::eval::tum_stamp(-1500000000LL), "-1.500000000");
}
