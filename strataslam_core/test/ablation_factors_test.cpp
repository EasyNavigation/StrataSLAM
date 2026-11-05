// Jacobians and exact-measurement zeros of the factors of the fusion ablation's other formulations (se3soft, euler),
// at random states including yaw near +-pi (audit A3-10 found them without tests).
#include <gtest/gtest.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/factorTesting.h>

#include <random>

#include "strataslam/fusion/ablation_factors.hpp"

using namespace strataslam;
using namespace strataslam::ablation;
using gtsam::Symbol;

namespace {

std::vector<Eigen::Vector3d> random_points(std::mt19937& rng, int n) {
  std::uniform_real_distribution<double> u(-5, 5);
  std::vector<Eigen::Vector3d> p;
  for (int i = 0; i < n; ++i) p.emplace_back(u(rng), u(rng), 0.3 * u(rng));
  return p;
}

}  // namespace

TEST(AblationFactors, JacobiansMatchFiniteDifferences) {
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(-3, 3), yaw(-M_PI, M_PI);
  const auto m1 = gtsam::noiseModel::Isotropic::Sigma(1, 0.1);
  const auto m4 = gtsam::noiseModel::Isotropic::Sigma(4, 0.1);
  for (int k = 0; k < 50; ++k) {
    // Near the +-pi cut every few states: the anchor factors use sin/cos and must be smooth there.
    const double ya = k % 5 == 0 ? M_PI - 1e-3 : yaw(rng), yb = k % 5 == 0 ? -M_PI + 1e-3 : yaw(rng);
    const gtsam::Pose3 A(gtsam::Rot3::Ypr(ya, 0.02 * u(rng), 0.02 * u(rng)), gtsam::Point3(u(rng), u(rng), u(rng)));
    const gtsam::Pose3 B(gtsam::Rot3::Ypr(yb, 0.02 * u(rng), 0.02 * u(rng)), gtsam::Point3(u(rng), u(rng), u(rng)));
    gtsam::Values p3;
    p3.insert(Symbol('f', 0), A);
    p3.insert(Symbol('f', 1), B);
    const gtsam::Vector4 va(u(rng), u(rng), u(rng), ya), vb(u(rng), u(rng), u(rng), yb);
    gtsam::Values v4;
    v4.insert(Symbol('f', 0), va);
    v4.insert(Symbol('f', 1), vb);
    const auto pa = random_points(rng, 3), pb = random_points(rng, 3);
    for (bool use_z : {false, true}) {
      EXPECT_TRUE(gtsam::internal::testFactorJacobians("Anchor3Factor", Anchor3Factor(Symbol('f', 0), Symbol('f', 1), pa, pb, use_z, 0.05), p3, 1e-6, 1e-5));
      EXPECT_TRUE(gtsam::internal::testFactorJacobians("EulerAnchorFactor", EulerAnchorFactor(Symbol('f', 0), Symbol('f', 1), pa, pb, use_z, 0.05), v4, 1e-6, 1e-5));
    }
    EXPECT_TRUE(gtsam::internal::testFactorJacobians("Height3Factor", Height3Factor(Symbol('f', 0), Symbol('f', 1), 2.5, m1), p3, 1e-6, 1e-5));
    EXPECT_TRUE(gtsam::internal::testFactorJacobians("EulerHeight", EulerHeight(Symbol('f', 0), Symbol('f', 1), 2.5, m1), v4, 1e-6, 1e-5));
    EXPECT_TRUE(gtsam::internal::testFactorJacobians("EulerBetween", EulerBetween(Symbol('f', 0), Symbol('f', 1), Pose4(0.3, Eigen::Vector3d(1, 2, 3)), m4), v4, 1e-6, 1e-5));
  }
}

TEST(AblationFactors, AnchorsOfAnExactTransformHaveZeroErrorInEveryFormulation) {
  std::mt19937 rng(3);
  // Floor b is floor a shifted by (4, -1, 3) and turned by 2.8 rad: points of b map onto the same world points.
  const double yaw = 2.8;
  const Eigen::Vector3d t(4, -1, 3);
  const auto pb = random_points(rng, 4);
  std::vector<Eigen::Vector3d> pa;
  const Pose4 Xb(yaw, t);
  for (const auto& p : pb) pa.push_back(Xb.act(p));  // floor a is the world frame
  gtsam::Values p3, v4;
  p3.insert(Symbol('f', 0), gtsam::Pose3());
  p3.insert(Symbol('f', 1), gtsam::Pose3(gtsam::Rot3::Yaw(yaw), t));
  v4.insert(Symbol('f', 0), gtsam::Vector4::Zero().eval());
  v4.insert(Symbol('f', 1), gtsam::Vector4(t.x(), t.y(), t.z(), yaw));
  EXPECT_LT(Anchor3Factor(Symbol('f', 0), Symbol('f', 1), pa, pb, true, 0.05).unwhitenedError(p3).norm(), 1e-12);
  EXPECT_LT(EulerAnchorFactor(Symbol('f', 0), Symbol('f', 1), pa, pb, true, 0.05).unwhitenedError(v4).norm(), 1e-12);
  // The same yaw written as 2.8 - 2 pi: the anchor residual does not care (no unwrapped angle difference).
  gtsam::Values v4w;
  v4w.insert(Symbol('f', 0), gtsam::Vector4::Zero().eval());
  v4w.insert(Symbol('f', 1), gtsam::Vector4(t.x(), t.y(), t.z(), yaw - 2 * M_PI));
  EXPECT_LT(EulerAnchorFactor(Symbol('f', 0), Symbol('f', 1), pa, pb, true, 0.05).unwhitenedError(v4w).norm(), 1e-12);
}
