#include <gtest/gtest.h>

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <random>

#include "strataslam/geometry/pose4.hpp"

using namespace strataslam;
using Vector4 = Pose4::Vector4;
using Matrix4 = Pose4::Matrix4;

namespace {

constexpr double kPi = M_PI;

struct Sampler {
  std::mt19937 rng{12345};  // fixed seed
  std::uniform_real_distribution<double> u{-1.0, 1.0};

  Vector4 xi(double max_phi = kPi - 1e-3, double max_rho = 10.0) {
    Vector4 v;
    v << u(rng) * max_phi, u(rng) * max_rho, u(rng) * max_rho, u(rng) * max_rho;
    return v;
  }
  Pose4 pose() {
    return Pose4(u(rng) * kPi, Eigen::Vector3d(u(rng) * 20, u(rng) * 20, u(rng) * 20));
  }
};

double angle_diff(double a, double b) { return std::abs(Pose4::WrapAngle(a - b)); }

double pose_err(const Pose4& a, const Pose4& b) {
  return std::max(angle_diff(a.theta(), b.theta()), (a.translation() - b.translation()).norm());
}

Matrix4 numeric_jacobian(const std::function<Vector4(const Vector4&)>& f, const Vector4& x,
                         double h = 1e-6) {
  Matrix4 J;
  for (int i = 0; i < 4; ++i) {
    Vector4 d = Vector4::Zero();
    d(i) = h;
    J.col(i) = (f(x + d) - f(x - d)) / (2 * h);
  }
  return J;
}

}  // namespace

// ---- group axioms and normalization ----

TEST(Pose4, AngleIsWrappedToHalfOpenInterval) {
  EXPECT_DOUBLE_EQ(Pose4(3 * kPi, {}).theta(), kPi);
  EXPECT_DOUBLE_EQ(Pose4(-kPi, {}).theta(), kPi);
  EXPECT_DOUBLE_EQ(Pose4(kPi, {}).theta(), kPi);
  EXPECT_NEAR(Pose4(kPi + 0.1, {}).theta(), -kPi + 0.1, 1e-15);
  EXPECT_NEAR(Pose4(-7.0, {}).theta(), -7.0 + 2 * kPi, 1e-15);
}

TEST(Pose4, IdentityInverseAndAssociativity) {
  Sampler s;
  for (int i = 0; i < 2000; ++i) {
    const Pose4 X = s.pose(), Y = s.pose(), Z = s.pose();
    EXPECT_LT(pose_err(X * Pose4::Identity(), X), 1e-12);
    EXPECT_LT(pose_err(Pose4::Identity() * X, X), 1e-12);
    EXPECT_LT(pose_err(X * X.inverse(), Pose4::Identity()), 1e-12);
    EXPECT_LT(pose_err(X.inverse() * X, Pose4::Identity()), 1e-12);
    EXPECT_LT(pose_err((X * Y) * Z, X * (Y * Z)), 1e-11);
  }
}

TEST(Pose4, ActionAndMatrixAgree) {
  Sampler s;
  for (int i = 0; i < 200; ++i) {
    const Pose4 X = s.pose(), Y = s.pose();
    const Eigen::Vector3d p(s.u(s.rng), s.u(s.rng), s.u(s.rng));
    const Eigen::Vector4d ph(p.x(), p.y(), p.z(), 1.0);
    EXPECT_LT(((X.matrix() * ph).head<3>() - X.act(p)).norm(), 1e-12);
    EXPECT_LT((X.matrix() * Y.matrix() - (X * Y).matrix()).norm(), 1e-12);
    EXPECT_LT((X.act(Y.act(p)) - (X * Y).act(p)).norm(), 1e-11);
  }
}

// ---- exp / log ----

TEST(Pose4, ExpLogRoundTrip10000Samples) {
  Sampler s;
  for (int i = 0; i < 10000; ++i) {
    Vector4 xi = s.xi();
    // Exercise the angle edge cases: near zero and near +-pi.
    if (i % 5 == 1) xi(0) = std::copysign(std::pow(10.0, -1.0 - (i % 17) * 0.8), xi(0));
    if (i % 5 == 2) xi(0) = std::copysign(kPi - std::pow(10.0, -(i % 12)) , xi(0));
    if (i % 5 == 3) xi(0) = 0.0;
    const Pose4 X = Pose4::Expmap(xi);
    const Vector4 back = Pose4::Logmap(X);
    ASSERT_LT((back - xi).cwiseAbs().maxCoeff(), 1e-9) << "xi=" << xi.transpose();
    ASSERT_LT(pose_err(Pose4::Expmap(back), X), 1e-9);
  }
}

TEST(Pose4, LogExpRoundTripFromPoses) {
  Sampler s;
  for (int i = 0; i < 10000; ++i) {
    const Pose4 X = s.pose();
    ASSERT_LT(pose_err(Pose4::Expmap(Pose4::Logmap(X)), X), 1e-9);
  }
  // Poses with theta exactly at, and just beside, +-pi.
  for (double th : {kPi, -kPi, kPi - 1e-12, -kPi + 1e-12, 0.0, 1e-300}) {
    const Pose4 X(th, Eigen::Vector3d(1.5, -2.5, 3.5));
    EXPECT_LT(pose_err(Pose4::Expmap(Pose4::Logmap(X)), X), 1e-9) << th;
  }
}

TEST(Pose4, TaylorBranchMatchesHighPrecisionClosedForm) {
  const Eigen::Vector3d rho(1.3, -0.7, 0.4);
  for (double phi : {1e-12, 1e-8, 1e-6, 9e-6, 1.1e-5, 1e-4, 1e-3, -9e-6, -1.1e-5}) {
    Vector4 xi;
    xi << phi, rho.x(), rho.y(), rho.z();
    const Pose4 X = Pose4::Expmap(xi);
    const long double p = phi;
    const long double sh = sinl(p / 2), a = sinl(p) / p, b = 2.0L * sh * sh / p;
    const long double tx = a * rho.x() - b * rho.y(), ty = b * rho.x() + a * rho.y();
    EXPECT_NEAR(X.translation().x(), (double)tx, 1e-13) << phi;
    EXPECT_NEAR(X.translation().y(), (double)ty, 1e-13) << phi;
  }
}

TEST(Pose4, ContinuityAcrossTaylorThreshold) {
  for (double sign : {1.0, -1.0}) {
    const double phi0 = sign * Pose4::kSmallAngle;
    Vector4 lo, hi;
    lo << phi0 * (1 - 1e-7), 3.0, -2.0, 1.0;
    hi << phi0 * (1 + 1e-7), 3.0, -2.0, 1.0;
    EXPECT_LT(pose_err(Pose4::Expmap(lo), Pose4::Expmap(hi)), 1e-5 * 1e-6 + 1e-9);
    EXPECT_LT((Pose4::Logmap(Pose4::Expmap(lo)) - lo).norm(), 1e-9);
    EXPECT_LT((Pose4::Logmap(Pose4::Expmap(hi)) - hi).norm(), 1e-9);
    EXPECT_LT((Pose4::RightJacobian(lo) - Pose4::RightJacobian(hi)).norm(), 1e-9);
    EXPECT_LT((Pose4::LeftJacobianInverse(lo) - Pose4::LeftJacobianInverse(hi)).norm(), 1e-9);
  }
  // Same check directly on the Logmap input pose (theta on both sides of the threshold).
  const Eigen::Vector3d t(2.0, -1.0, 0.5);
  const Vector4 l1 = Pose4::Logmap(Pose4(Pose4::kSmallAngle * (1 - 1e-9), t));
  const Vector4 l2 = Pose4::Logmap(Pose4(Pose4::kSmallAngle * (1 + 1e-9), t));
  EXPECT_LT((l1 - l2).norm(), 1e-9);
}

TEST(Pose4, SmallRotationIsPureTranslationLimit) {
  Vector4 xi;
  xi << 0.0, 1.0, 2.0, 3.0;
  const Pose4 X = Pose4::Expmap(xi);
  EXPECT_EQ(X.theta(), 0.0);
  EXPECT_EQ(X.translation(), Eigen::Vector3d(1, 2, 3));
}

// ---- adjoint ----

TEST(Pose4, AdjointIdentity) {
  Sampler s;
  for (int i = 0; i < 2000; ++i) {
    const Pose4 X = s.pose();
    const Vector4 xi = s.xi(2.0, 3.0);
    const Pose4 lhs = X * Pose4::Expmap(xi);
    const Pose4 rhs = Pose4::Expmap(X.AdjointMap() * xi) * X;
    ASSERT_LT(pose_err(lhs, rhs), 1e-10);
  }
}

// ---- Jacobians vs central differences ----

TEST(Pose4, RightJacobianMatchesFiniteDifferences) {
  Sampler s;
  for (int i = 0; i < 300; ++i) {
    Vector4 xi = s.xi(3.0, 5.0);
    if (i % 4 == 0) xi(0) *= 1e-7;  // Taylor branch
    const auto f = [&](const Vector4& x) { return Pose4::Logmap(Pose4::Expmap(xi).inverse() * Pose4::Expmap(x)); };
    ASSERT_LT((numeric_jacobian(f, xi) - Pose4::RightJacobian(xi)).cwiseAbs().maxCoeff(), 1e-6);
  }
}

TEST(Pose4, LeftJacobianMatchesFiniteDifferences) {
  Sampler s;
  for (int i = 0; i < 300; ++i) {
    Vector4 xi = s.xi(3.0, 5.0);
    if (i % 4 == 0) xi(0) *= 1e-7;
    const auto f = [&](const Vector4& x) { return Pose4::Logmap(Pose4::Expmap(x) * Pose4::Expmap(xi).inverse()); };
    ASSERT_LT((numeric_jacobian(f, xi) - Pose4::LeftJacobian(xi)).cwiseAbs().maxCoeff(), 1e-6);
  }
}

TEST(Pose4, JacobianInversesAndRelations) {
  Sampler s;
  for (int i = 0; i < 500; ++i) {
    Vector4 xi = s.xi(3.0, 5.0);
    if (i % 4 == 0) xi(0) *= 1e-7;
    EXPECT_LT((Pose4::RightJacobian(xi) * Pose4::RightJacobianInverse(xi) - Matrix4::Identity()).norm(), 1e-9);
    EXPECT_LT((Pose4::LeftJacobian(xi) * Pose4::LeftJacobianInverse(xi) - Matrix4::Identity()).norm(), 1e-9);
    // J_l = Ad_exp(xi) J_r
    EXPECT_LT((Pose4::LeftJacobian(xi) - Pose4::Expmap(xi).AdjointMap() * Pose4::RightJacobian(xi)).norm(), 1e-9);
    // J_r(xi) = J_l(-xi)
    EXPECT_LT((Pose4::RightJacobian(xi) - Pose4::LeftJacobian(-xi)).norm(), 1e-12);
  }
}

TEST(Pose4, LogmapDerivativeMatchesFiniteDifferences) {
  Sampler s;
  for (int i = 0; i < 200; ++i) {
    const Pose4 X = s.pose();
    const auto g = [&](const Vector4& d) { return Pose4::Logmap(X * Pose4::Expmap(d)); };
    const Matrix4 num = numeric_jacobian(g, Vector4::Zero());
    ASSERT_LT((num - Pose4::LogmapDerivative(X)).cwiseAbs().maxCoeff(), 1e-6);
  }
}

// ---- Pose3 -> Pose4 ----

TEST(Pose4, FromPose3LevelFrameGivesExactYaw) {
  for (double yaw : {0.0, 0.3, -1.2, 3.0, -3.1}) {
    const gtsam::Pose3 P(gtsam::Rot3::Yaw(yaw), gtsam::Point3(1, 2, 3));
    const auto r = Pose4::FromPose3(P);
    ASSERT_TRUE(r.ok());
    EXPECT_NEAR(r.value().theta(), yaw, 1e-14);
    EXPECT_EQ(r.value().translation(), Eigen::Vector3d(1, 2, 3));
  }
}

TEST(Pose4, FromPose3RejectsTiltedFrames) {
  const double deg = kPi / 180.0;
  const gtsam::Point3 t(0, 0, 0);
  EXPECT_FALSE(Pose4::FromPose3(gtsam::Pose3(gtsam::Rot3::RzRyRx(5 * deg, 0, 0.3), t)).ok());   // roll 5°
  EXPECT_FALSE(Pose4::FromPose3(gtsam::Pose3(gtsam::Rot3::RzRyRx(0, 5 * deg, 0.3), t)).ok());   // pitch 5°
  EXPECT_FALSE(Pose4::FromPose3(gtsam::Pose3(gtsam::Rot3::RzRyRx(-5 * deg, 0, 0), t)).ok());
  EXPECT_TRUE(Pose4::FromPose3(gtsam::Pose3(gtsam::Rot3::RzRyRx(1 * deg, 1 * deg, 0.3), t)).ok());
  // The threshold is configurable.
  EXPECT_TRUE(Pose4::FromPose3(gtsam::Pose3(gtsam::Rot3::RzRyRx(5 * deg, 0, 0.3), t), 6 * deg).ok());
  const auto err = Pose4::FromPose3(gtsam::Pose3(gtsam::Rot3::RzRyRx(5 * deg, 0, 0), t));
  EXPECT_NE(err.error().message.find("not gravity-aligned"), std::string::npos);
}

// ---- interpolation ----

TEST(Pose4, Interpolation) {
  Sampler s;
  for (int i = 0; i < 200; ++i) {
    const Pose4 X = s.pose(), Y = s.pose();
    EXPECT_LT(pose_err(X.interpolate(Y, 0.0), X), 1e-10);
    EXPECT_LT(pose_err(X.interpolate(Y, 1.0), Y), 1e-10);
    const Pose4 M = X.interpolate(Y, 0.5);
    // Half-way pose is equidistant (in the tangent space) from both ends.
    EXPECT_LT((Pose4::Logmap(X.inverse() * M) - Pose4::Logmap(M.inverse() * Y)).norm(), 1e-9);
  }
}

// ---- GTSAM integration ----

TEST(Pose4Gtsam, TraitsAndManifoldOps) {
  static_assert(gtsam::traits<Pose4>::dimension == 4, "Pose4 must be 4-dimensional");
  Sampler s;
  for (int i = 0; i < 100; ++i) {
    const Pose4 X = s.pose(), Y = s.pose();
    const Vector4 v = s.xi(2.5, 4.0);
    EXPECT_LT(pose_err(X.retract(v), X * Pose4::Expmap(v)), 1e-12);
    EXPECT_LT((X.localCoordinates(Y) - Pose4::Logmap(X.inverse() * Y)).norm(), 1e-12);
    EXPECT_LT(pose_err(gtsam::traits<Pose4>::Retract(X, v), X.retract(v)), 1e-12);
    // Jacobians reported by the library agree with numerical derivatives.
    gtsam::Matrix44 H1, H2;
    X.compose(Y, H1, H2);
    const auto cmp = [&](const Pose4& a, const Pose4& b) { return a.compose(b); };
    EXPECT_LT((gtsam::numericalDerivative21<Pose4, Pose4, Pose4>(cmp, X, Y) - H1).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_LT((gtsam::numericalDerivative22<Pose4, Pose4, Pose4>(cmp, X, Y) - H2).cwiseAbs().maxCoeff(), 1e-6);
    X.between(Y, H1, H2);
    const auto bet = [&](const Pose4& a, const Pose4& b) { return a.between(b); };
    EXPECT_LT((gtsam::numericalDerivative21<Pose4, Pose4, Pose4>(bet, X, Y) - H1).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_LT((gtsam::numericalDerivative22<Pose4, Pose4, Pose4>(bet, X, Y) - H2).cwiseAbs().maxCoeff(), 1e-6);
  }
}

TEST(Pose4Gtsam, FactorJacobiansAreCorrectAtZeroError) {
  // GTSAM's stock BetweenFactor/PriorFactor use J_r^-1(error) ≈ I unless built with
  // GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR, so their Jacobians are exact only at zero error.
  using gtsam::Symbol;
  const Pose4 X1(0.4, Eigen::Vector3d(1, 2, 3)), X2(-1.1, Eigen::Vector3d(-2, 0.5, 4));
  gtsam::Values v;
  v.insert(Symbol('x', 1), X1);
  v.insert(Symbol('x', 2), X2);
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(4, 0.1);
  gtsam::BetweenFactor<Pose4> between(Symbol('x', 1), Symbol('x', 2), X1.between(X2), noise);
  EXPECT_LT(between.unwhitenedError(v).norm(), 1e-12);
  EXPECT_TRUE(gtsam::internal::testFactorJacobians("BetweenFactor<Pose4>", between, v, 1e-5, 1e-6));
  gtsam::PriorFactor<Pose4> prior(Symbol('x', 1), X1, noise);
  EXPECT_LT(prior.unwhitenedError(v).norm(), 1e-12);
  EXPECT_TRUE(gtsam::internal::testFactorJacobians("PriorFactor<Pose4>", prior, v, 1e-5, 1e-6));
}

TEST(Pose4Gtsam, ExactErrorChainJacobiansAtNonzeroError) {
  // d/dx1 and d/dx2 of Local(meas, Between(x1, x2)) with full chain rule, away from zero error.
  Sampler s;
  for (int i = 0; i < 100; ++i) {
    const Pose4 X1 = s.pose(), X2 = s.pose(), meas = s.pose();
    const auto err = [&](const Pose4& a, const Pose4& b) {
      return gtsam::traits<Pose4>::Local(meas, a.between(b));
    };
    gtsam::Matrix44 Hb1, Hb2, Hl;
    const Pose4 hx = X1.between(X2, Hb1, Hb2);
    meas.localCoordinates(hx, boost::none, Hl);
    const gtsam::Matrix44 A1 = Hl * Hb1, A2 = Hl * Hb2;
    EXPECT_LT((gtsam::numericalDerivative21<gtsam::Vector4, Pose4, Pose4>(err, X1, X2) - A1).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_LT((gtsam::numericalDerivative22<gtsam::Vector4, Pose4, Pose4>(err, X1, X2) - A2).cwiseAbs().maxCoeff(), 1e-6);
  }
}

TEST(Pose4Gtsam, PoseGraphConvergesAcrossTheAngleWrap) {
  using gtsam::Symbol;
  // Chain of poses turning past +-pi, a loop closure, and a perturbed initial guess.
  std::vector<Pose4> truth{Pose4::Identity()};
  const Pose4 step(0.9, Eigen::Vector3d(1.0, 0.2, 0.1));
  for (int i = 1; i < 12; ++i) truth.push_back(truth.back() * step);

  gtsam::NonlinearFactorGraph g;
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(4, 0.05);
  g.add(gtsam::PriorFactor<Pose4>(Symbol('x', 0), truth[0], noise));
  for (size_t i = 1; i < truth.size(); ++i) {
    g.add(gtsam::BetweenFactor<Pose4>(Symbol('x', i - 1), Symbol('x', i), truth[i - 1].between(truth[i]), noise));
  }
  g.add(gtsam::BetweenFactor<Pose4>(Symbol('x', 0), Symbol('x', 11), truth[0].between(truth[11]), noise));

  Sampler s;
  gtsam::Values init;
  for (size_t i = 0; i < truth.size(); ++i) {
    init.insert(Symbol('x', i), truth[i].retract(0.2 * s.xi(1.0, 1.0)));
  }
  const gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(g, init).optimize();
  for (size_t i = 0; i < truth.size(); ++i) {
    EXPECT_LT(pose_err(result.at<Pose4>(Symbol('x', i)), truth[i]), 1e-6) << i;
  }
}
