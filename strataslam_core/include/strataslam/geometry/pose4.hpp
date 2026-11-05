#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/geometry/Pose3.h>

#include <Eigen/Core>
#include <cmath>
#include <iosfwd>
#include <string>
#include <utility>

#include "strataslam/common/result.hpp"

namespace strataslam {

/// Element of SE(2)×R: a yaw rotation theta about z and a 3D translation t.
/// Acts on points as x' = Rz(theta) x + t. Tangent vector xi = (phi, rho_x, rho_y, rho_z).
/// Notation follows Solà et al., "A micro Lie theory for state estimation in robotics";
/// derivation in docs/math/pose4.md. Retraction is right-multiplicative: X.retract(xi) = X·exp(xi).
class Pose4 : public gtsam::LieGroup<Pose4, 4> {
 public:
  using Vector4 = Eigen::Matrix<double, 4, 1>;
  using Matrix4 = Eigen::Matrix<double, 4, 4>;

  /// |phi| below which exp/log use Taylor expansions instead of dividing by phi.
  static constexpr double kSmallAngle = 1e-5;
  /// Default tilt (roll or pitch) beyond which a Pose3 is not considered gravity-aligned.
  static constexpr double kDefaultMaxTiltRad = 2.0 * M_PI / 180.0;

  Pose4() : theta_(0.0), t_(Eigen::Vector3d::Zero()) {}
  /// `theta` is wrapped to (-pi, pi].
  Pose4(double theta, const Eigen::Vector3d& t) : theta_(WrapAngle(theta)), t_(t) {}

  static Pose4 Identity() { return Pose4(); }
  /// Wraps an angle to (-pi, pi].
  static double WrapAngle(double a);

  /// Conversion from a gravity-aligned pose. Fails if |roll| or |pitch| exceed `max_tilt_rad`.
  static Result<Pose4> FromPose3(const gtsam::Pose3& pose,
                                 double max_tilt_rad = kDefaultMaxTiltRad);
  static Result<Pose4> FromRotationTranslation(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                                               double max_tilt_rad = kDefaultMaxTiltRad);

  double theta() const { return theta_; }
  const Eigen::Vector3d& translation() const { return t_; }
  Eigen::Matrix3d rotation() const;
  /// Homogeneous 4x4 matrix of the equivalent SE(3) transform.
  Eigen::Matrix4d matrix() const;

  // ---- group ----
  Pose4 operator*(const Pose4& other) const;
  Pose4 inverse() const;
  using gtsam::LieGroup<Pose4, 4>::inverse;  // version with Jacobian
  /// Transforms a point from the local frame to the parent frame.
  Eigen::Vector3d act(const Eigen::Vector3d& p) const;
  /// Ad_X: X·exp(xi) = exp(Ad_X xi)·X.
  Matrix4 AdjointMap() const;
  /// X·exp(s·log(X^-1 Y)), s in [0, 1] for interpolation.
  Pose4 interpolate(const Pose4& other, double s) const;

  // ---- Lie algebra ----
  static Pose4 Expmap(const Vector4& xi, ChartJacobian H = boost::none);
  static Vector4 Logmap(const Pose4& X, ChartJacobian H = boost::none);

  /// Left/right Jacobians of exp: exp(xi+d) ≈ exp(J_l d)·exp(xi) ≈ exp(xi)·exp(J_r d).
  static Matrix4 LeftJacobian(const Vector4& xi);
  static Matrix4 RightJacobian(const Vector4& xi);
  static Matrix4 LeftJacobianInverse(const Vector4& xi);
  static Matrix4 RightJacobianInverse(const Vector4& xi);
  /// GTSAM naming: derivative of Expmap (right Jacobian) and of Logmap (its inverse at log X).
  static Matrix4 ExpmapDerivative(const Vector4& xi) { return RightJacobian(xi); }
  static Matrix4 LogmapDerivative(const Pose4& X) { return RightJacobianInverse(Logmap(X)); }

  struct ChartAtOrigin {
    // Signatures fixed by GTSAM's LieGroup traits (ChartJacobian is passed by value there).
    static Pose4 Retract(const Vector4& xi, ChartJacobian H = boost::none) {
      return Expmap(xi, std::move(H));
    }
    static Vector4 Local(const Pose4& X, ChartJacobian H = boost::none) {
      return Logmap(X, std::move(H));
    }
  };

  // ---- Testable ----
  void print(const std::string& s = "") const;
  bool equals(const Pose4& other, double tol = 1e-9) const;

 private:
  double theta_;
  Eigen::Vector3d t_;
};

std::ostream& operator<<(std::ostream& os, const Pose4& p);

}  // namespace strataslam

namespace gtsam {
template <>
struct traits<strataslam::Pose4> : public internal::LieGroup<strataslam::Pose4> {};
template <>
struct traits<const strataslam::Pose4> : public internal::LieGroup<strataslam::Pose4> {};
}  // namespace gtsam
