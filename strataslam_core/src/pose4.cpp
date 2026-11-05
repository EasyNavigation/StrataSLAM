#include "strataslam/geometry/pose4.hpp"

#include <algorithm>
#include <iostream>
#include <ostream>
#include <sstream>

namespace strataslam {
namespace {

using Vector4 = Pose4::Vector4;
using Matrix4 = Pose4::Matrix4;

/// a = sin(phi)/phi, b = (1-cos(phi))/phi  (the entries of V(phi) = [[a,-b],[b,a]]).
void ab(double phi, double* a, double* b) {
  if (std::abs(phi) < Pose4::kSmallAngle) {
    const double p2 = phi * phi;
    *a = 1.0 - p2 / 6.0;
    *b = phi * (0.5 - p2 / 24.0);
  } else {
    const double s = std::sin(0.5 * phi);
    *a = std::sin(phi) / phi;
    *b = 2.0 * s * s / phi;  // stable form of (1 - cos(phi)) / phi
  }
}

/// c = (phi/2)·cot(phi/2); V(phi)^-1 = [[c, phi/2], [-phi/2, c]].
double half_cot(double phi) {
  if (std::abs(phi) < Pose4::kSmallAngle) {
    const double p2 = phi * phi;
    return 1.0 - p2 / 12.0 - p2 * p2 / 720.0;
  }
  return 0.5 * phi / std::tan(0.5 * phi);
}

/// d = (phi - sin(phi))/phi^2, e = (1 - cos(phi))/phi^2 (series for small phi to avoid
/// cancellation; the series is accurate to ~1e-14 below the 1e-2 switch).
void de(double phi, double* d, double* e) {
  if (std::abs(phi) < 1e-2) {
    const double p2 = phi * phi;
    *d = phi * (1.0 / 6.0 - p2 / 120.0 + p2 * p2 / 5040.0);
    *e = 0.5 - p2 / 24.0 + p2 * p2 / 720.0;
  } else {
    const double s = std::sin(0.5 * phi);
    *d = (phi - std::sin(phi)) / (phi * phi);
    *e = 2.0 * s * s / (phi * phi);
  }
}

Matrix4 left_jacobian(const Vector4& xi) {
  const double phi = xi(0), rx = xi(1), ry = xi(2);
  double a, b, d, e;
  ab(phi, &a, &b);
  de(phi, &d, &e);
  Matrix4 J = Matrix4::Identity();
  // First column: integral over s in [0,1] of the phi-column of Ad_exp(s·xi).
  J(1, 0) = rx * d + ry * e;
  J(2, 0) = ry * d - rx * e;
  J(1, 1) = a;
  J(1, 2) = -b;
  J(2, 1) = b;
  J(2, 2) = a;
  return J;
}

Matrix4 left_jacobian_inverse(const Vector4& xi) {
  const double phi = xi(0);
  const double c = half_cot(phi);
  Eigen::Matrix2d Vinv;
  Vinv << c, 0.5 * phi, -0.5 * phi, c;
  const Matrix4 J = left_jacobian(xi);
  Matrix4 Ji = Matrix4::Identity();
  Ji.block<2, 2>(1, 1) = Vinv;
  Ji.block<2, 1>(1, 0) = -Vinv * J.block<2, 1>(1, 0);
  return Ji;
}

}  // namespace

double Pose4::WrapAngle(double a) {
  if (a > -M_PI && a <= M_PI) return a;
  return a - 2.0 * M_PI * std::ceil((a - M_PI) / (2.0 * M_PI));
}

Result<Pose4> Pose4::FromRotationTranslation(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                                             double max_tilt_rad) {
  const double roll = std::atan2(R(2, 1), R(2, 2));
  const double pitch = std::asin(std::max(-1.0, std::min(1.0, -R(2, 0))));
  if (std::abs(roll) > max_tilt_rad || std::abs(pitch) > max_tilt_rad) {
    std::ostringstream os;
    os << "frame is not gravity-aligned: roll=" << roll << " rad, pitch=" << pitch
       << " rad exceed the " << max_tilt_rad << " rad limit";
    return Error{os.str()};
  }
  return Pose4(std::atan2(R(1, 0), R(0, 0)), t);
}

Result<Pose4> Pose4::FromPose3(const gtsam::Pose3& pose, double max_tilt_rad) {
  return FromRotationTranslation(pose.rotation().matrix(), pose.translation(), max_tilt_rad);
}

Eigen::Matrix3d Pose4::rotation() const {
  const double c = std::cos(theta_), s = std::sin(theta_);
  Eigen::Matrix3d R;
  R << c, -s, 0, s, c, 0, 0, 0, 1;
  return R;
}

Eigen::Matrix4d Pose4::matrix() const {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = rotation();
  T.block<3, 1>(0, 3) = t_;
  return T;
}

Pose4 Pose4::operator*(const Pose4& o) const { return Pose4(theta_ + o.theta_, act(o.t_)); }

Pose4 Pose4::inverse() const {
  const double c = std::cos(theta_), s = std::sin(theta_);
  // -R(-theta) t = -R^T t
  Eigen::Vector3d ti(-(c * t_.x() + s * t_.y()), -(-s * t_.x() + c * t_.y()), -t_.z());
  return Pose4(-theta_, ti);
}

Eigen::Vector3d Pose4::act(const Eigen::Vector3d& p) const {
  const double c = std::cos(theta_), s = std::sin(theta_);
  return {c * p.x() - s * p.y() + t_.x(), s * p.x() + c * p.y() + t_.y(), p.z() + t_.z()};
}

Pose4::Matrix4 Pose4::AdjointMap() const {
  const double c = std::cos(theta_), s = std::sin(theta_);
  Matrix4 Ad = Matrix4::Identity();
  Ad(1, 0) = t_.y();   // -J t
  Ad(2, 0) = -t_.x();
  Ad(1, 1) = c;
  Ad(1, 2) = -s;
  Ad(2, 1) = s;
  Ad(2, 2) = c;
  return Ad;
}

Pose4 Pose4::interpolate(const Pose4& other, double s) const {
  return *this * Expmap(s * Logmap(inverse() * other));
}

Pose4 Pose4::Expmap(const Vector4& xi, ChartJacobian H) {
  double a, b;
  ab(xi(0), &a, &b);
  const Eigen::Vector3d t(a * xi(1) - b * xi(2), b * xi(1) + a * xi(2), xi(3));
  if (H) *H = RightJacobian(xi);
  return Pose4(xi(0), t);
}

Pose4::Vector4 Pose4::Logmap(const Pose4& X, ChartJacobian H) {
  const double phi = X.theta_;
  const double c = half_cot(phi);
  Vector4 xi;
  xi << phi, c * X.t_.x() + 0.5 * phi * X.t_.y(), -0.5 * phi * X.t_.x() + c * X.t_.y(), X.t_.z();
  if (H) *H = RightJacobianInverse(xi);
  return xi;
}

Pose4::Matrix4 Pose4::LeftJacobian(const Vector4& xi) { return left_jacobian(xi); }
Pose4::Matrix4 Pose4::RightJacobian(const Vector4& xi) { return left_jacobian(-xi); }
Pose4::Matrix4 Pose4::LeftJacobianInverse(const Vector4& xi) { return left_jacobian_inverse(xi); }
Pose4::Matrix4 Pose4::RightJacobianInverse(const Vector4& xi) {
  return left_jacobian_inverse(-xi);
}

void Pose4::print(const std::string& s) const {
  std::cout << s << *this << '\n';
}

bool Pose4::equals(const Pose4& o, double tol) const {
  return std::abs(WrapAngle(theta_ - o.theta_)) <= tol && (t_ - o.t_).cwiseAbs().maxCoeff() <= tol;
}

std::ostream& operator<<(std::ostream& os, const Pose4& p) {
  return os << "Pose4(theta=" << p.theta() << ", t=[" << p.translation().transpose() << "])";
}

}  // namespace strataslam
