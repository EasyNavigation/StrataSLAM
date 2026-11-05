#pragma once

// Factors of the two formulations the fusion ablation compares with Pose4 (spec 12.5): `se3soft` (Pose3 with a soft
// roll/pitch prior) and `euler` (a plain (x, y, z, yaw) vector, no manifold). Public so that their Jacobians are
// tested (ablation_factors_test) like the Pose4 ones.

#include <cmath>
#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include "strataslam/geometry/pose4.hpp"

namespace strataslam::ablation {


/// Pose3 height difference (z of the translation), exact Jacobians.
class Height3Factor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
 public:
  Height3Factor(gtsam::Key a, gtsam::Key b, double dz, const gtsam::SharedNoiseModel& m)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(m, a, b), dz_(dz) {}
  gtsam::Vector evaluateError(const gtsam::Pose3& Xa, const gtsam::Pose3& Xb, boost::optional<gtsam::Matrix&> Ha = boost::none,
                              boost::optional<gtsam::Matrix&> Hb = boost::none) const override {
    if (Ha) { *Ha = gtsam::Matrix::Zero(1, 6); Ha->block<1, 3>(0, 3) = -Xa.rotation().matrix().row(2); }
    if (Hb) { *Hb = gtsam::Matrix::Zero(1, 6); Hb->block<1, 3>(0, 3) = Xb.rotation().matrix().row(2); }
    return (gtsam::Vector(1) << (Xb.z() - Xa.z()) - dz_).finished();
  }
 public:
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new Height3Factor(*this)));
  }
 private:
  double dz_;
};

/// Naive (x, y, z, yaw) relative-pose factor: no manifold, the angle difference is not wrapped.
class EulerBetween : public gtsam::NoiseModelFactor2<gtsam::Vector4, gtsam::Vector4> {
 public:
  EulerBetween(gtsam::Key a, gtsam::Key b, const Pose4& z, const gtsam::SharedNoiseModel& m)
      : gtsam::NoiseModelFactor2<gtsam::Vector4, gtsam::Vector4>(m, a, b), z_(z) {}
  gtsam::Vector evaluateError(const gtsam::Vector4& va, const gtsam::Vector4& vb, boost::optional<gtsam::Matrix&> Ha = boost::none,
                              boost::optional<gtsam::Matrix&> Hb = boost::none) const override {
    const double c = std::cos(va(3)), s = std::sin(va(3));
    const Eigen::Vector2d d(vb(0) - va(0), vb(1) - va(1));
    const Eigen::Vector2d rel(c * d.x() + s * d.y(), -s * d.x() + c * d.y());  // R(-yaw_a) d
    gtsam::Vector4 e;
    e << rel.x() - z_.translation().x(), rel.y() - z_.translation().y(), vb(2) - va(2) - z_.translation().z(), (vb(3) - va(3)) - z_.theta();
    // Euler ordering is (x, y, z, yaw); the whitening model uses the same ordering.
    if (Ha) {
      gtsam::Matrix44 J = gtsam::Matrix44::Zero();
      J(0, 0) = -c; J(0, 1) = -s; J(0, 3) = -s * d.x() + c * d.y();
      J(1, 0) = s; J(1, 1) = -c; J(1, 3) = -c * d.x() - s * d.y();
      J(2, 2) = -1; J(3, 3) = -1;
      *Ha = J;
    }
    if (Hb) {
      gtsam::Matrix44 J = gtsam::Matrix44::Zero();
      J(0, 0) = c; J(0, 1) = s; J(1, 0) = -s; J(1, 1) = c; J(2, 2) = 1; J(3, 3) = 1;
      *Hb = J;
    }
    return e;
  }
 public:
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new EulerBetween(*this)));
  }
 private:
  Pose4 z_;
};

class EulerHeight : public gtsam::NoiseModelFactor2<gtsam::Vector4, gtsam::Vector4> {
 public:
  EulerHeight(gtsam::Key a, gtsam::Key b, double dz, const gtsam::SharedNoiseModel& m)
      : gtsam::NoiseModelFactor2<gtsam::Vector4, gtsam::Vector4>(m, a, b), dz_(dz) {}
  gtsam::Vector evaluateError(const gtsam::Vector4& va, const gtsam::Vector4& vb, boost::optional<gtsam::Matrix&> Ha = boost::none,
                              boost::optional<gtsam::Matrix&> Hb = boost::none) const override {
    if (Ha) { *Ha = gtsam::Matrix::Zero(1, 4); (*Ha)(0, 2) = -1; }
    if (Hb) { *Hb = gtsam::Matrix::Zero(1, 4); (*Hb)(0, 2) = 1; }
    return (gtsam::Vector(1) << vb(2) - va(2) - dz_).finished();
  }
 public:
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new EulerHeight(*this)));
  }
 private:
  double dz_;
};


/// Anchor point correspondences between two floors for `se3soft`: the same residual as the Pose4 AnchorFactor
/// (X_a * p_a - X_b * p_b, horizontal only unless `use_z`, sigma sqrt(2) * marking error), on Pose3 variables.
/// Before 2026-11-04 the other formulations got the anchors as one pre-aligned pose with a heuristic yaw covariance,
/// so the ablation compared how the anchors entered, not the formulations (audit AUD-016, ADR-0049).
class Anchor3Factor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
 public:
  Anchor3Factor(gtsam::Key a, gtsam::Key b, std::vector<Eigen::Vector3d> pts_a, std::vector<Eigen::Vector3d> pts_b, bool use_z, double sigma)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(
            gtsam::noiseModel::Isotropic::Sigma(static_cast<size_t>((use_z ? 3 : 2) * pts_a.size()), std::sqrt(2.0) * sigma), a, b),
        pa_(std::move(pts_a)), pb_(std::move(pts_b)), use_z_(use_z) {}
  gtsam::Vector evaluateError(const gtsam::Pose3& Xa, const gtsam::Pose3& Xb, boost::optional<gtsam::Matrix&> Ha = boost::none,
                              boost::optional<gtsam::Matrix&> Hb = boost::none) const override {
    const Eigen::Index d = use_z_ ? 3 : 2;
    const auto n = static_cast<Eigen::Index>(pa_.size());
    gtsam::Vector e(d * n);
    if (Ha) *Ha = gtsam::Matrix::Zero(d * n, 6);
    if (Hb) *Hb = gtsam::Matrix::Zero(d * n, 6);
    for (Eigen::Index i = 0; i < n; ++i) {
      gtsam::Matrix36 Ja, Jb;
      const gtsam::Point3 wa = Xa.transformFrom(pa_[static_cast<size_t>(i)], Ha ? &Ja : nullptr);
      const gtsam::Point3 wb = Xb.transformFrom(pb_[static_cast<size_t>(i)], Hb ? &Jb : nullptr);
      e.segment(d * i, d) = (wa - wb).head(d);
      if (Ha) Ha->block(d * i, 0, d, 6) = Ja.topRows(d);
      if (Hb) Hb->block(d * i, 0, d, 6) = -Jb.topRows(d);
    }
    return e;
  }
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new Anchor3Factor(*this)));
  }

 private:
  std::vector<Eigen::Vector3d> pa_, pb_;
  bool use_z_;
};

/// The same anchor residual for `euler`, on (x, y, z, yaw) vectors: a point maps to (R(yaw) p_xy + t_xy, p_z + z).
class EulerAnchorFactor : public gtsam::NoiseModelFactor2<gtsam::Vector4, gtsam::Vector4> {
 public:
  EulerAnchorFactor(gtsam::Key a, gtsam::Key b, std::vector<Eigen::Vector3d> pts_a, std::vector<Eigen::Vector3d> pts_b, bool use_z, double sigma)
      : gtsam::NoiseModelFactor2<gtsam::Vector4, gtsam::Vector4>(
            gtsam::noiseModel::Isotropic::Sigma(static_cast<size_t>((use_z ? 3 : 2) * pts_a.size()), std::sqrt(2.0) * sigma), a, b),
        pa_(std::move(pts_a)), pb_(std::move(pts_b)), use_z_(use_z) {}
  gtsam::Vector evaluateError(const gtsam::Vector4& va, const gtsam::Vector4& vb, boost::optional<gtsam::Matrix&> Ha = boost::none,
                              boost::optional<gtsam::Matrix&> Hb = boost::none) const override {
    const Eigen::Index d = use_z_ ? 3 : 2;
    const auto n = static_cast<Eigen::Index>(pa_.size());
    gtsam::Vector e(d * n);
    if (Ha) *Ha = gtsam::Matrix::Zero(d * n, 4);
    if (Hb) *Hb = gtsam::Matrix::Zero(d * n, 4);
    for (Eigen::Index i = 0; i < n; ++i) {
      Eigen::Matrix<double, 3, 4> Ja, Jb;
      const Eigen::Vector3d wa = act(va, pa_[static_cast<size_t>(i)], &Ja), wb = act(vb, pb_[static_cast<size_t>(i)], &Jb);
      e.segment(d * i, d) = (wa - wb).head(d);
      if (Ha) Ha->block(d * i, 0, d, 4) = Ja.topRows(d);
      if (Hb) Hb->block(d * i, 0, d, 4) = -Jb.topRows(d);
    }
    return e;
  }
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new EulerAnchorFactor(*this)));
  }

 private:
  static Eigen::Vector3d act(const gtsam::Vector4& v, const Eigen::Vector3d& p, Eigen::Matrix<double, 3, 4>* J) {
    const double c = std::cos(v(3)), s = std::sin(v(3));
    if (J) {
      J->setZero();
      (*J)(0, 0) = 1; (*J)(1, 1) = 1; (*J)(2, 2) = 1;
      (*J)(0, 3) = -s * p.x() - c * p.y();
      (*J)(1, 3) = c * p.x() - s * p.y();
    }
    return {c * p.x() - s * p.y() + v(0), s * p.x() + c * p.y() + v(1), p.z() + v(2)};
  }
  std::vector<Eigen::Vector3d> pa_, pb_;
  bool use_z_;
};

}  // namespace strataslam::ablation
