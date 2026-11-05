#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <cmath>
#include <vector>

#include "strataslam/geometry/pose4.hpp"

namespace strataslam {

/// Exact error for a relative Pose4 measurement: Local(Z, Between(Xa, Xb)) with the full chain-rule
/// Jacobians (GTSAM's stock BetweenFactor approximates the Local Jacobian by the identity).
inline gtsam::Vector pose4_between_error(const Pose4& Xa, const Pose4& Xb, const Pose4& Z,
                                         const boost::optional<gtsam::Matrix&>& Ha,
                                         const boost::optional<gtsam::Matrix&>& Hb) {
  gtsam::Matrix44 Hb1, Hb2, Hl;
  const Pose4 h = Xa.between(Xb, Hb1, Hb2);
  const Pose4::Vector4 e = Z.localCoordinates(h, boost::none, Hl);
  if (Ha) *Ha = Hl * Hb1;
  if (Hb) *Hb = Hl * Hb2;
  return e;
}

/// Floor-to-floor relation measured through a transition (stairs, ramp, elevator, flight).
class TransitionFactor : public gtsam::NoiseModelFactor2<Pose4, Pose4> {
 public:
  TransitionFactor(gtsam::Key a, gtsam::Key b, const Pose4& measured,
                   const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor2<Pose4, Pose4>(model, a, b), z_(measured) {}
  gtsam::Vector evaluateError(const Pose4& Xa, const Pose4& Xb,
                              boost::optional<gtsam::Matrix&> Ha = boost::none,
                              boost::optional<gtsam::Matrix&> Hb = boost::none) const override {
    return pose4_between_error(Xa, Xb, z_, Ha, Hb);
  }
  const Pose4& measured() const { return z_; }
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new TransitionFactor(*this)));
  }

 private:
  Pose4 z_;
};

/// Relative pose from place recognition inside a verified overlap zone (same math, own type so the
/// report can tell the sources apart).
class CrossFloorMatchFactor : public TransitionFactor {
 public:
  using TransitionFactor::TransitionFactor;
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new CrossFloorMatchFactor(*this)));
  }
};

/// Height difference between two floor frames: error = (z_b - z_a) - dz. Base of the barometric
/// and prior height factors.
class HeightDifferenceFactor : public gtsam::NoiseModelFactor2<Pose4, Pose4> {
 public:
  HeightDifferenceFactor(gtsam::Key a, gtsam::Key b, double dz,
                         const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor2<Pose4, Pose4>(model, a, b), dz_(dz) {}
  gtsam::Vector evaluateError(const Pose4& Xa, const Pose4& Xb,
                              boost::optional<gtsam::Matrix&> Ha = boost::none,
                              boost::optional<gtsam::Matrix&> Hb = boost::none) const override {
    // X.retract(xi) = X exp(xi) leaves the rotation about z and adds R V rho: dz'/d rho_z = 1.
    if (Ha) {
      *Ha = gtsam::Matrix::Zero(1, 4);
      (*Ha)(0, 3) = -1.0;
    }
    if (Hb) {
      *Hb = gtsam::Matrix::Zero(1, 4);
      (*Hb)(0, 3) = 1.0;
    }
    return (gtsam::Vector(1) << (Xb.translation().z() - Xa.translation().z()) - dz_).finished();
  }
  double dz() const { return dz_; }
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new HeightDifferenceFactor(*this)));
  }

 private:
  double dz_;
};

class BaroHeightFactor : public HeightDifferenceFactor {
 public:
  using HeightDifferenceFactor::HeightDifferenceFactor;
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new BaroHeightFactor(*this)));
  }
};

/// Weak prior on the height between adjacent floors (`floor_height_prior` of the profile).
class FloorHeightPrior : public HeightDifferenceFactor {
 public:
  using HeightDifferenceFactor::HeightDifferenceFactor;
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new FloorHeightPrior(*this)));
  }
};

/// Point correspondences between two floor frames: structural points (elevator door frame, shaft
/// corners) or points marked by the user. Each pair says X_a p_a = X_b p_b in the map frame; with
/// `use_z = false` only x and y are compared (the same structure replicated on every floor sits at
/// a different height).
class AnchorFactor : public gtsam::NoiseModelFactor2<Pose4, Pose4> {
 public:
  AnchorFactor(gtsam::Key a, gtsam::Key b, std::vector<Eigen::Vector3d> pts_a,
               std::vector<Eigen::Vector3d> pts_b, bool use_z, double sigma)
      // `sigma` is the marking error of one point; the residual is the difference of two
      // independent marks, hence sqrt(2) * sigma (with sigma alone a correct set failed the GNC
      // inlier test).
      : gtsam::NoiseModelFactor2<Pose4, Pose4>(
            gtsam::noiseModel::Isotropic::Sigma(static_cast<size_t>((use_z ? 3 : 2) * pts_a.size()),
                                                std::sqrt(2.0) * sigma),
            a, b),
        pa_(std::move(pts_a)),
        pb_(std::move(pts_b)),
        use_z_(use_z) {}

  gtsam::Vector evaluateError(const Pose4& Xa, const Pose4& Xb,
                              boost::optional<gtsam::Matrix&> Ha = boost::none,
                              boost::optional<gtsam::Matrix&> Hb = boost::none) const override;
  size_t size_points() const { return pa_.size(); }
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new AnchorFactor(*this)));
  }

 private:
  std::vector<Eigen::Vector3d> pa_, pb_;
  bool use_z_;
};

}  // namespace strataslam
