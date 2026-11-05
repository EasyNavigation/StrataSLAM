#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace strataslam {

/// Binary barometric factor between keyframes i and j (in the same gravity-aligned floor frame):
/// error = (z_j - z_i) - dh_ij. Binary and used over short windows so that slow atmospheric
/// drift does not leak into the map.
class BaroDeltaZFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
 public:
  BaroDeltaZFactor(gtsam::Key i, gtsam::Key j, double delta_h, const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model, i, j), delta_h_(delta_h) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& Xi, const gtsam::Pose3& Xj, boost::optional<gtsam::Matrix&> Hi = boost::none,
                              boost::optional<gtsam::Matrix&> Hj = boost::none) const override {
    // t' = t + R * rho under X.retract(xi): d z / d rho = R.row(2); rotations do not move t.
    if (Hi) {
      *Hi = gtsam::Matrix::Zero(1, 6);
      Hi->block<1, 3>(0, 3) = -Xi.rotation().matrix().row(2);
    }
    if (Hj) {
      *Hj = gtsam::Matrix::Zero(1, 6);
      Hj->block<1, 3>(0, 3) = Xj.rotation().matrix().row(2);
    }
    return (gtsam::Vector(1) << (Xj.z() - Xi.z()) - delta_h_).finished();
  }

  double delta_h() const { return delta_h_; }
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new BaroDeltaZFactor(*this)));
  }

 private:
  double delta_h_;
};

}  // namespace strataslam
