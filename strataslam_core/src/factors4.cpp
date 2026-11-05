#include "strataslam/fusion/factors4.hpp"

namespace strataslam {

namespace {
/// d(X p)/d xi for the right perturbation X exp(xi): R_z(theta) * [[-p_y, 1, 0, 0], [p_x, 0, 1, 0], [0, 0, 0, 1]].
Eigen::Matrix<double, 3, 4> point_jacobian(const Pose4& X, const Eigen::Vector3d& p) {
  Eigen::Matrix<double, 3, 4> J0;
  J0 << -p.y(), 1, 0, 0, p.x(), 0, 1, 0, 0, 0, 0, 1;
  return X.rotation() * J0;
}
}  // namespace

gtsam::Vector AnchorFactor::evaluateError(const Pose4& Xa, const Pose4& Xb, boost::optional<gtsam::Matrix&> Ha,
                                          boost::optional<gtsam::Matrix&> Hb) const {
  const Eigen::Index d = use_z_ ? 3 : 2;
  const auto n = static_cast<Eigen::Index>(pa_.size());
  gtsam::Vector e(d * n);
  if (Ha) *Ha = gtsam::Matrix::Zero(d * n, 4);
  if (Hb) *Hb = gtsam::Matrix::Zero(d * n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::Vector3d diff = Xa.act(pa_[static_cast<size_t>(i)]) - Xb.act(pb_[static_cast<size_t>(i)]);
    e.segment(d * i, d) = diff.head(d);
    if (Ha) Ha->block(d * i, 0, d, 4) = point_jacobian(Xa, pa_[static_cast<size_t>(i)]).topRows(d);
    if (Hb) Hb->block(d * i, 0, d, 4) = -point_jacobian(Xb, pb_[static_cast<size_t>(i)]).topRows(d);
  }
  return e;
}

}  // namespace strataslam
