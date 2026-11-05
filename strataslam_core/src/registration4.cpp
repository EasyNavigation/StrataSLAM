#include "strataslam/fusion/registration4.hpp"

#include <algorithm>
#include <cmath>

namespace strataslam {

Pose4 align_pose4_closed_form(const std::vector<PointPair>& pairs,
                              const std::vector<double>* weights, bool use_z) {
  double wsum = 0;
  Eigen::Vector3d ms = Eigen::Vector3d::Zero(), md = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < pairs.size(); ++i) {
    const double w = weights ? (*weights)[i] : 1.0;
    wsum += w;
    ms += w * pairs[i].src;
    md += w * pairs[i].dst;
  }
  if (wsum <= 0) return Pose4();
  ms /= wsum;
  md /= wsum;
  double dot = 0, cross = 0;
  for (size_t i = 0; i < pairs.size(); ++i) {
    const double w = weights ? (*weights)[i] : 1.0;
    const Eigen::Vector3d a = pairs[i].src - ms, b = pairs[i].dst - md;
    dot += w * (a.x() * b.x() + a.y() * b.y());
    cross += w * (a.x() * b.y() - a.y() * b.x());
  }
  const double theta = std::atan2(cross, dot);
  const Pose4 R(theta, Eigen::Vector3d::Zero());
  Eigen::Vector3d t = md - R.act(ms);
  if (!use_z) t.z() = md.z() - ms.z();
  return Pose4(theta, t);
}

RobustAlignment align_pose4_gnc(const std::vector<PointPair>& pairs, double c, bool use_z) {
  RobustAlignment out;
  const size_t n = pairs.size();
  out.inlier.assign(n, false);
  if (n < 2) return out;
  std::vector<double> w(n, 1.0);
  Pose4 T = align_pose4_closed_form(pairs, &w, use_z);
  const auto residual = [&](size_t i, const Pose4& X) {
    const Eigen::Vector3d d = X.act(pairs[i].src) - pairs[i].dst;
    return use_z ? d.norm() : d.head<2>().norm();
  };
  double r2max = 0;
  for (size_t i = 0; i < n; ++i) r2max = std::max(r2max, std::pow(residual(i, T), 2));
  const double c2 = c * c;
  double mu = std::max(1e-6, c2 / std::max(1e-12, 2.0 * r2max - c2));
  if (r2max <= c2) mu = 1e6;  // already consistent
  for (int it = 0; it < 200; ++it) {
    T = align_pose4_closed_form(pairs, &w, use_z);
    for (size_t i = 0; i < n; ++i) {
      const double r2 = std::pow(residual(i, T), 2);
      const double hi = (mu + 1.0) / mu * c2, lo = mu / (mu + 1.0) * c2;
      if (r2 >= hi)
        w[i] = 0.0;
      else if (r2 <= lo)
        w[i] = 1.0;
      else
        w[i] = std::clamp(c / std::sqrt(r2) * std::sqrt(mu * (mu + 1.0)) - mu, 0.0, 1.0);
    }
    if (mu > 1e5) break;
    mu *= 1.4;
  }
  T = align_pose4_closed_form(pairs, &w, use_z);
  out.transform = T;
  for (size_t i = 0; i < n; ++i) {
    out.inlier[i] = residual(i, T) <= 1.5 * c;
    out.n_inliers += out.inlier[i];
  }
  // Final non-robust refinement on the inlier set.
  std::vector<double> wi(n);
  for (size_t i = 0; i < n; ++i) wi[i] = out.inlier[i] ? 1.0 : 0.0;
  if (out.n_inliers >= 2) out.transform = align_pose4_closed_form(pairs, &wi, use_z);
  out.converged = out.n_inliers >= 2;
  return out;
}

}  // namespace strataslam
