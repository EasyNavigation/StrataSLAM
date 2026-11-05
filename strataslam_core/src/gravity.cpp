#include "strataslam/geometry/gravity.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <random>

#include "strataslam/sensors/barometer.hpp"

namespace strataslam {

std::optional<GravityFromImu> estimate_gravity_from_imu(const std::vector<ImuSample>& s,
                                                        double max_gyro, double accel_tol) {
  if (s.size() < 20) return std::nullopt;
  Eigen::Vector3d a = Eigen::Vector3d::Zero(), w = Eigen::Vector3d::Zero();
  for (const auto& m : s) {
    a += m.accel;
    w += m.gyro;
  }
  a /= s.size();
  w /= s.size();
  double gyro_dev = 0;
  for (const auto& m : s) gyro_dev = std::max(gyro_dev, (m.gyro - w).norm());
  if (w.norm() > max_gyro || gyro_dev > 10 * max_gyro) return std::nullopt;
  if (std::abs(a.norm() - kStandardGravity) > accel_tol) return std::nullopt;
  return GravityFromImu{a.normalized(), a.norm()};
}

gtsam::Rot3 rotation_from_up(const Eigen::Vector3d& up_body) {
  const Eigen::Vector3d u = up_body.normalized();
  // Shortest rotation R with R u = (0, 0, 1) (Rodrigues; the antiparallel case is a half turn about
  // x).
  const Eigen::Vector3d axis = u.cross(Eigen::Vector3d::UnitZ());
  const double s = axis.norm(), c = u.z();
  if (s < 1e-12) return c > 0 ? gtsam::Rot3() : gtsam::Rot3::Rx(M_PI);
  return gtsam::Rot3::Expmap(axis / s * std::atan2(s, c));
}

gtsam::Rot3 level_to(const gtsam::Rot3& R_world_body, const Eigen::Vector3d& up_body) {
  const gtsam::Rot3 level = rotation_from_up(up_body);  // the right tilt, an arbitrary heading
  const auto heading = [](const gtsam::Rot3& R) {
    const auto M = R.matrix();
    return std::atan2(M(1, 0), M(0, 0));
  };
  return gtsam::Rot3::Yaw(heading(R_world_body) - heading(level)) * level;
}

std::optional<Eigen::Vector3d> estimate_up_from_ground_plane(
    const std::vector<Eigen::Vector3f>& pts, const Eigen::Vector3d& expected_up, uint64_t seed,
    double max_tilt_deg, double thr, int iterations) {
  if (pts.size() < 100) return std::nullopt;
  const Eigen::Vector3d up = expected_up.normalized();
  // The ground is below the sensor. Looking for it only there keeps RANSAC away from the slabs of
  // wall that a horizontal-ish plane cuts at any height (they passed for the ground when the floor
  // was a tenth of a corridor scan, audit AUD-022), and makes the floor a larger share of the
  // candidates, so that it is actually drawn.
  constexpr double kMinBelow = 0.3;  // m under the sensor, along the expected up
  std::vector<Eigen::Vector3d> low;
  low.reserve(pts.size());
  for (const auto& p : pts) {
    const Eigen::Vector3d q = p.cast<double>();
    if (up.dot(q) < -kMinBelow) low.push_back(q);
  }
  if (low.size() < 100) return std::nullopt;
  std::mt19937_64 rng(seed);
  const double cos_max = std::cos(max_tilt_deg * M_PI / 180.0);
  size_t best_inliers = 0;
  Eigen::Vector3d best_normal = Eigen::Vector3d::Zero();
  for (int it = 0; it < iterations; ++it) {
    const Eigen::Vector3d& a = low[rng() % low.size()];
    const Eigen::Vector3d& b = low[rng() % low.size()];
    const Eigen::Vector3d& c = low[rng() % low.size()];
    Eigen::Vector3d n = (b - a).cross(c - a);
    if (n.norm() < 1e-6) continue;
    n.normalize();
    if (std::abs(n.dot(up)) < cos_max) continue;  // only (nearly) horizontal planes are candidates
    size_t inl = 0;
    for (const auto& q : low) inl += std::abs(n.dot(q - a)) < thr;
    if (inl > best_inliers) {
      best_inliers = inl;
      best_normal = n;
    }
  }
  // The ground must be a fair part of the whole scan, as before (a twentieth), not only of what is
  // below.
  if (best_inliers < pts.size() / 20) return std::nullopt;
  // Refine with a least-squares fit over the inliers: the plane offset along best_normal with most
  // points close to it, then the plane through those points.
  std::vector<double> d(low.size());
  for (size_t i = 0; i < low.size(); ++i) d[i] = best_normal.dot(low[i]);
  double best_off = 0;
  size_t best_cnt = 0;
  for (size_t i = 0; i < low.size(); i += std::max<size_t>(1, low.size() / 400)) {
    size_t n_in = 0;
    for (double v : d) n_in += std::abs(v - d[i]) < thr;
    if (n_in > best_cnt) {
      best_cnt = n_in;
      best_off = d[i];
    }
  }
  Eigen::Vector3d p0 = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> in;
  for (size_t i = 0; i < low.size(); ++i) {
    if (std::abs(d[i] - best_off) < thr) {
      in.push_back(low[i]);
      p0 += low[i];
    }
  }
  if (in.size() < 30) return std::nullopt;
  p0 /= static_cast<double>(in.size());
  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  for (const auto& q : in) C += (q - p0) * (q - p0).transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
  Eigen::Vector3d n = es.eigenvectors().col(0);
  if (n.dot(up) < 0) n = -n;
  if (n.dot(up) < cos_max) return std::nullopt;
  // Only a broad, flat floor says where up is. Treads of a staircase, the small floor of a cabin or
  // a band of points that is not one surface tilt the fit by degrees: then there is no answer (the
  // caller keeps its attitude).
  const double count = static_cast<double>(in.size());
  const double rms = std::sqrt(std::max(0.0, es.eigenvalues()(0)) / count);
  const double width = std::sqrt(std::max(0.0, es.eigenvalues()(1)) /
                                 count);  // std along the narrower in-plane axis
  if (rms > 0.5 * thr || width < 0.5) return std::nullopt;
  if (n.dot(best_normal.dot(up) < 0 ? -best_normal : best_normal) < std::cos(2.0 * M_PI / 180.0))
    return std::nullopt;
  return n.normalized();
}

}  // namespace strataslam
