#include "strataslam/evaluation/metrics.hpp"

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace strataslam::eval {

std::string tum_stamp(int64_t stamp_ns) {
  char buf[32];
  const long long sec = stamp_ns / 1000000000LL, nsec = stamp_ns % 1000000000LL;
  if (stamp_ns < 0)
    std::snprintf(buf, sizeof buf, "-%lld.%09lld", -sec, -nsec);
  else
    std::snprintf(buf, sizeof buf, "%lld.%09lld", sec, nsec);
  return buf;
}

using Eigen::Matrix3d;
using Eigen::Vector3d;

bool interpolate_pose(const std::vector<TimedPose>& traj, Timestamp stamp, gtsam::Pose3* out) {
  if (traj.empty() || stamp < traj.front().stamp || stamp > traj.back().stamp) return false;
  const auto it = std::lower_bound(traj.begin(), traj.end(), stamp,
                                   [](const TimedPose& a, Timestamp s) { return a.stamp < s; });
  if (it == traj.begin()) {
    *out = it->pose;
    return true;
  }
  const auto& b = *it;
  const auto& a = *(it - 1);
  const double f = b.stamp == a.stamp ? 0.0 : double(stamp - a.stamp) / double(b.stamp - a.stamp);
  const Eigen::Quaterniond qa(a.pose.rotation().matrix()), qb(b.pose.rotation().matrix());
  *out = gtsam::Pose3(gtsam::Rot3(qa.slerp(f, qb).normalized().toRotationMatrix()),
                      Vector3d((1 - f) * a.pose.translation() + f * b.pose.translation()));
  return true;
}

gtsam::Pose3 align_se3(const std::vector<Vector3d>& p, const std::vector<Vector3d>& q) {
  const double n = static_cast<double>(p.size());
  Vector3d mp = Vector3d::Zero(), mq = Vector3d::Zero();
  for (size_t i = 0; i < p.size(); ++i) {
    mp += p[i];
    mq += q[i];
  }
  mp /= n;
  mq /= n;
  Matrix3d C = Matrix3d::Zero();
  for (size_t i = 0; i < p.size(); ++i) C += (q[i] - mq) * (p[i] - mp).transpose();
  Eigen::JacobiSVD<Matrix3d> svd(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix3d S = Matrix3d::Identity();
  if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0) S(2, 2) = -1;
  const Matrix3d R = svd.matrixU() * S * svd.matrixV().transpose();
  return gtsam::Pose3(gtsam::Rot3(R), Vector3d(mq - R * mp));
}

Pose4 align_4dof(const std::vector<Vector3d>& p, const std::vector<Vector3d>& q) {
  const double n = static_cast<double>(p.size());
  Vector3d mp = Vector3d::Zero(), mq = Vector3d::Zero();
  for (size_t i = 0; i < p.size(); ++i) {
    mp += p[i];
    mq += q[i];
  }
  mp /= n;
  mq /= n;
  double s_cross = 0, s_dot = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    const Vector3d a = p[i] - mp, b = q[i] - mq;
    s_dot += a.x() * b.x() + a.y() * b.y();
    s_cross += a.x() * b.y() - a.y() * b.x();
  }
  const double theta = std::atan2(s_cross, s_dot);
  const Pose4 R(theta, Vector3d::Zero());
  return Pose4(theta, Vector3d(mq - R.act(mp)));
}

// ATE and RPE as defined by J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, "A benchmark
// for the evaluation of RGB-D SLAM systems", IROS 2012. The SE(3) alignment is the closed form of
// S. Umeyama, "Least-squares estimation of transformation parameters between two point patterns",
// IEEE TPAMI 13(4), 1991 (without the scale).
AteResult ate(const std::vector<TimedPose>& est, const std::vector<TimedPose>& gt, AlignMode mode) {
  std::vector<Vector3d> pe, pg;
  for (const auto& e : est) {
    gtsam::Pose3 g;
    if (!interpolate_pose(gt, e.stamp, &g)) continue;
    pe.push_back(e.pose.translation());
    pg.push_back(g.translation());
  }
  AteResult r;
  r.n = pe.size();
  if (pe.size() < 3) return r;
  std::vector<Vector3d> aligned = pe;
  if (mode == AlignMode::SE3) {
    const gtsam::Pose3 T = align_se3(pe, pg);
    for (auto& a : aligned) a = T.transformFrom(a);
  } else if (mode == AlignMode::FourDof) {
    const Pose4 T = align_4dof(pe, pg);
    for (auto& a : aligned) a = T.act(a);
  }
  double s2 = 0, sm = 0;
  for (size_t i = 0; i < pe.size(); ++i) {
    const double d = (aligned[i] - pg[i]).norm();
    s2 += d * d;
    sm += d;
    r.max = std::max(r.max, d);
  }
  r.rmse = std::sqrt(s2 / pe.size());
  r.mean = sm / pe.size();
  return r;
}

RpeResult rpe(const std::vector<TimedPose>& est, const std::vector<TimedPose>& gt,
              int64_t delta_ns) {
  RpeResult r;
  double st = 0, sr = 0;
  for (const auto& e0 : est) {
    // Find the estimate pose at e0.stamp + delta (exact stamp match within the estimate list).
    gtsam::Pose3 e1p, g0, g1;
    if (!interpolate_pose(est, e0.stamp + delta_ns, &e1p)) continue;
    if (!interpolate_pose(gt, e0.stamp, &g0) || !interpolate_pose(gt, e0.stamp + delta_ns, &g1))
      continue;
    const gtsam::Pose3 err = g0.between(g1).between(e0.pose.between(e1p));
    st += err.translation().squaredNorm();
    const double ang = gtsam::Rot3::Logmap(err.rotation()).norm() * 180.0 / M_PI;
    sr += ang * ang;
    ++r.n;
  }
  if (r.n) {
    r.trans_rmse = std::sqrt(st / r.n);
    r.rot_rmse_deg = std::sqrt(sr / r.n);
  }
  return r;
}

double collapse_rate(const std::vector<LoopRecord>& accepted) {
  if (accepted.empty()) return 0.0;
  size_t bad = 0;
  for (const auto& l : accepted) {
    if (l.from_floor != l.to_floor || l.from_floor < 0) ++bad;
  }
  return static_cast<double>(bad) / accepted.size();
}

std::map<FloorId, double> floor_height_errors(const std::map<FloorId, std::vector<double>>& est_z,
                                              const std::map<FloorId, double>& true_z) {
  std::map<FloorId, double> out;
  for (const auto& [f, zs] : est_z) {
    const auto it = true_z.find(f);
    if (it == true_z.end() || zs.empty()) continue;
    double m = 0;
    for (double z : zs) m += z;
    out[f] = std::abs(m / zs.size() - it->second);
  }
  return out;
}

std::map<FloorId, FusionError> fusion_errors(const std::map<FloorId, Pose4>& est,
                                             const std::map<FloorId, Pose4>& truth) {
  std::map<FloorId, FusionError> out;
  for (const auto& [f, e] : est) {
    const auto it = truth.find(f);
    if (it == truth.end()) continue;
    const Pose4 d = it->second.between(e);
    out[f] = {d.translation().norm(), std::abs(d.theta()) * 180.0 / M_PI};
  }
  return out;
}

std::string Report::to_markdown() const {
  std::ostringstream os;
  os << "### " << title << "\n\n| Metric | Value |\n| --- | --- |\n";
  os << std::setprecision(6);
  for (const auto& [k, v] : values) os << "| " << k << " | " << v << " |\n";
  return os.str();
}

std::string Report::to_json() const {
  std::ostringstream os;
  os << std::setprecision(10) << "{\"title\": \"" << title << "\", \"metrics\": {";
  for (size_t i = 0; i < values.size(); ++i) {
    os << (i ? ", " : "") << "\"" << values[i].first << "\": " << values[i].second;
  }
  os << "}}";
  return os.str();
}

}  // namespace strataslam::eval
