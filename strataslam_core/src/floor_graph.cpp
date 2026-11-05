#include "strataslam/backend/floor_graph.hpp"

#include <algorithm>
#include <cmath>

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/AttitudeFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PoseTranslationPrior.h>
#include <gtsam/slam/PriorFactor.h>

#include "strataslam/backend/factors.hpp"

namespace strataslam {

using gtsam::Symbol;

namespace {
gtsam::Key key(KeyframeId id) { return gtsam::Symbol('k', id); }

gtsam::SharedNoiseModel diag_model(double sr, double st) {
  return gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << sr, sr, sr, st, st, st).finished());
}

EdgeRecord make_edge(const std::string& type, KeyframeId a, KeyframeId b, const gtsam::Pose3& m, double sr, double st) {
  EdgeRecord e;
  e.type = type;
  e.from = a;
  e.to = b;
  const Eigen::Quaterniond q(m.rotation().matrix());
  e.measurement = {m.x(), m.y(), m.z(), q.x(), q.y(), q.z(), q.w()};
  e.info_dim = 6;
  // Upper triangle of the diagonal information matrix (rot, rot, rot, trans, trans, trans).
  const double ir = 1.0 / (sr * sr), it = 1.0 / (st * st);
  const double diag[6] = {ir, ir, ir, it, it, it};
  for (int i = 0; i < 6; ++i)
    for (int j = i; j < 6; ++j) e.information.push_back(i == j ? diag[i] : 0.0);
  return e;
}
}  // namespace

FloorGraph::FloorGraph(FloorId id, const gtsam::Pose3& T_odom_floor, const BackendConfig& config)
    : id_(id), T_odom_floor_(T_odom_floor), cfg_(config) {
  gtsam::ISAM2Params p;
  p.relinearizeThreshold = 0.05;
  p.relinearizeSkip = 1;
  isam_ = std::make_unique<gtsam::ISAM2>(p);
}

Pose4 FloorGraph::origin() const {
  const auto R = T_odom_floor_.rotation().matrix();
  return Pose4(std::atan2(R(1, 0), R(0, 0)), T_odom_floor_.translation());
}

void FloorGraph::add_keyframe(KeyframeId id, Timestamp stamp, SessionId session, const gtsam::Pose3& T_floor_base,
                              std::optional<KeyframeId> odom_from, const gtsam::Pose3& odom_rel,
                              double noise_scale, std::optional<KeyframeId> bridge_from, std::optional<double> baro_dz) {
  nodes_[id] = {id, stamp, session, T_floor_base};
  order_.push_back(id);
  gtsam::NonlinearFactorGraph fg;
  gtsam::Values vals;
  vals.insert(key(id), T_floor_base);
  if (order_.size() == 1) {
    fg.add(gtsam::PriorFactor<gtsam::Pose3>(key(id), T_floor_base, diag_model(1e-4, 1e-4)));
  }
  if (odom_from && has(*odom_from)) {
    fg.add(gtsam::BetweenFactor<gtsam::Pose3>(key(*odom_from), key(id), odom_rel,
                                              diag_model(cfg_.odom_sigma_rot * noise_scale, cfg_.odom_sigma_trans * noise_scale)));
    edges_.push_back(make_edge("odometry", *odom_from, id, odom_rel, cfg_.odom_sigma_rot * noise_scale,
                               cfg_.odom_sigma_trans * noise_scale));
  } else if (bridge_from && has(*bridge_from)) {
    const gtsam::Pose3 rel = estimate_.exists(key(*bridge_from)) ? estimate_.at<gtsam::Pose3>(key(*bridge_from)).between(T_floor_base)
                                                                  : nodes_.at(*bridge_from).initial.between(T_floor_base);
    fg.add(gtsam::BetweenFactor<gtsam::Pose3>(key(*bridge_from), key(id), rel,
                                              diag_model(cfg_.bridge_sigma_rot, cfg_.bridge_sigma_trans)));
    edges_.push_back(make_edge("bridge", *bridge_from, id, rel, cfg_.bridge_sigma_rot, cfg_.bridge_sigma_trans));
  }
  if (order_.size() > 1 && fg.size() == 0) {
    // Nothing links this keyframe to the graph (its predecessor is on another floor or was never added):
    // a weak prior keeps the linear system determined instead of aborting inside ISAM2.
    fg.add(gtsam::PriorFactor<gtsam::Pose3>(key(id), T_floor_base, diag_model(10.0, 100.0)));
  }
  if (odom_from && has(*odom_from) && baro_dz) {
    fg.add(BaroDeltaZFactor(key(*odom_from), key(id), *baro_dz, gtsam::noiseModel::Isotropic::Sigma(1, cfg_.baro_sigma)));
  }
  graph_.push_back(fg);
  isam_->update(fg, vals);
  estimate_ = isam_->calculateEstimate();
}

void FloorGraph::add_gravity(KeyframeId id, const Eigen::Vector3d& up_body, double sigma) {
  if (!has(id)) return;
  gtsam::NonlinearFactorGraph fg;
  fg.add(gtsam::Pose3AttitudeFactor(key(id), gtsam::Unit3(0, 0, 1), gtsam::noiseModel::Isotropic::Sigma(2, sigma),
                                    gtsam::Unit3(up_body)));
  graph_.push_back(fg);
  isam_->update(fg, gtsam::Values());
  estimate_ = isam_->calculateEstimate();
}

void FloorGraph::add_plane(KeyframeId id, const Eigen::Vector3d& up_body) {
  if (!has(id)) return;
  gtsam::NonlinearFactorGraph fg;
  // Height only: x and y stay free (a huge sigma), as the heading does in the attitude factor.
  fg.add(gtsam::PoseTranslationPrior<gtsam::Pose3>(key(id), gtsam::Point3(0, 0, 0),
                                                    gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(1e4, 1e4, cfg_.plane_sigma))));
  fg.add(gtsam::Pose3AttitudeFactor(key(id), gtsam::Unit3(0, 0, 1), gtsam::noiseModel::Isotropic::Sigma(2, cfg_.plane_attitude_sigma),
                                    gtsam::Unit3(up_body)));
  graph_.push_back(fg);
  isam_->update(fg, gtsam::Values());
  estimate_ = isam_->calculateEstimate();
}

gtsam::Pose3 FloorGraph::pose(KeyframeId id) const {
  if (estimate_.exists(key(id))) return estimate_.at<gtsam::Pose3>(key(id));
  return nodes_.at(id).initial;
}

std::vector<KeyframeId> FloorGraph::keyframe_ids() const { return order_; }

void FloorGraph::sort_by_stamp() {
  std::stable_sort(order_.begin(), order_.end(), [&](KeyframeId a, KeyframeId b) { return nodes_.at(a).stamp < nodes_.at(b).stamp; });
}

double FloorGraph::loop_mahalanobis2(KeyframeId from, KeyframeId to, const gtsam::Pose3& T_from_to) const {
  const gtsam::Pose3 Xa = pose(from), Xb = pose(to);
  gtsam::Matrix H1, H2;
  gtsam::BetweenFactor<gtsam::Pose3> f(key(from), key(to), T_from_to, diag_model(cfg_.loop_sigma_rot, cfg_.loop_sigma_trans));
  const gtsam::Vector e = f.evaluateError(Xa, Xb, H1, H2);
  gtsam::Matrix Sigma_pred;
  try {
    gtsam::Marginals marg(graph_, estimate_);
    const gtsam::KeyVector keys{key(from), key(to)};
    const gtsam::Matrix J = marg.jointMarginalCovariance(keys).fullMatrix();
    gtsam::Matrix H(6, 12);
    H << H1, H2;
    Sigma_pred = H * J * H.transpose();
  } catch (const std::exception&) {
    Sigma_pred = gtsam::Matrix::Zero(6, 6);
  }
  gtsam::Vector sig(6);
  sig << cfg_.loop_sigma_rot, cfg_.loop_sigma_rot, cfg_.loop_sigma_rot, cfg_.loop_sigma_trans, cfg_.loop_sigma_trans, cfg_.loop_sigma_trans;
  const gtsam::Matrix Sigma = Sigma_pred + gtsam::Matrix(sig.array().square().matrix().asDiagonal());
  return e.transpose() * Sigma.ldlt().solve(e);
}

Result<void> FloorGraph::add_loop(KeyframeId from, KeyframeId to, const gtsam::Pose3& T_from_to, bool check_consistency) {
  // Layer 3 of the floor gate: a factor may only join two nodes of this graph. It checks graph
  // membership, not the floor labels that layer 2 compares, so it holds even if those are wrong.
  if (!has(from) || !has(to)) return Error{"backend assertion: cross-floor loop closure factors are not allowed (keyframe not in this floor graph)"};
  if (check_consistency) {
    const double m2 = loop_mahalanobis2(from, to, T_from_to);
    if (m2 > cfg_.mahalanobis_chi2) return Error{"loop closure rejected by the Mahalanobis consistency check"};
  }
  const auto base = diag_model(cfg_.loop_sigma_rot, cfg_.loop_sigma_trans);
  const auto robust = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(cfg_.loop_cauchy_k), base);
  gtsam::NonlinearFactorGraph fg;
  fg.add(gtsam::BetweenFactor<gtsam::Pose3>(key(from), key(to), T_from_to, robust));
  graph_.push_back(fg);
  isam_->update(fg, gtsam::Values());
  for (int i = 0; i < 3; ++i) isam_->update();
  estimate_ = isam_->calculateEstimate();
  edges_.push_back(make_edge("loop", from, to, T_from_to, cfg_.loop_sigma_rot, cfg_.loop_sigma_trans));
  return {};
}

void FloorGraph::rebuild_isam() {
  gtsam::ISAM2Params p;
  p.relinearizeThreshold = 0.05;
  p.relinearizeSkip = 1;
  isam_ = std::make_unique<gtsam::ISAM2>(p);
  isam_->update(graph_, estimate_);
  estimate_ = isam_->calculateEstimate();
}

void FloorGraph::optimize_batch() {
  if (nodes_.empty()) return;
  gtsam::LevenbergMarquardtParams lm;
  lm.maxIterations = 50;
  estimate_ = gtsam::LevenbergMarquardtOptimizer(graph_, estimate_, lm).optimize();
  rebuild_isam();
}

}  // namespace strataslam
