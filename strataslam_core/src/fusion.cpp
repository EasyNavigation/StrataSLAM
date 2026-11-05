#include "strataslam/fusion/fusion.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/AttitudeFactor.h>
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <queue>
#include <set>
#include <sstream>

#include "strataslam/fusion/ablation_factors.hpp"
#include "strataslam/fusion/factors4.hpp"
#include "strataslam/fusion/registration4.hpp"

namespace strataslam {

const char* to_string(Formulation f) {
  switch (f) {
    case Formulation::Lie4:
      return "lie4";
    case Formulation::Se3Soft:
      return "se3soft";
    case Formulation::Euler:
      return "euler";
  }
  return "lie4";
}
bool parse_formulation(const std::string& n, Formulation* out) {
  for (Formulation f : {Formulation::Lie4, Formulation::Se3Soft, Formulation::Euler}) {
    if (n == to_string(f)) {
      *out = f;
      return true;
    }
  }
  return false;
}

Pose4 keyframe_pose4(const KeyframeRecord& k) {
  const Eigen::Matrix3d R = k.q.toRotationMatrix();
  return Pose4(std::atan2(R(1, 0), R(0, 0)), k.t);
}

namespace {

gtsam::Key key_of(FloorId f) { return gtsam::Symbol('f', static_cast<uint64_t>(f)); }

/// A relative constraint between two floor frames, in the Pose4 tangent space.
struct Constraint {
  enum class Kind { Transition, BaroHeight, FloorHeightPrior, Match, Anchor } kind;
  std::string label;
  FloorId a = 0, b = 0;
  Pose4 z;  // relative pose X_a^-1 X_b (Transition, Match, Anchor closed form)
  Eigen::Matrix4d cov = Eigen::Matrix4d::Identity();
  double dz = 0, sigma = 1;             // height constraints
  std::vector<Eigen::Vector3d> pa, pb;  // anchors
  bool use_z = false;
  double anchor_sigma = 0.03;
  bool has_relative = false;  // z is meaningful (propagates initial values)
};

const char* kind_name(Constraint::Kind k) {
  switch (k) {
    case Constraint::Kind::Transition:
      return "transition";
    case Constraint::Kind::BaroHeight:
      return "baro_height";
    case Constraint::Kind::FloorHeightPrior:
      return "floor_height_prior";
    case Constraint::Kind::Match:
      return "cross_floor_match";
    case Constraint::Kind::Anchor:
      return "anchor";
  }
  return "?";
}

const KeyframeRecord* find_keyframe(const MapData& m, FloorId floor, KeyframeId id) {
  for (const auto& f : m.floors) {
    if (f.metadata.id != floor) continue;
    for (const auto& k : f.keyframes)
      if (k.id == id) return &k;
  }
  return nullptr;
}

gtsam::SharedNoiseModel model_from_cov(const Eigen::Matrix4d& cov) {
  return gtsam::noiseModel::Gaussian::Covariance(cov);
}

Eigen::Matrix4d reorder_cov_euler(const Eigen::Matrix4d& c4) {
  // Pose4 tangent (phi, rho_x, rho_y, rho_z) -> Euler ordering (x, y, z, yaw).
  Eigen::Matrix4d P = Eigen::Matrix4d::Zero();
  P(0, 1) = 1;
  P(1, 2) = 1;
  P(2, 3) = 1;
  P(3, 0) = 1;
  return P * c4 * P.transpose();
}

Pose4 pose4_from_pose3(const gtsam::Pose3& p) {
  const auto R = p.rotation().matrix();
  return Pose4(std::atan2(R(1, 0), R(0, 0)), p.translation());
}

}  // namespace

bool FusionResult::finite() const {
  for (const auto& kv : floor_to_map) {
    if (!std::isfinite(kv.second.theta()) || !kv.second.translation().allFinite()) return false;
  }
  return true;
}

namespace {

/// State shared by the phases of `fuse`.
struct FusionContext {
  const MapData& map;
  const FusionOptions& opt;
  FusionResult& res;
  std::vector<FloorId> floors;         ///< sorted
  std::map<FloorId, double> altitude;  ///< metadata altitude of each floor, when known
  FloorId root = 0;
  std::vector<Constraint> cons;
  std::set<std::pair<FloorId, FloorId>> height_constrained;
  std::set<std::pair<FloorId, FloorId>> declared_same_level;
  std::map<FloorId, Pose4> init;  ///< initial values propagated from the root
};

/// The factor graph of the chosen formulation.
struct FusionGraph {
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  std::vector<size_t> known_inliers;
  std::vector<int> factor_of_constraint;  ///< graph index of each constraint (-1: none)
};

/// 1a. Transitions with a relative pose (and the barometric height of elevators).
void add_transition_constraints(FusionContext& c) {
  const MapData& map = c.map;
  const FusionOptions& opt = c.opt;
  FusionResult& res = c.res;
  const std::vector<FloorId>& floors = c.floors;
  std::vector<Constraint>& cons = c.cons;
  std::set<std::pair<FloorId, FloorId>>& height_constrained = c.height_constrained;
  const auto has_floor = [&](FloorId f) {
    return std::find(floors.begin(), floors.end(), f) != floors.end();
  };
  for (const auto& t : map.transitions) {
    if (!has_floor(t.from) || !has_floor(t.to) || t.from == t.to) continue;
    const std::string label =
        std::string(to_string(t.type)) + " " + std::to_string(t.from) + "->" + std::to_string(t.to);
    if (t.relative && t.continuous_tracking) {
      const KeyframeRecord *ka = find_keyframe(map, t.from, t.kf_before),
                           *kb = find_keyframe(map, t.to, t.kf_after);
      if (!ka || !kb) {
        res.warnings.push_back("transition " + label + ": keyframes not found, skipped");
      } else {
        // T_fa_fb = T_fa_kfa * Z_rel * T_fb_kfb^-1 (all gravity-aligned Pose4).
        const Pose4 rel((*t.relative)[0],
                        Eigen::Vector3d((*t.relative)[1], (*t.relative)[2], (*t.relative)[3]));
        Constraint c;
        c.kind = Constraint::Kind::Transition;
        c.label = label;
        c.a = t.from;
        c.b = t.to;
        c.z = keyframe_pose4(*ka) * rel * keyframe_pose4(*kb).inverse();
        c.cov = t.covariance + Eigen::Vector4d(0.005 * 0.005, 0.02 * 0.02, 0.02 * 0.02, 0.02 * 0.02)
                                   .asDiagonal()
                                   .toDenseMatrix();
        {
          const double travelled =
              rel.translation().norm();  // a lower bound of the path the odometry integrated
          const double sd = opt.transition_drift_per_m * travelled,
                       sy = opt.transition_yaw_drift_per_m * travelled;
          c.cov(0, 0) += sy * sy;
          c.cov(1, 1) += sd * sd;
          c.cov(2, 2) += sd * sd;
        }
        c.has_relative = true;
        if (t.type == TransitionType::Elevator && t.delta_z) {
          // The height of an elevator comes from the barometer/IMU: its own factor, and the
          // relative pose only constrains x, y and yaw.
          c.cov(3, 3) = 1e4;
          Constraint h;
          h.kind = Constraint::Kind::BaroHeight;
          h.label = label;
          h.a = t.from;
          h.b = t.to;
          h.dz = *t.delta_z - ka->t.z() +
                 kb->t.z();  // frame origins: dz between kf bases corrected to the frames
          h.sigma = std::sqrt(std::max(t.covariance(3, 3), 0.05 * 0.05));
          cons.push_back(h);
          height_constrained.insert({t.from, t.to});
        } else {
          height_constrained.insert({t.from, t.to});
        }
        cons.push_back(c);
      }
    } else if (t.delta_z) {
      res.warnings.push_back("transition " + label +
                             ": no relative pose (unobservable change); needs anchors");
    }
  }
}

/// 1b. Cross-floor matches and anchor sets (robustly pre-aligned).
void add_match_and_anchor_constraints(FusionContext& c) {
  const MapData& map = c.map;
  const FusionOptions& opt = c.opt;
  FusionResult& res = c.res;
  const std::vector<FloorId>& floors = c.floors;
  std::vector<Constraint>& cons = c.cons;
  std::set<std::pair<FloorId, FloorId>>& height_constrained = c.height_constrained;
  const auto has_floor = [&](FloorId f) {
    return std::find(floors.begin(), floors.end(), f) != floors.end();
  };
  for (const auto& m : opt.matches) {
    if (!has_floor(m.floor_a) || !has_floor(m.floor_b)) continue;
    const KeyframeRecord *ka = find_keyframe(map, m.floor_a, m.kf_a),
                         *kb = find_keyframe(map, m.floor_b, m.kf_b);
    if (!ka || !kb) {
      res.warnings.push_back("cross-floor match: keyframes not found, skipped");
      continue;
    }
    Constraint c;
    c.kind = Constraint::Kind::Match;
    c.label = "match " + std::to_string(m.floor_a) + "<->" + std::to_string(m.floor_b);
    c.a = m.floor_a;
    c.b = m.floor_b;
    c.z = keyframe_pose4(*ka) * m.relative * keyframe_pose4(*kb).inverse();
    c.cov = m.covariance;
    c.has_relative = true;
    cons.push_back(c);
  }
  for (const auto& set : opt.anchors) {
    // Consecutive floors of the set; correspondences are pre-filtered by robust alignment.
    std::vector<FloorId> fs;
    for (const auto& kv : set.points)
      if (has_floor(kv.first)) fs.push_back(kv.first);
    for (size_t i = 0; i + 1 < fs.size(); ++i) {
      const FloorId a = fs[i], b = fs[i + 1];
      const auto& pa = set.points.at(a);
      const auto& pb = set.points.at(b);
      if (pa.size() != pb.size() || pa.size() < 2) {
        res.warnings.push_back("anchor set '" + set.name + "': floors " + std::to_string(a) +
                               " and " + std::to_string(b) + " need >= 2 matching points");
        continue;
      }
      // X_a p_a = X_b p_b: align b's points onto a's gives T_fa_fb.
      std::vector<PointPair> pairs;
      for (size_t k = 0; k < pa.size(); ++k) pairs.push_back({pb[k], pa[k]});
      const RobustAlignment ra = align_pose4_gnc(pairs, opt.anchor_outlier_bound, set.use_z);
      if (!ra.converged) {
        res.warnings.push_back("anchor set '" + set.name + "': robust alignment failed");
        continue;
      }
      Constraint c;
      c.kind = Constraint::Kind::Anchor;
      c.label = set.name + " " + std::to_string(a) + "<->" + std::to_string(b);
      c.a = a;
      c.b = b;
      c.use_z = set.use_z;
      c.anchor_sigma = set.sigma;
      for (size_t k = 0; k < pa.size(); ++k)
        if (ra.inlier[k]) {
          c.pa.push_back(pa[k]);
          c.pb.push_back(pb[k]);
        }
      c.z = ra.transform;
      const double n = std::max<double>(2.0, ra.n_inliers);
      const double s2 =
          2.0 * set.sigma * set.sigma / n;  // two independent marks per correspondence
      c.cov = Eigen::Vector4d(s2, s2, s2, set.use_z ? s2 : 1e4).asDiagonal().toDenseMatrix();
      c.has_relative = true;
      if (set.use_z) height_constrained.insert({a, b});
      if (ra.n_inliers < pa.size())
        res.warnings.push_back("anchor set '" + set.name +
                               "': " + std::to_string(pa.size() - ra.n_inliers) +
                               " outlier correspondence(s) rejected");
      cons.push_back(c);
    }
  }
}

/// 1c. Declared same levels (which release the transition height) and the floor height prior.
void add_height_constraints(FusionContext& c) {
  const FusionOptions& opt = c.opt;
  FusionResult& res = c.res;
  const std::vector<FloorId>& floors = c.floors;
  std::vector<Constraint>& cons = c.cons;
  std::set<std::pair<FloorId, FloorId>>& height_constrained = c.height_constrained;
  std::set<std::pair<FloorId, FloorId>>& declared_same_level = c.declared_same_level;
  const auto has_floor = [&](FloorId f) {
    return std::find(floors.begin(), floors.end(), f) != floors.end();
  };
  std::map<FloorId, double>& altitude = c.altitude;
  for (const auto& s : opt.same_level) {
    if (!has_floor(s.a) || !has_floor(s.b) || s.a == s.b) {
      res.warnings.push_back("same level " + std::to_string(s.a) + "," + std::to_string(s.b) +
                             ": unknown floors, skipped");
      continue;
    }
    Constraint h;
    h.kind = Constraint::Kind::FloorHeightPrior;
    h.label = "same level " + std::to_string(s.a) + "," + std::to_string(s.b);
    h.a = s.a;
    h.b = s.b;
    h.dz = 0.0;
    h.sigma = s.sigma;
    cons.push_back(h);
    height_constrained.insert({s.a, s.b});
    height_constrained.insert({s.b, s.a});
    declared_same_level.insert({std::min(s.a, s.b), std::max(s.a, s.b)});
  }
  // Declaring two floors the same level says that the height read through the transition between
  // them is the error (a drift): x, y and yaw stay, the height of that transition is released.
  for (auto& c : cons) {
    if (c.kind != Constraint::Kind::Transition) continue;
    if (declared_same_level.count({std::min(c.a, c.b), std::max(c.a, c.b)}))
      c.cov(3, 3) += 2.0 * 2.0;
  }
  if (opt.floor_height_prior) {
    // Weak height prior between floors adjacent in altitude (negligible next to a barometer or
    // continuous odometry, decisive when the height change was unobservable).
    std::vector<std::pair<double, FloorId>> by_alt;
    by_alt.reserve(floors.size());
    for (FloorId f : floors)
      by_alt.push_back({altitude.count(f) ? altitude[f] : f * *opt.floor_height_prior, f});
    std::sort(by_alt.begin(), by_alt.end());
    for (size_t i = 0; i + 1 < by_alt.size(); ++i) {
      const FloorId a = by_alt[i].second, b = by_alt[i + 1].second;
      const double n = std::max(
          1.0, std::round((by_alt[i + 1].first - by_alt[i].first) / *opt.floor_height_prior));
      Constraint h;
      h.kind = Constraint::Kind::FloorHeightPrior;
      h.label = "floors " + std::to_string(a) + "," + std::to_string(b);
      h.a = a;
      h.b = b;
      h.dz = n * *opt.floor_height_prior;
      h.sigma = opt.floor_height_sigma;
      cons.push_back(h);
      height_constrained.insert({a, b});
    }
  }
}

/// 2. Initial values: relative constraints, then heights, propagated from the root.
void propagate_initial_values(FusionContext& c) {
  std::vector<Constraint>& cons = c.cons;
  std::map<FloorId, Pose4>& init = c.init;
  const FloorId root = c.root;
  init[root] = Pose4();
  bool grew = true;
  while (grew) {
    grew = false;
    for (const auto& c : cons) {
      if (!c.has_relative) continue;
      if (init.count(c.a) && !init.count(c.b)) {
        init[c.b] = init[c.a] * c.z;
        grew = true;
      } else if (init.count(c.b) && !init.count(c.a)) {
        init[c.a] = init[c.b] * c.z.inverse();
        grew = true;
      }
    }
    // Heights propagate through height constraints for floors without a relative pose.
    for (const auto& c : cons) {
      if (c.kind != Constraint::Kind::BaroHeight && c.kind != Constraint::Kind::FloorHeightPrior)
        continue;
      if (init.count(c.a) && !init.count(c.b)) {
        init[c.b] = Pose4(0, init[c.a].translation() + Eigen::Vector3d(0, 0, c.dz));
        grew = true;
      } else if (init.count(c.b) && !init.count(c.a)) {
        init[c.a] = Pose4(0, init[c.b].translation() - Eigen::Vector3d(0, 0, c.dz));
        grew = true;
      }
    }
  }
}

/// The test that makes a matched pair of floors a same-level candidate (ADR-0031).
bool is_level_candidate(double dz, double score, const FusionOptions& opt) {
  return opt.same_level_max_dz > 0 && std::abs(dz) <= opt.same_level_max_dz &&
         score >= opt.same_level_min_score;
}

/// 2b. Structural stacking of every pair of floors (unambiguous matches become constraints).
void add_stacking_constraints(FusionContext& c) {
  const MapData& map = c.map;
  const FusionOptions& opt = c.opt;
  FusionResult& res = c.res;
  const std::vector<FloorId>& floors = c.floors;
  std::vector<Constraint>& cons = c.cons;
  std::set<std::pair<FloorId, FloorId>>& declared_same_level = c.declared_same_level;
  std::map<FloorId, Pose4>& init = c.init;
  std::map<FloorId, const FloorData*> data;
  for (const auto& f : map.floors) data[f.metadata.id] = &f;
  for (size_t i = 0; i < floors.size(); ++i)
    for (size_t j = i + 1; j < floors.size(); ++j) {
      const FloorId a = floors[i], b = floors[j];
      const std::optional<PointCloud>& cloud_a = data[a]->floor_cloud;
      const std::optional<PointCloud>& cloud_b = data[b]->floor_cloud;
      if (!init.count(a) || !init.count(b) || !cloud_a || !cloud_b) continue;
      std::string why;
      StackingOptions so = opt.stacking;
      so.allow_ambiguous =
          true;  // an ambiguous pose is no constraint, but it can still be a candidate
      const Pose4 chained = init[a].inverse() * init[b];
      const auto m = match_floor_structure(*cloud_a, *cloud_b, chained, so, &why);
      const std::string label = std::to_string(a) + "<->" + std::to_string(b);
      if (!m) {
        res.warnings.push_back("stacking " + label + " not used: ");
        res.warnings.back() += why;
        continue;
      }
      const double dz = chained.translation().z();
      if (is_level_candidate(dz, m->score, opt) &&
          !declared_same_level.count({std::min(a, b), std::max(a, b)})) {
        res.same_level_candidates.push_back({a, b, dz, m->score, m->ambiguous});
        res.warnings.push_back("floors " + std::to_string(a) + " and " + std::to_string(b) +
                               " could be the same level: compatible structure (score " +
                               std::to_string(m->score) + "), heights " +
                               std::to_string(std::abs(dz)) +
                               " m apart; confirm with `same_level` if they are");
      }
      if (m->ambiguous) {
        res.warnings.push_back("stacking " + label + " not used: ambiguous: best score " +
                               std::to_string(m->score) + " against " +
                               std::to_string(m->runner_up) + " at another pose");
        continue;
      }
      Constraint c;
      c.kind = Constraint::Kind::Match;
      c.label = "stacking " + label;
      c.a = a;
      c.b = b;
      c.z = m->a_to_b;
      c.cov = m->covariance;
      c.has_relative = true;
      cons.push_back(c);
    }
}

/// 3. The factor graph of `c.opt.formulation` over the constraints.
FusionGraph build_graph(FusionContext& c) {
  const FusionOptions& opt = c.opt;
  const std::vector<FloorId>& floors = c.floors;
  std::vector<Constraint>& cons = c.cons;
  std::map<FloorId, Pose4>& init = c.init;
  const FloorId root = c.root;
  FusionGraph g;
  gtsam::NonlinearFactorGraph& graph = g.graph;
  gtsam::Values& values = g.values;
  std::vector<size_t>& known_inliers = g.known_inliers;
  std::vector<int>& factor_of_constraint = g.factor_of_constraint;
  const Formulation form = opt.formulation;

  const auto tight = [](size_t d) { return gtsam::noiseModel::Isotropic::Sigma(d, 1e-6); };
  if (form == Formulation::Lie4) {
    for (FloorId f : floors) values.insert(key_of(f), init[f]);
    known_inliers.push_back(graph.size());
    graph.add(gtsam::PriorFactor<Pose4>(key_of(root), Pose4(), tight(4)));
  } else if (form == Formulation::Se3Soft) {
    for (FloorId f : floors)
      values.insert(key_of(f),
                    gtsam::Pose3(gtsam::Rot3::Yaw(init[f].theta()), init[f].translation()));
    known_inliers.push_back(graph.size());
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(key_of(root), gtsam::Pose3(), tight(6)));
    for (FloorId f : floors) {  // soft roll/pitch prior: the body z axis stays vertical
      known_inliers.push_back(graph.size());
      graph.add(gtsam::Pose3AttitudeFactor(key_of(f), gtsam::Unit3(0, 0, 1),
                                           gtsam::noiseModel::Isotropic::Sigma(2, 0.01),
                                           gtsam::Unit3(0, 0, 1)));
    }
  } else {
    for (FloorId f : floors)
      values.insert(key_of(f), gtsam::Vector4(init[f].translation().x(), init[f].translation().y(),
                                              init[f].translation().z(), init[f].theta()));
    known_inliers.push_back(graph.size());
    graph.add(gtsam::PriorFactor<gtsam::Vector4>(key_of(root), gtsam::Vector4::Zero(), tight(4)));
  }

  for (const auto& c : cons) {
    factor_of_constraint.push_back(static_cast<int>(graph.size()));
    if (c.kind == Constraint::Kind::BaroHeight || c.kind == Constraint::Kind::FloorHeightPrior) {
      const auto m = gtsam::noiseModel::Isotropic::Sigma(1, c.sigma);
      if (form == Formulation::Lie4) {
        if (c.kind == Constraint::Kind::BaroHeight)
          graph.add(BaroHeightFactor(key_of(c.a), key_of(c.b), c.dz, m));
        else
          graph.add(FloorHeightPrior(key_of(c.a), key_of(c.b), c.dz, m));
      } else if (form == Formulation::Se3Soft) {
        graph.add(ablation::Height3Factor(key_of(c.a), key_of(c.b), c.dz, m));
      } else {
        graph.add(ablation::EulerHeight(key_of(c.a), key_of(c.b), c.dz, m));
      }
      continue;
    }
    if (form == Formulation::Lie4) {
      if (c.kind == Constraint::Kind::Anchor) {
        // Point correspondences enter directly; the height of a replicated structure is not
        // compared.
        graph.add(AnchorFactor(key_of(c.a), key_of(c.b), c.pa, c.pb, c.use_z, c.anchor_sigma));
      } else if (c.kind == Constraint::Kind::Match) {
        graph.add(CrossFloorMatchFactor(key_of(c.a), key_of(c.b), c.z, model_from_cov(c.cov)));
      } else {
        graph.add(TransitionFactor(key_of(c.a), key_of(c.b), c.z, model_from_cov(c.cov)));
      }
    } else if (c.kind == Constraint::Kind::Anchor) {
      // The anchors' point correspondences, as for Pose4: every formulation gets the same
      // information (ADR-0049).
      if (form == Formulation::Se3Soft)
        graph.add(
            ablation::Anchor3Factor(key_of(c.a), key_of(c.b), c.pa, c.pb, c.use_z, c.anchor_sigma));
      else
        graph.add(ablation::EulerAnchorFactor(key_of(c.a), key_of(c.b), c.pa, c.pb, c.use_z,
                                              c.anchor_sigma));
    } else if (form == Formulation::Se3Soft) {
      gtsam::Vector6 s;
      s << 0.01, 0.01, std::sqrt(c.cov(0, 0)), std::sqrt(c.cov(1, 1)), std::sqrt(c.cov(2, 2)),
          std::sqrt(c.cov(3, 3));  // roll, pitch, yaw, x, y, z
      graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
          key_of(c.a), key_of(c.b), gtsam::Pose3(gtsam::Rot3::Yaw(c.z.theta()), c.z.translation()),
          gtsam::noiseModel::Diagonal::Sigmas(s)));
    } else {
      graph.add(ablation::EulerBetween(
          key_of(c.a), key_of(c.b), c.z,
          gtsam::noiseModel::Gaussian::Covariance(reorder_cov_euler(c.cov))));
    }
  }

  return g;
}

/// 4. Levenberg-Marquardt, with GNC (truncated least squares) to reject wrong constraints when the
///    graph is redundant enough. Returns the solution and the weight of every factor.
Result<std::pair<gtsam::Values, gtsam::Vector>> optimize_graph(FusionContext& c,
                                                               const FusionGraph& g) {
  const FusionOptions& opt = c.opt;
  FusionResult& res = c.res;
  const gtsam::NonlinearFactorGraph& graph = g.graph;
  const gtsam::Values& values = g.values;
  const std::vector<size_t>& known_inliers = g.known_inliers;
  gtsam::LevenbergMarquardtParams lm;
  lm.maxIterations = opt.max_iterations;
  gtsam::Values result;
  gtsam::Vector weights = gtsam::Vector::Ones(static_cast<Eigen::Index>(graph.size()));
  try {
    const size_t redundant = graph.size() - known_inliers.size();
    if (opt.use_gnc && redundant >= 2) {
      gtsam::GncParams<gtsam::LevenbergMarquardtParams> gp(lm);
      gp.setLossType(gtsam::GncLossType::TLS);
      gtsam::GncParams<gtsam::LevenbergMarquardtParams>::IndexVector known(known_inliers.begin(),
                                                                           known_inliers.end());
      gp.setKnownInliers(known);
      gtsam::GncOptimizer<gtsam::GncParams<gtsam::LevenbergMarquardtParams>> gnc(graph, values, gp);
      result = gnc.optimize();
      weights = gnc.getWeights();
      // GNC with a truncated loss has converged when every weight is 0 or 1 (inlier or outlier
      // decided).
      res.converged = gnc.checkWeightsConvergence(weights);
      // GNC runs one LM per step of its continuation; the iterations of those inner runs are not
      // exposed.
      res.iterations = -1;
    } else {
      gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm);
      result = optimizer.optimize();
      res.iterations = static_cast<int>(optimizer.iterations());
      // Levenberg-Marquardt stops on its error tolerances or on the iteration limit; only the
      // former is converged.
      res.converged = optimizer.iterations() < lm.maxIterations;
    }
  } catch (const std::exception& e) {
    return Error{std::string("fusion optimization failed: ") + e.what()};
  }
  res.final_error = graph.error(result);
  return std::make_pair(result, weights);
}

/// 5. Poses, constraint report and marginal covariances into `c.res`.
void collect_results(FusionContext& c, const FusionGraph& g, const gtsam::Values& result,
                     const gtsam::Vector& weights) {
  const FusionOptions& opt = c.opt;
  FusionResult& res = c.res;
  const std::vector<FloorId>& floors = c.floors;
  std::vector<Constraint>& cons = c.cons;
  const FloorId root = c.root;
  const gtsam::NonlinearFactorGraph& graph = g.graph;
  const std::vector<int>& factor_of_constraint = g.factor_of_constraint;
  const Formulation form = opt.formulation;
  for (FloorId f : floors) {
    if (form == Formulation::Lie4)
      res.floor_to_map[f] = result.at<Pose4>(key_of(f));
    else if (form == Formulation::Se3Soft)
      res.floor_to_map[f] = pose4_from_pose3(result.at<gtsam::Pose3>(key_of(f)));
    else {
      const gtsam::Vector4 v = result.at<gtsam::Vector4>(key_of(f));
      res.floor_to_map[f] = Pose4(v(3), Eigen::Vector3d(v(0), v(1), v(2)));
    }
  }
  for (size_t i = 0; i < cons.size(); ++i) {
    ConstraintReport r;
    r.kind = kind_name(cons[i].kind);
    r.label = cons[i].label;
    r.a = cons[i].a;
    r.b = cons[i].b;
    const size_t gi = static_cast<size_t>(factor_of_constraint[i]);
    r.weight = weights(static_cast<Eigen::Index>(gi));
    r.accepted = r.weight > 0.5;
    const auto nm = boost::dynamic_pointer_cast<gtsam::NoiseModelFactor>(graph.at(gi));
    r.residual = nm ? std::sqrt(2.0 * nm->error(result)) : 0.0;
    if (!r.accepted)
      res.warnings.push_back("constraint rejected as an outlier: " + r.kind + " " + r.label);
    res.constraints.push_back(r);
  }
  try {
    gtsam::Marginals marg(graph, result);
    for (FloorId f : floors) {
      const gtsam::Matrix C = marg.marginalCovariance(key_of(f));
      Eigen::Matrix4d c4 = Eigen::Matrix4d::Zero();
      if (form == Formulation::Lie4)
        c4 = C;
      else if (form == Formulation::Se3Soft) {  // (rx, ry, rz, x, y, z) -> (yaw, x, y, z)
        const int idx[4] = {2, 3, 4, 5};
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j) c4(i, j) = C(idx[i], idx[j]);
      } else {  // (x, y, z, yaw) -> (yaw, x, y, z)
        const int idx[4] = {3, 0, 1, 2};
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j) c4(i, j) = C(idx[i], idx[j]);
      }
      res.covariance[f] = c4;
      if (f != root && std::sqrt(std::max(0.0, c4(0, 0))) > 0.2) {
        res.warnings.push_back("floor " + std::to_string(f) +
                               " has no usable rotation constraint (yaw sigma " +
                               std::to_string(std::sqrt(c4(0, 0)) * 180 / M_PI) +
                               " deg): add anchors or a cross-floor match");
      }
    }
  } catch (const std::exception&) {
    res.warnings.push_back("marginal covariances unavailable (under-constrained graph)");
  }
}

}  // namespace

std::optional<LevelCandidate> same_level_candidate(FloorId a, const PointCloud& cloud_a, FloorId b,
                                                   const PointCloud& cloud_b, const Pose4& a_to_b,
                                                   const FusionOptions& opt, std::string* why) {
  const double dz = a_to_b.translation().z();
  if (opt.same_level_max_dz <= 0 || std::abs(dz) > opt.same_level_max_dz) {
    if (why) *why = "heights too far apart";
    return std::nullopt;
  }
  StackingOptions so = opt.stacking;
  so.allow_ambiguous = true;
  const auto m = match_floor_structure(cloud_a, cloud_b, a_to_b, so, why);
  if (!m || !is_level_candidate(dz, m->score, opt)) {
    if (m && why)
      *why = "structure score " + std::to_string(m->score) + " below " +
             std::to_string(opt.same_level_min_score);
    return std::nullopt;
  }
  return LevelCandidate{a, b, dz, m->score, m->ambiguous};
}

Result<FusionResult> fuse(const MapData& map, const FusionOptions& opt) {
  const auto t0 = std::chrono::steady_clock::now();
  FusionResult res;
  res.formulation = opt.formulation;
  if (map.floors.empty()) return Error{"the map has no floors"};
  FusionContext c{map, opt, res, {}, {}, 0, {}, {}, {}, {}};
  for (const auto& f : map.floors) {
    c.floors.push_back(f.metadata.id);
    if (f.metadata.altitude_m) c.altitude[f.metadata.id] = *f.metadata.altitude_m;
  }
  std::sort(c.floors.begin(), c.floors.end());
  c.root = c.floors.front();

  // 1. constraints, common to every formulation
  add_transition_constraints(c);
  add_match_and_anchor_constraints(c);
  add_height_constraints(c);
  // 2. initial values propagated from the root, then the structural stacking of every pair of
  // floors
  propagate_initial_values(c);
  if (opt.auto_stacking) add_stacking_constraints(c);
  for (FloorId f : c.floors) {
    if (!c.init.count(f)) {
      c.init[f] = Pose4();
      res.warnings.push_back("floor " + std::to_string(f) + " is not connected to floor " +
                             std::to_string(c.root) + " by any constraint; left at the origin");
    }
  }
  // 3-5. graph, optimization, results
  const auto t_graph = std::chrono::steady_clock::now();
  const FusionGraph g = build_graph(c);
  const auto solved = optimize_graph(c, g);
  if (!solved) return solved.error();
  res.optimize_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_graph).count();
  collect_results(c, g, solved.value().first, solved.value().second);
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return res;
}

std::string FusionResult::transforms_yaml() const {
  YAML::Emitter out;
  out.SetDoublePrecision(12);
  out << YAML::BeginMap << YAML::Key << "formulation" << YAML::Value << to_string(formulation)
      << YAML::Key << "floors" << YAML::Value << YAML::BeginSeq;
  for (const auto& [f, X] : floor_to_map) {
    out << YAML::BeginMap << YAML::Key << "floor" << YAML::Value << f << YAML::Key << "theta"
        << YAML::Value << X.theta() << YAML::Key << "t" << YAML::Value << YAML::Flow
        << YAML::BeginSeq << X.translation().x() << X.translation().y() << X.translation().z()
        << YAML::EndSeq << YAML::EndMap;
  }
  out << YAML::EndSeq << YAML::EndMap;
  return out.c_str();
}

std::string FusionResult::report_yaml() const {
  YAML::Emitter out;
  out.SetDoublePrecision(9);
  out << YAML::BeginMap << YAML::Key << "formulation" << YAML::Value << to_string(formulation)
      << YAML::Key << "converged" << YAML::Value << converged << YAML::Key << "final_error"
      << YAML::Value << final_error << YAML::Key << "seconds" << YAML::Value << seconds;
  out << YAML::Key << "constraints" << YAML::Value << YAML::BeginSeq;
  for (const auto& c : constraints) {
    out << YAML::BeginMap << YAML::Key << "kind" << YAML::Value << c.kind << YAML::Key << "label"
        << YAML::Value << c.label << YAML::Key << "floors" << YAML::Value << YAML::Flow
        << YAML::BeginSeq << c.a << c.b << YAML::EndSeq << YAML::Key << "accepted" << YAML::Value
        << c.accepted << YAML::Key << "weight" << YAML::Value << c.weight << YAML::Key << "residual"
        << YAML::Value << c.residual << YAML::EndMap;
  }
  out << YAML::EndSeq << YAML::Key << "floors" << YAML::Value << YAML::BeginSeq;
  for (const auto& [f, C] : covariance) {
    out << YAML::BeginMap << YAML::Key << "floor" << YAML::Value << f << YAML::Key << "sigma"
        << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (int i = 0; i < 4; ++i) out << std::sqrt(std::max(0.0, C(i, i)));
    out << YAML::EndSeq << YAML::EndMap;  // (yaw rad, x m, y m, z m)
  }
  out << YAML::EndSeq << YAML::Key << "same_level_candidates" << YAML::Value << YAML::BeginSeq;
  for (const auto& c : same_level_candidates) {
    out << YAML::BeginMap << YAML::Key << "floors" << YAML::Value << YAML::Flow << YAML::BeginSeq
        << c.a << c.b << YAML::EndSeq << YAML::Key << "dz" << YAML::Value << c.dz << YAML::Key
        << "score" << YAML::Value << c.score << YAML::Key << "structure_ambiguous" << YAML::Value
        << c.structure_ambiguous << YAML::EndMap;
  }
  out << YAML::EndSeq << YAML::Key << "warnings" << YAML::Value << YAML::BeginSeq;
  for (const auto& w : warnings) out << w;
  out << YAML::EndSeq << YAML::EndMap;
  return out.c_str();
}

Result<void> write_fusion_outputs(const std::string& dir, const MapData& map,
                                  const FusionResult& r) {
  try {
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/floor_transforms.yaml") << r.transforms_yaml();
    std::ofstream(dir + "/fusion_report.yaml") << r.report_yaml();
    PointCloud global;
    for (const auto& f : map.floors) {
      if (!f.floor_cloud || !r.floor_to_map.count(f.metadata.id)) continue;
      const PointCloud& cloud = *f.floor_cloud;
      const Pose4& X = r.floor_to_map.at(f.metadata.id);
      for (size_t i = 0; i < cloud.points.size(); ++i) {
        global.points.push_back(X.act(cloud.points[i].cast<double>()).cast<float>());
        if (!cloud.colors.empty()) global.colors.push_back(cloud.colors[i]);
      }
    }
    if (!global.colors.empty() && global.colors.size() != global.points.size())
      global.colors.clear();
    return write_pcd(dir + "/global_cloud.pcd", global);
  } catch (const std::exception& e) {
    return Error{e.what()};
  }
}

Result<std::vector<AnchorSet>> load_anchors(const std::string& path) {
  try {
    const YAML::Node root = YAML::LoadFile(path);
    std::vector<AnchorSet> out;
    for (const auto& n : root["anchors"]) {
      AnchorSet s;
      s.name = n["name"].as<std::string>("anchor");
      s.use_z = n["use_z"].as<bool>(false);
      s.sigma = n["sigma"].as<double>(0.03);
      for (const auto& kv : n["points"]) {
        std::vector<Eigen::Vector3d> pts;
        for (const auto& p : kv.second)
          pts.emplace_back(p[0].as<double>(), p[1].as<double>(), p[2].as<double>());
        s.points[kv.first.as<FloorId>()] = std::move(pts);
      }
      out.push_back(std::move(s));
    }
    return out;
  } catch (const std::exception& e) {
    return Error{path + ": " + e.what()};
  }
}

}  // namespace strataslam
