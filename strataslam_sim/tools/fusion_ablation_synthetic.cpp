// Fusion ablation on synthetic floor graphs with an exact truth, in the cases where the formulations
// can differ (ADR-0040). Each case builds floor frames with known Pose4s, relative transition
// measurements and anchor marks with noise, fuses them with the three formulations (same inputs) and
// measures the 4-DoF error of every floor against the truth.
//   wrap   two floors turned by ~180 deg: the transition and the anchors put the relative yaw on
//          either side of +-pi (a formulation that does not wrap the yaw residual sees ~360 deg).
//   chain  ten floors chained by transitions with yaw noise, anchors on the first and last only.
//   sparse two floors, one anchor set of two points with 10 cm of marking error, no transition.
//   wrap-nognc, chain-nognc: the same without GNC, which otherwise hides a wrong residual by
//          rejecting the constraint (the formulation's own behaviour shows only without it).
// Usage: fusion_ablation_synthetic <case> <seed_first> <seed_count>     (JSON lines, as fusion_ablation)
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "strataslam/evaluation/metrics.hpp"
#include "strataslam/fusion/fusion.hpp"

namespace {

using strataslam::FloorId;
using strataslam::Pose4;

struct Case {
  strataslam::MapData map;
  strataslam::FusionOptions options;
  std::map<FloorId, Pose4> truth;  ///< T_map_floor, floor 0 = identity
};

strataslam::KeyframeRecord keyframe_at_origin(strataslam::KeyframeId id) {
  strataslam::KeyframeRecord k;
  k.id = id;
  k.stamp_ns = 1000000000LL * static_cast<int64_t>(id + 1);
  return k;
}

/// Floors 0..n-1 with the given true poses; transition i -> i+1 measured with yaw/xy/z noise.
Case make_case(const std::vector<Pose4>& truth, std::mt19937_64& rng, bool transitions, double yaw_sigma, double xy_sigma,
               const std::vector<std::pair<FloorId, FloorId>>& anchored, int anchor_points, double anchor_sigma) {
  std::normal_distribution<double> n01(0.0, 1.0);
  Case c;
  for (size_t f = 0; f < truth.size(); ++f) {
    strataslam::FloorData fd;
    fd.metadata.id = static_cast<FloorId>(f);
    fd.keyframes = {keyframe_at_origin(10 * f), keyframe_at_origin(10 * f + 1)};  // kf_after, kf_before (both at the frame origin)
    c.map.floors.push_back(fd);
    c.truth[static_cast<FloorId>(f)] = truth[0].inverse() * truth[f];
  }
  if (transitions) {
    for (size_t f = 0; f + 1 < truth.size(); ++f) {
      const Pose4 rel = truth[f].inverse() * truth[f + 1];
      strataslam::TransitionRecord t;
      t.type = strataslam::TransitionType::Stairs;
      t.from = static_cast<FloorId>(f);
      t.to = static_cast<FloorId>(f + 1);
      t.kf_before = 10 * f + 1;
      t.kf_after = 10 * (f + 1);
      t.continuous_tracking = true;
      t.delta_z = rel.translation().z();
      t.relative = std::array<double, 4>{rel.theta() + yaw_sigma * n01(rng), rel.translation().x() + xy_sigma * n01(rng),
                                         rel.translation().y() + xy_sigma * n01(rng), rel.translation().z() + 0.02 * n01(rng)};
      Eigen::Matrix4d cov = Eigen::Vector4d(yaw_sigma * yaw_sigma, xy_sigma * xy_sigma, xy_sigma * xy_sigma, 0.02 * 0.02).asDiagonal();
      t.covariance = cov;
      c.map.transitions.push_back(t);
    }
  }
  // Anchor marks: the same world points (a shaft, 4 m wide) seen from each anchored floor.
  for (const auto& [a, b] : anchored) {
    strataslam::AnchorSet set;
    set.name = "shaft " + std::to_string(a) + "-" + std::to_string(b);
    set.sigma = anchor_sigma;
    for (int i = 0; i < anchor_points; ++i) {
      const Eigen::Vector3d world(8.0 + 4.0 * (i % 2), 3.0 + 4.0 * (i / 2), 1.0);
      for (FloorId f : {a, b}) {
        const Pose4& X = truth[static_cast<size_t>(f)];
        const Eigen::Vector3d at_level = world + Eigen::Vector3d(0, 0, X.translation().z());
        set.points[f].push_back(X.inverse().act(at_level) + anchor_sigma * Eigen::Vector3d(n01(rng), n01(rng), n01(rng)));
      }
    }
    c.options.anchors.push_back(set);
  }
  return c;
}

Case build(const std::string& name, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  if (name == "wrap" || name == "wrap-nognc") {
    // 180 deg +- 0.3 deg: the measured relative yaw falls on either side of +-pi from seed to seed.
    const std::vector<Pose4> truth = {Pose4(), Pose4(M_PI - 0.003 * (1.0 + u(rng)), Eigen::Vector3d(20.0, 10.0, 3.0))};
    return make_case(truth, rng, true, 0.004, 0.05, {{0, 1}}, 4, 0.03);
  }
  if (name == "chain" || name == "chain-nognc") {
    std::vector<Pose4> truth;
    for (int f = 0; f < 10; ++f) truth.emplace_back(0.3 * f + 0.05 * u(rng), Eigen::Vector3d(2.0 * u(rng), 2.0 * u(rng), 3.0 * f));
    return make_case(truth, rng, true, 0.01, 0.05, {{0, 9}}, 4, 0.03);
  }
  // sparse
  const std::vector<Pose4> truth = {Pose4(), Pose4(1.0 + 0.5 * u(rng), Eigen::Vector3d(5.0 * u(rng), 5.0 * u(rng), 3.0))};
  Case c = make_case(truth, rng, false, 0.0, 0.0, {{0, 1}}, 2, 0.10);
  c.options.floor_height_prior = 3.0;
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <wrap|chain|sparse|wrap-nognc|chain-nognc> <seed_first> <seed_count>\n", argv[0]);
    return 2;
  }
  const std::string name = argv[1];
  const uint64_t first = std::stoull(argv[2]);
  const int count = std::stoi(argv[3]);
  for (int i = 0; i < count; ++i) {
    const uint64_t seed = first + static_cast<uint64_t>(i);
    const Case c = build(name, seed);
    for (strataslam::Formulation f : {strataslam::Formulation::Lie4, strataslam::Formulation::Se3Soft, strataslam::Formulation::Euler}) {
      strataslam::FusionOptions o = c.options;
      o.formulation = f;
      o.auto_stacking = false;
      o.use_gnc = name.find("-nognc") == std::string::npos;
      const auto fused = strataslam::fuse(c.map, o);
      if (!fused.ok()) {
        std::printf("{\"scenario\": \"%s\", \"seed\": %llu, \"formulation\": \"%s\", \"converged\": false, \"finite\": false, \"trans_err_m\": null, \"rot_err_deg\": null, \"seconds\": null}\n",
                    name.c_str(), static_cast<unsigned long long>(seed), strataslam::to_string(f));
        continue;
      }
      const auto errors = strataslam::eval::fusion_errors(fused.value().floor_to_map, c.truth);
      double t = 0, r = 0;
      for (const auto& [fid, e] : errors) {
        t = std::max(t, e.trans_m);
        r = std::max(r, e.rot_deg);
      }
      const bool finite = fused.value().finite() && std::isfinite(t) && std::isfinite(r);
      std::printf("{\"scenario\": \"%s\", \"seed\": %llu, \"formulation\": \"%s\", \"converged\": %s, \"finite\": %s, \"trans_err_m\": %.6f, \"rot_err_deg\": %.6f, \"seconds\": %.6f}\n",
                  name.c_str(), static_cast<unsigned long long>(seed), strataslam::to_string(f), fused.value().converged ? "true" : "false",
                  finite ? "true" : "false", t, r, fused.value().optimize_seconds);
    }
  }
  return 0;
}
