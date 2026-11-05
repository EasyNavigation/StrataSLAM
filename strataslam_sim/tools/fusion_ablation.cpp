// Runs a scenario per seed once and fuses the resulting map with every formulation (same inputs).
// Usage: fusion_ablation <S07|S11> <seed_first> <seed_count>
// Output: one JSON object per line: {scenario, seed, formulation, trans_err_m, rot_err_deg, seconds, converged, finite};
// `seconds` is the graph construction and optimization (FusionResult::optimize_seconds), not the common stacking.
#include <cstdio>
#include <string>

#include "strataslam_sim/scenarios.hpp"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <S07|S11> <seed_first> <seed_count>\n", argv[0]);
    return 2;
  }
  const std::string id = argv[1];
  const uint64_t first = std::stoull(argv[2]);
  const int count = std::stoi(argv[3]);
  for (int i = 0; i < count; ++i) {
    const uint64_t seed = first + i;
    const auto spec = strataslam_sim::make_scenario(id, seed);
    const auto r = strataslam_sim::run_scenario(spec);
    strataslam_sim::Simulator sim(spec.building, spec.trajectory, spec.sim);
    strataslam::FusionOptions base;
    base.floor_height_prior = spec.slam.multifloor.floor_height_prior;
    strataslam_sim::AnchorChoice choice;
    choice.stairs = id == "S11";
    base.anchors = strataslam_sim::make_anchors(sim, r, seed, 0.05, choice);
    for (strataslam::Formulation f : {strataslam::Formulation::Lie4, strataslam::Formulation::Se3Soft, strataslam::Formulation::Euler}) {
      strataslam::FusionOptions o = base;
      o.formulation = f;
      const auto fused = strataslam::fuse(r.map, o);
      if (!fused.ok()) {
        std::printf("{\"scenario\": \"%s\", \"seed\": %llu, \"formulation\": \"%s\", \"converged\": false, \"finite\": false, \"trans_err_m\": null, \"rot_err_deg\": null, \"seconds\": null}\n",
                    id.c_str(), static_cast<unsigned long long>(seed), strataslam::to_string(f));
        continue;
      }
      const auto m = strataslam_sim::evaluate_fusion(spec, sim, r, fused.value());
      const bool finite = fused.value().finite() && std::isfinite(m.worst_translation) && std::isfinite(m.worst_rotation_deg);
      std::printf("{\"scenario\": \"%s\", \"seed\": %llu, \"formulation\": \"%s\", \"converged\": %s, \"finite\": %s, \"trans_err_m\": %.6f, \"rot_err_deg\": %.6f, \"seconds\": %.6f}\n",
                  id.c_str(), static_cast<unsigned long long>(seed), strataslam::to_string(f), fused.value().converged ? "true" : "false", finite ? "true" : "false",
                  m.worst_translation, m.worst_rotation_deg, fused.value().optimize_seconds);
    }
    std::fflush(stdout);
  }
  return 0;
}
