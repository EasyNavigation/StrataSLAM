// Diagnostic: runs a scenario with tuning overrides (key=value) and prints its report.
// Keys: odom_t, odom_r, loop_t, loop_r, degraded, cooldown, bow_min, bridge_t, bridge_r
#include <cstdio>
#include <cstdlib>
#include <string>

#include "strataslam_sim/scenarios.hpp"

int main(int argc, char** argv) {
  auto spec = strataslam_sim::make_scenario(argc > 1 ? argv[1] : "S01");
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const auto eq = a.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = a.substr(0, eq);
    const double v = std::stod(a.substr(eq + 1));
    auto& b = spec.tuning.backend;
    if (k == "odom_t") b.odom_sigma_trans = v;
    else if (k == "odom_r") b.odom_sigma_rot = v;
    else if (k == "loop_t") b.loop_sigma_trans = v;
    else if (k == "loop_r") b.loop_sigma_rot = v;
    else if (k == "degraded") b.degraded_scale = v;
    else if (k == "cooldown") spec.tuning.loop_cooldown_keyframes = static_cast<int>(v);
    else if (k == "bow_min") spec.tuning.bow_min_score = v;
    else if (k == "bridge_t") b.bridge_sigma_trans = v;
    else if (k == "bridge_r") b.bridge_sigma_rot = v;
  }
  const auto r = strataslam_sim::run_scenario(spec);
  std::printf("%s", r.report.to_markdown().c_str());
  for (const auto& f : r.failures) std::printf("FAILED: %s\n", f.c_str());
  return r.failures.empty() ? 0 : 1;
}
