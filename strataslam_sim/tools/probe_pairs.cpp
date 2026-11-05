// Diagnostic: registers consecutive simulated scans with ground-truth initialisation.
#include <cstdio>
#include <string>

#include "strataslam/frontend/lidar_frontend.hpp"
#include "strataslam_sim/scenarios.hpp"

int main(int argc, char** argv) {
  const auto spec = strataslam_sim::make_scenario(argc > 1 ? argv[1] : "S02");
  strataslam_sim::Simulator sim(spec.building, spec.trajectory, spec.sim);
  strataslam::LidarFrontendConfig cfg;
  cfg.threads = 4;
  const size_t from = argc > 2 ? std::stoul(argv[2]) : 30, to = argc > 3 ? std::stoul(argv[3]) : 50;
  for (size_t i = from; i < to; ++i) {
    const auto a = sim.lidar(i - 1), b = sim.lidar(i);
    const auto Ta = sim.gt_pose(a.stamp) * spec.sim.extrinsics.T_base_lidar, Tb = sim.gt_pose(b.stamp) * spec.sim.extrinsics.T_base_lidar;
    const auto rel = Ta.between(Tb);
    const auto r = strataslam::LidarFrontend::register_clouds(a.points, b.points, rel, cfg);
    const auto err = r.T_target_source.between(rel);
    std::printf("%zu rel=%.3f,%.3f,%.3f  err=%.4f fit=%.2f mineig=%.3f conv=%d\n", i, rel.x(), rel.y(), rel.z(), err.translation().norm(),
                r.quality.fitness, r.quality.min_eigenvalue, r.converged);
  }
}
