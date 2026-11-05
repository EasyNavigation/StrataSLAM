// Diagnostic: runs the LiDAR frontend over a scenario and prints per-scan tracking quality.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "strataslam/frontend/lidar_frontend.hpp"
#include "strataslam_sim/scenarios.hpp"

int main(int argc, char** argv) {
  const std::string id = argc > 1 ? argv[1] : "S02";
  auto spec = strataslam_sim::make_scenario(id);
  if (const char* b = std::getenv("PROBE_BEAMS")) spec.sim.lidar_beams = std::atoi(b);
  if (const char* b = std::getenv("PROBE_COLUMNS")) spec.sim.lidar_columns = std::atoi(b);
  strataslam_sim::Simulator sim(spec.building, spec.trajectory, spec.sim);
  strataslam::LidarFrontendConfig cfg;
  cfg.threads = 4;
  if (argc > 3) cfg.deskew = std::stoi(argv[3]) != 0;
  if (argc > 4) cfg.max_correspondence = std::stod(argv[4]);
  if (argc > 5) { cfg.scan_voxel = std::stod(argv[5]); cfg.map_voxel = cfg.scan_voxel; }
  if (argc > 6) cfg.threads = std::stoi(argv[6]);
  if (argc > 7) cfg.pin_degenerate = std::stoi(argv[7]) != 0;
  if (argc > 8) cfg.degeneracy_threshold = std::stod(argv[8]);
  if (argc > 9) cfg.map_max_keyframes = std::stoi(argv[9]);
  if (argc > 10) cfg.map_radius = std::stod(argv[10]);
  strataslam::LidarFrontend fe(cfg, spec.sim.extrinsics.T_base_lidar);
  fe.reset(sim.gt_pose(sim.stamp_at(0)));
  const size_t n = sim.count(strataslam_sim::SensorEvent::Type::Lidar);
  for (size_t i = 0; i < n; i += (argc > 2 ? std::stoul(argv[2]) : 1)) {
    const auto scan = sim.lidar(i);
    strataslam::SensorFrame f;
    f.stamp = scan.stamp;
    f.lidar = &scan;
    const auto r = fe.process(f);
    const auto gt = sim.gt_pose(scan.stamp);
    const Eigen::Vector3d d = r.valid ? Eigen::Vector3d(r.T_odom_base.translation() - gt.translation()) : Eigen::Vector3d::Zero();
    std::printf("%4zu pts=%5zu valid=%d state=%d fit=%.2f mineig=%.3f kf=%d err=%.3f d=(%.3f %.3f %.3f) gt=(%.1f %.1f %.1f)\n", i, scan.points.size(), r.valid,
                int(r.quality.state), r.quality.fitness, r.quality.min_eigenvalue, r.is_keyframe, r.valid ? d.norm() : -1.0, d.x(), d.y(), d.z(), gt.x(), gt.y(), gt.z());
  }
}
