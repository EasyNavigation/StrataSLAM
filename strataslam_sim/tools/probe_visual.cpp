// Diagnostic: runs the visual frontend over a scenario and prints per-frame tracking quality.
#include <cstdio>
#include <string>

#include "strataslam/frontend/visual_frontend.hpp"
#include "strataslam_sim/scenarios.hpp"

int main(int argc, char** argv) {
  const std::string id = argc > 1 ? argv[1] : "S01";
  auto spec = strataslam_sim::make_scenario(id);
  if (argc > 3) spec.sim.depth_coeff = std::stod(argv[3]);
  if (argc > 4) spec.sim.pixel_sigma = std::stod(argv[4]);
  if (argc > 5) spec.sim.desc_flip_bits = std::stoi(argv[5]);
  strataslam_sim::Simulator sim(spec.building, spec.trajectory, spec.sim);
  strataslam::VisualFrontendConfig cfg;
  strataslam::VisualFrontend fe(cfg, spec.sim.extrinsics.T_base_camera);
  fe.reset(sim.gt_pose(sim.stamp_at(0)));
  const size_t n = sim.count(strataslam_sim::SensorEvent::Type::Rgbd);
  const size_t step = argc > 2 ? std::stoul(argv[2]) : 1;
  for (size_t i = 0; i < n; i += step) {
    const auto f = sim.rgbd(i);
    strataslam::SensorFrame sf;
    sf.stamp = f.stamp;
    sf.rgbd = &f;
    const auto r = fe.process(sf);
    const auto gt = sim.gt_pose(f.stamp);
    const Eigen::Vector3d d = r.valid ? Eigen::Vector3d(r.T_odom_base.translation() - gt.translation()) : Eigen::Vector3d::Zero();
    std::printf("%4zu kps=%3zu valid=%d state=%d inl=%3d fit=%.2f kf=%d err=%.3f d=(%.3f %.3f %.3f) gt=(%.1f %.1f)\n", i, f.keypoints.size(), r.valid, int(r.quality.state),
                r.quality.inliers, r.quality.fitness, r.is_keyframe, r.valid ? d.norm() : -1.0, d.x(), d.y(), d.z(), gt.x(), gt.y());
  }
}
