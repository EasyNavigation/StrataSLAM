// Visualisation helper: runs a scenario and dumps CSVs (estimated keyframes vs ground truth, floor
// clouds, building boxes) for scripts/plot_scenario.py.  Usage: strataslam_dump_scenario Sxx
// OUT_DIR [SEED]
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "strataslam/io/map_io.hpp"
#include "strataslam_sim/scenarios.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s SCENARIO OUT_DIR [SEED]\n", argv[0]);
    return 2;
  }
  const std::string out = argv[2];
  std::filesystem::create_directories(out);
  auto spec = argc > 3 ? strataslam_sim::make_scenario(argv[1], std::stoull(argv[3]))
                       : strataslam_sim::make_scenario(argv[1]);
  const auto r = strataslam_sim::run_scenario(spec);
  strataslam_sim::Simulator sim(spec.building, spec.trajectory, spec.sim);

  std::ofstream(out + "/report.md") << r.report.to_markdown();
  // The map itself, with every floor's placement: keyframes in one continuous frame (ADR-0064).
  if (const auto w = strataslam::write_map(out + "/map", r.map); !w.ok())
    std::fprintf(stderr, "map not written: %s\n", w.error().message.c_str());
  {
    std::ofstream f(out + "/boxes.csv");
    f << "level,cx,cy,cz,hx,hy,hz,r00,r01,r10,r11,dynamic,door\n";
    for (const auto& b : spec.building->boxes())
      f << b.level << ',' << b.center.x() << ',' << b.center.y() << ',' << b.center.z() << ','
        << b.half.x() << ',' << b.half.y() << ',' << b.half.z() << ',' << b.R(0, 0) << ','
        << b.R(0, 1) << ',' << b.R(1, 0) << ',' << b.R(1, 1) << ',' << b.dynamic << ',' << b.door
        << '\n';
  }
  {
    std::ofstream f(out + "/keyframes.csv");
    f << "floor,id,time,ex,ey,ez,qx,qy,qz,qw,gx,gy,gz\n";
    for (const auto& fl : r.map.floors)
      for (const auto& k : fl.keyframes) {
        const auto g = sim.gt_pose(k.stamp_ns).translation();
        f << fl.metadata.id << ',' << k.id << ',' << sim.time_of(k.stamp_ns) << ',' << k.t.x()
          << ',' << k.t.y() << ',' << k.t.z() << ',' << k.q.x() << ',' << k.q.y() << ',' << k.q.z()
          << ',' << k.q.w() << ',' << g.x() << ',' << g.y() << ',' << g.z() << '\n';
      }
  }
  {
    std::ofstream f(out + "/gt_path.csv");
    f << "t,x,y,z,true_floor\n";
    for (double t = 0; t < spec.trajectory->duration(); t += 0.1) {
      const auto p = spec.trajectory->pose_at(t).translation();
      f << t << ',' << p.x() << ',' << p.y() << ',' << p.z() << ','
        << spec.trajectory->state_at(t).true_floor << '\n';
    }
  }
  {
    // Per-keyframe clouds in the robot base frame (LiDAR extrinsic applied), thinned for plotting.
    std::ofstream f(out + "/kf_clouds.csv");
    f << "floor,id,x,y,z\n";
    const gtsam::Pose3 T_bl = spec.sim.extrinsics.T_base_lidar;
    for (const auto& fl : r.map.floors)
      for (const auto& [id, cloud] : fl.clouds) {
        const size_t step = std::max<size_t>(1, cloud.points.size() / 150);
        for (size_t i = 0; i < cloud.points.size(); i += step) {
          const Eigen::Vector3d p = T_bl * cloud.points[i].cast<double>();
          f << fl.metadata.id << ',' << id << ',' << p.x() << ',' << p.y() << ',' << p.z() << '\n';
        }
      }
  }
  for (const auto& fl : r.map.floors) {
    if (!fl.floor_cloud) continue;
    std::ofstream f(out + "/cloud_" + std::to_string(fl.metadata.id) + ".csv");
    f << "x,y,z\n";
    const auto& pts = fl.floor_cloud->points;
    const size_t step = std::max<size_t>(1, pts.size() / 60000);
    for (size_t i = 0; i < pts.size(); i += step)
      f << pts[i].x() << ',' << pts[i].y() << ',' << pts[i].z() << '\n';
  }
  std::printf("%s: %zu floors, %zu keyframes, %s\n", argv[1], r.map.floors.size(),
              r.keyframes.size(), r.passed() ? "PASSED" : "FAILED");
  return 0;
}
