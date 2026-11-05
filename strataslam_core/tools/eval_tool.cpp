// Computes ATE / RPE between two TUM trajectories (used to cross-check the Python implementation).
// Usage: eval_tool <est.tum> <gt.tum> <none|se3|4dof> [rpe_delta_s]
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "strataslam/evaluation/metrics.hpp"

namespace {
std::vector<strataslam::eval::TimedPose> read_tum(const std::string& path) {
  std::vector<strataslam::eval::TimedPose> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    double t, x, y, z, qx, qy, qz, qw;
    if (!(ls >> t >> x >> y >> z >> qx >> qy >> qz >> qw)) continue;
    out.push_back({static_cast<strataslam::Timestamp>(std::llround(t * 1e9)),
                   gtsam::Pose3(gtsam::Rot3(Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix()),
                                Eigen::Vector3d(x, y, z))});
  }
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s est.tum gt.tum none|se3|4dof [rpe_delta_s]\n", argv[0]);
    return 2;
  }
  using namespace strataslam::eval;
  const std::string mode = argv[3];
  const AlignMode m = mode == "se3" ? AlignMode::SE3 : mode == "4dof" ? AlignMode::FourDof : AlignMode::None;
  const auto est = read_tum(argv[1]), gt = read_tum(argv[2]);
  const AteResult a = ate(est, gt, m);
  std::printf("{\"ate_rmse\": %.12g, \"ate_mean\": %.12g, \"ate_max\": %.12g, \"n\": %zu", a.rmse, a.mean, a.max, a.n);
  if (argc > 4) {
    const RpeResult r = rpe(est, gt, static_cast<int64_t>(std::stod(argv[4]) * 1e9));
    std::printf(", \"rpe_trans_rmse\": %.12g, \"rpe_rot_rmse_deg\": %.12g, \"rpe_n\": %zu", r.trans_rmse, r.rot_rmse_deg, r.n);
  }
  std::printf("}\n");
  return 0;
}
