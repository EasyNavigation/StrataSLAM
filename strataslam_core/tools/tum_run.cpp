// Runs the visual frontend on a TUM RGB-D sequence and reports the ATE against its ground truth.
// Usage: tum_run <sequence_dir> <out.tum> [max_frames] [stride]
// Intrinsics are picked from the sequence directory name (freiburg1/2/3).
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>


#include "strataslam/io/image_io.hpp"
#include "strataslam/evaluation/metrics.hpp"
#include "strataslam/frontend/feature_extractor.hpp"
#include "strataslam/frontend/visual_frontend.hpp"

namespace {

struct Entry { double t; std::string file; };

std::vector<Entry> read_list(const std::string& path) {
  std::vector<Entry> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    Entry e;
    if (ls >> e.t >> e.file) out.push_back(e);
  }
  return out;
}

std::vector<strataslam::eval::TimedPose> read_tum(const std::string& path) {
  std::vector<strataslam::eval::TimedPose> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    double t, x, y, z, qx, qy, qz, qw;
    if (!(ls >> t >> x >> y >> z >> qx >> qy >> qz >> qw)) continue;
    out.push_back({static_cast<strataslam::Timestamp>(t * 1e9),
                   gtsam::Pose3(gtsam::Rot3(Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix()), Eigen::Vector3d(x, y, z))});
  }
  return out;
}

strataslam::CameraIntrinsics intrinsics_for(const std::string& dir) {
  strataslam::CameraIntrinsics K;
  K.width = 640;
  K.height = 480;
  if (dir.find("freiburg2") != std::string::npos) { K.fx = 520.9; K.fy = 521.0; K.cx = 325.1; K.cy = 249.7; }
  else if (dir.find("freiburg3") != std::string::npos) { K.fx = 535.4; K.fy = 539.2; K.cx = 320.1; K.cy = 247.6; }
  else { K.fx = 517.3; K.fy = 516.5; K.cx = 318.6; K.cy = 255.3; }
  return K;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <sequence_dir> <out.tum> [max_frames] [stride]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const size_t max_frames = argc > 3 ? std::stoul(argv[3]) : 1000000;
  const size_t stride = argc > 4 ? std::stoul(argv[4]) : 1;
  const auto rgb = read_list(dir + "/rgb.txt"), depth = read_list(dir + "/depth.txt");
  const auto gt = read_tum(dir + "/groundtruth.txt");
  if (rgb.empty() || depth.empty() || gt.empty()) {
    std::fprintf(stderr, "missing rgb.txt/depth.txt/groundtruth.txt in %s\n", dir.c_str());
    return 1;
  }
  const auto K = intrinsics_for(dir);
  auto orb = strataslam::make_orb_extractor(1000);
  strataslam::VisualFrontendConfig cfg;
  cfg.tracker.depth.depth_max = 6.0;
  strataslam::VisualFrontend fe(cfg, gtsam::Pose3());
  // Start at the ground-truth pose of the first frame so the estimate is directly comparable.
  gtsam::Pose3 g0;
  strataslam::eval::interpolate_pose(gt, static_cast<strataslam::Timestamp>(rgb.front().t * 1e9), &g0);
  fe.reset(g0);

  std::vector<strataslam::eval::TimedPose> est;
  size_t processed = 0, tracked = 0, di = 0;
  for (size_t i = 0; i < rgb.size() && processed < max_frames; i += stride) {
    while (di + 1 < depth.size() && std::abs(depth[di + 1].t - rgb[i].t) < std::abs(depth[di].t - rgb[i].t)) ++di;
    if (std::abs(depth[di].t - rgb[i].t) > 0.03) continue;
    // libpng, not OpenCV's imgcodecs, which loads GDAL and poppler (GPL); the pixels are the same (ADR-0047).
    auto gray = strataslam::read_gray_image(dir + "/" + rgb[i].file);
    auto d16 = strataslam::read_png16(dir + "/" + depth[di].file);
    if (!gray || !d16) continue;
    strataslam::GrayImage gi = std::move(gray.value());
    strataslam::RgbdFrame f;
    f.stamp = static_cast<strataslam::Timestamp>(rgb[i].t * 1e9);
    f.K = K;
    f.depth.width = d16.value().width;
    f.depth.height = d16.value().height;
    f.depth.meters.resize(d16.value().data.size());
    for (size_t k = 0; k < d16.value().data.size(); ++k) f.depth.meters[k] = d16.value().data[k] / 5000.0f;
    // As the node does: keypoints only where there is a depth (they are the ones the tracker can use).
    const auto with_depth = strataslam::valid_depth_mask(f.depth, cfg.tracker.depth.depth_min, cfg.tracker.depth.depth_max);
    f.keypoints = orb->extract(gi, &with_depth);
    strataslam::SensorFrame sf;
    sf.stamp = f.stamp;
    sf.rgbd = &f;
    const auto r = fe.process(sf);
    ++processed;
    if (r.valid) {
      ++tracked;
      est.push_back({f.stamp, r.T_odom_base});
    }
  }
  std::ofstream out(argv[2]);
  out.precision(12);
  for (const auto& e : est) {
    const Eigen::Quaterniond q(e.pose.rotation().matrix());
    out << strataslam::eval::tum_stamp(e.stamp) << ' ' << e.pose.x() << ' ' << e.pose.y() << ' ' << e.pose.z() << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
  }
  const auto ate = strataslam::eval::ate(est, gt, strataslam::eval::AlignMode::SE3);
  std::printf("{\"sequence\": \"%s\", \"frames\": %zu, \"tracked\": %zu, \"ate_rmse\": %.5f, \"ate_max\": %.5f}\n", dir.c_str(), processed, tracked, ate.rmse, ate.max);
  return 0;
}
