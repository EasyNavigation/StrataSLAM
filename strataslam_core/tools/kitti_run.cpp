// Runs the full pipeline (LiDAR odometry, Scan Context loop closure, backend) on a KITTI raw drive
// and reports the ATE of the optimized keyframes against the OXTS (GPS/INS) ground truth. Usage:
// kitti_run <drive_sync_dir> <calib_dir> <out_prefix> [max_scans] [threads]
//   drive_sync_dir: .../2011_09_30/2011_09_30_drive_0027_sync (velodyne_points/, oxts/)
//   calib_dir:      .../2011_09_30 (calib_imu_to_velo.txt)
// Writes <out_prefix>.est.tum and <out_prefix>.gt.tum (base = OXTS/IMU frame) and prints a JSON
// line. KITTI is CC BY-NC-SA 3.0: evaluation only, never redistributed
// (scripts/download_datasets.sh).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "strataslam/evaluation/metrics.hpp"
#include "strataslam/pipeline/slam_system.hpp"

namespace {

using strataslam::Timestamp;

/// "2011-09-30 12:40:47.123456789" -> ns since the epoch (UTC).
Timestamp parse_stamp(const std::string& s) {
  std::tm tm{};
  if (s.size() < 19 || !strptime(s.substr(0, 19).c_str(), "%Y-%m-%d %H:%M:%S", &tm)) return 0;
  const Timestamp sec = static_cast<Timestamp>(timegm(&tm));
  Timestamp frac = 0;
  if (s.size() > 20) {
    std::string digits = s.substr(20);
    digits.resize(9, '0');
    frac = std::stoll(digits);
  }
  return sec * 1000000000LL + frac;
}

std::vector<Timestamp> read_stamps(const std::string& path) {
  std::vector<Timestamp> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line))
    if (!line.empty()) out.push_back(parse_stamp(line));
  return out;
}

std::string frame_file(const std::string& dir, size_t i, const char* ext) {
  char name[32];
  std::snprintf(name, sizeof(name), "%010zu.%s", i, ext);
  return dir + "/" + name;
}

/// KITTI devkit (convertOxtsToPose): Mercator projection with the scale of the first latitude.
gtsam::Pose3 oxts_pose(const std::string& file, double* scale) {
  std::ifstream in(file);
  double lat = 0, lon = 0, alt = 0, roll = 0, pitch = 0, yaw = 0;
  in >> lat >> lon >> alt >> roll >> pitch >> yaw;
  if (*scale == 0) *scale = std::cos(lat * M_PI / 180.0);
  constexpr double er = 6378137.0;
  const double x = *scale * lon * M_PI * er / 180.0;
  const double y = *scale * er * std::log(std::tan((90.0 + lat) * M_PI / 360.0));
  return gtsam::Pose3(gtsam::Rot3::Ypr(yaw, pitch, roll), gtsam::Point3(x, y, alt));
}

/// calib_imu_to_velo.txt: x_velo = R x_imu + T, i.e. T_velo_imu.
gtsam::Pose3 read_imu_to_velo(const std::string& file) {
  std::ifstream in(file);
  std::string line;
  gtsam::Matrix3 R = gtsam::Matrix3::Identity();
  gtsam::Vector3 T = gtsam::Vector3::Zero();
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key == "R:")
      for (int i = 0; i < 9; ++i) ls >> R(i / 3, i % 3);
    if (key == "T:")
      for (int i = 0; i < 3; ++i) ls >> T(i);
  }
  return gtsam::Pose3(gtsam::Rot3(R), T);
}

void write_tum(const std::string& path, const std::vector<strataslam::eval::TimedPose>& poses) {
  std::ofstream out(path);
  out.precision(12);
  for (const auto& p : poses) {
    const Eigen::Quaterniond q(p.pose.rotation().matrix());
    out << strataslam::eval::tum_stamp(p.stamp) << ' ' << p.pose.x() << ' ' << p.pose.y() << ' '
        << p.pose.z() << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <drive_sync_dir> <calib_dir> <out_prefix> [max_scans] [threads]\n",
                 argv[0]);
    return 2;
  }
  const std::string drive = argv[1], calib = argv[2], prefix = argv[3];
  const size_t max_scans = argc > 4 ? std::stoul(argv[4]) : 1000000;
  const auto lidar_stamps = read_stamps(drive + "/velodyne_points/timestamps.txt");
  const auto oxts_stamps = read_stamps(drive + "/oxts/timestamps.txt");
  if (lidar_stamps.empty() || oxts_stamps.empty()) {
    std::fprintf(stderr, "missing velodyne_points/timestamps.txt or oxts/timestamps.txt in %s\n",
                 drive.c_str());
    return 1;
  }

  // Ground truth: OXTS poses (base = IMU frame), relative to the first one.
  std::vector<strataslam::eval::TimedPose> gt;
  double scale = 0;
  gtsam::Pose3 first;
  for (size_t i = 0; i < oxts_stamps.size(); ++i) {
    const gtsam::Pose3 p = oxts_pose(frame_file(drive + "/oxts/data", i, "txt"), &scale);
    if (i == 0) first = p;
    gt.push_back({oxts_stamps[i], first.between(p)});
  }

  // Profile: LiDAR only, one level (an outdoor drive has slopes, not floors).
  strataslam::Config cfg;
  cfg.sensors.lidar.enabled = true;
  cfg.sensors.lidar.topic = "/kitti/velo";
  cfg.sensors.lidar.min_range = 2.0;
  cfg.sensors.lidar.max_range = 80.0;
  cfg.multifloor.enabled = false;
  cfg.multifloor.floor_height_prior = std::nullopt;
  cfg.output.keyframe_translation = 1.0;
  cfg.runtime.threads = argc > 5 ? std::atoi(argv[5]) : 1;
  strataslam::Extrinsics ext;
  ext.T_base_lidar = read_imu_to_velo(calib + "/calib_imu_to_velo.txt").inverse();  // T_imu_velo
  strataslam::SlamSystem slam(cfg, ext);

  const auto t0 = std::chrono::steady_clock::now();
  size_t scans = 0;
  for (size_t i = 0; i < lidar_stamps.size() && scans < max_scans; ++i) {
    std::ifstream in(frame_file(drive + "/velodyne_points/data", i, "bin"), std::ios::binary);
    if (!in) continue;
    strataslam::LidarScan scan;
    scan.stamp = lidar_stamps[i];
    float v[4];  // x, y, z, reflectance
    while (in.read(reinterpret_cast<char*>(v), sizeof(v)))
      scan.points.emplace_back(v[0], v[1], v[2]);
    slam.add_lidar(scan);
    ++scans;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const strataslam::MapData map = slam.finalize(
      "# generated by strataslam_kitti_run (KITTI raw, LiDAR only); the configuration is the "
      "tool's own (strataslam_core/tools/kitti_run.cpp)\n");

  std::vector<strataslam::eval::TimedPose> est;
  for (const auto& f : map.floors)
    for (const auto& k : f.keyframes)
      est.push_back({k.stamp_ns, gtsam::Pose3(gtsam::Rot3(k.q.toRotationMatrix()), k.t)});
  std::sort(est.begin(), est.end(), [](const auto& a, const auto& b) { return a.stamp < b.stamp; });
  const auto se3 = strataslam::eval::ate(est, gt, strataslam::eval::AlignMode::SE3);
  write_tum(prefix + ".est.tum", est);
  write_tum(prefix + ".gt.tum", gt);
  double length = 0;
  for (size_t i = 1; i < gt.size(); ++i)
    length += (gt[i].pose.translation() - gt[i - 1].pose.translation()).norm();
  std::printf(
      "{\"scans\":%zu,\"keyframes\":%zu,\"floors\":%zu,\"loops\":%zu,\"path_length_m\":%.1f,\"ate_"
      "rmse_m\":%.4f,\"ate_max_m\":%.4f,"
      "\"ate_n\":%zu,\"seconds\":%.1f,\"ms_per_scan\":%.1f}\n",
      scans, est.size(), map.floors.size(), slam.loops().size(), length, se3.rmse, se3.max, se3.n,
      seconds, 1e3 * seconds / std::max<size_t>(1, scans));
  return 0;
}
