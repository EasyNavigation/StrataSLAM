#include "strataslam_sim/simulator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "strataslam_sim/rng.hpp"

namespace strataslam_sim {

using strataslam::Descriptor;
using Eigen::Vector3d;
using Type = SensorEvent::Type;

namespace {
constexpr double kRd = 287.05;

uint64_t stream_seed(uint64_t seed, int stream, uint64_t idx) { return hash_combine(seed, 1000 + stream, idx); }
}  // namespace

Simulator::Simulator(std::shared_ptr<const Building> b, std::shared_ptr<const Trajectory> tr, const SimConfig& cfg)
    : building_(std::move(b)), trajectory_(std::move(tr)), cfg_(cfg) {
  const double T = trajectory_->duration();
  const auto add = [&](Type type, bool enabled, double hz) {
    if (!enabled) return;
    const size_t n = static_cast<size_t>(std::floor(T * hz)) + 1;
    counts_[static_cast<int>(type)] = n;
    for (size_t i = 0; i < n; ++i) events_.push_back({stamp_at(i / hz), type, i});
  };
  add(Type::Imu, cfg_.imu, cfg_.imu_hz);
  add(Type::Baro, cfg_.baro, cfg_.baro_hz);
  add(Type::Odom, cfg_.external_odom, cfg_.odom_hz);
  add(Type::Rgbd, cfg_.camera, cfg_.camera_hz);
  add(Type::Lidar, cfg_.lidar, cfg_.lidar_hz);
  std::stable_sort(events_.begin(), events_.end(), [](const SensorEvent& a, const SensorEvent& b) {
    if (a.stamp != b.stamp) return a.stamp < b.stamp;
    return static_cast<int>(a.type) < static_cast<int>(b.type);
  });

  // Sequential random walks (biases, pressure drift) are precomputed from the seed.
  if (cfg_.imu) {
    Rng rng(stream_seed(cfg_.seed, 90, 0));
    Vector3d ba, bg;
    for (int k = 0; k < 3; ++k) {
      ba[k] = cfg_.accel_bias_init * rng.normal();
      bg[k] = cfg_.gyro_bias_init * rng.normal();
    }
    accel_bias_.resize(counts_[0]);
    gyro_bias_.resize(counts_[0]);
    const double dt = 1.0 / cfg_.imu_hz;
    for (size_t i = 0; i < counts_[0]; ++i) {
      accel_bias_[i] = ba;
      gyro_bias_[i] = bg;
      for (int k = 0; k < 3; ++k) {
        ba[k] += cfg_.accel_random_walk * std::sqrt(dt) * rng.normal();
        bg[k] += cfg_.gyro_random_walk * std::sqrt(dt) * rng.normal();
      }
    }
  }
  if (cfg_.baro) {
    Rng rng(stream_seed(cfg_.seed, 91, 0));
    double d = 0;
    baro_drift_.resize(counts_[1]);
    for (size_t i = 0; i < counts_[1]; ++i) {
      baro_drift_[i] = d;
      d += cfg_.baro_drift_pa * std::sqrt(1.0 / cfg_.baro_hz) * rng.normal();
    }
  }
  if (cfg_.external_odom) {
    Rng rng(stream_seed(cfg_.seed, 92, 0));
    odom_poses_.resize(counts_[2]);
    gtsam::Pose3 prev_gt = trajectory_->pose_at(0), cur;  // odom frame = initial base frame
    gtsam::Pose3 pose;
    for (size_t i = 0; i < counts_[2]; ++i) {
      const gtsam::Pose3 gt = trajectory_->pose_at(i / cfg_.odom_hz);
      if (i > 0) {
        gtsam::Pose3 d = prev_gt.between(gt);
        const double scale = 1.0 + cfg_.odom_scale_sigma * rng.normal();
        Vector3d tn(cfg_.odom_trans_sigma * rng.normal(), cfg_.odom_trans_sigma * rng.normal(),
                    cfg_.odom_trans_sigma * rng.normal());
        Vector3d rn(cfg_.odom_rot_sigma * rng.normal(), cfg_.odom_rot_sigma * rng.normal(),
                    cfg_.odom_rot_sigma * rng.normal());
        d = gtsam::Pose3(d.rotation() * gtsam::Rot3::Expmap(rn), Vector3d(scale * d.translation() + tn));
        pose = pose * d;
      }
      odom_poses_[i] = pose;
      prev_gt = gt;
    }
  }
}

double Simulator::pressure_at(double z) const {
  return cfg_.base_pressure_pa * std::exp(-cfg_.gravity * z / (kRd * cfg_.temperature_k));
}

strataslam::CameraIntrinsics Simulator::intrinsics() const {
  strataslam::CameraIntrinsics K;
  K.width = cfg_.image_width;
  K.height = cfg_.image_height;
  K.fx = K.fy = 0.5 * K.width / std::tan(0.5 * cfg_.hfov_deg * M_PI / 180.0);
  K.cx = 0.5 * K.width - 0.5;
  K.cy = 0.5 * K.height - 0.5;
  return K;
}

strataslam::ImuSample Simulator::imu(size_t i) const {
  const double t = i / cfg_.imu_hz;
  const TrajectorySample s = trajectory_->state_at(t);
  Rng rng(stream_seed(cfg_.seed, 1, i));
  const double sa = cfg_.accel_noise_density * std::sqrt(cfg_.imu_hz), sg = cfg_.gyro_noise_density * std::sqrt(cfg_.imu_hz);
  // Sample's R / omega come from the nearest 200 Hz sample; use the extrinsic for non-base IMUs.
  const Eigen::Matrix3d Rib = cfg_.extrinsics.T_base_imu.rotation().matrix();
  strataslam::ImuSample m;
  m.stamp = stamp_at(t);
  const Vector3d f_base = s.R.transpose() * (s.a + Vector3d(0, 0, cfg_.gravity));
  m.accel = Rib.transpose() * f_base + accel_bias_[i] + sa * Vector3d(rng.normal(), rng.normal(), rng.normal());
  m.gyro = Rib.transpose() * s.omega_body + gyro_bias_[i] + sg * Vector3d(rng.normal(), rng.normal(), rng.normal());
  return m;
}

strataslam::BaroSample Simulator::baro(size_t i) const {
  const double t = i / cfg_.baro_hz;
  const double z = trajectory_->pose_at(t).z();
  Rng rng(stream_seed(cfg_.seed, 2, i));
  return {stamp_at(t), pressure_at(z) + baro_drift_[i] + cfg_.baro_sigma_pa * rng.normal()};
}

strataslam::OdomMeasurement Simulator::odom(size_t i) const { return {stamp_at(i / cfg_.odom_hz), odom_poses_[i]}; }

strataslam::RgbdFrame Simulator::rgbd(size_t idx) const {
  const double t = idx / cfg_.camera_hz;
  const TrajectorySample st = trajectory_->state_at(t);
  const gtsam::Pose3 T_wb = trajectory_->pose_at(t);
  const gtsam::Pose3 T_wc = T_wb * cfg_.extrinsics.T_base_camera;
  const gtsam::Pose3 T_cw = T_wc.inverse();
  const Eigen::Matrix3d Rwc = T_wc.rotation().matrix();
  const Vector3d oc = T_wc.translation();
  const auto K = intrinsics();
  strataslam::RgbdFrame f;
  f.stamp = stamp_at(t);
  f.K = K;

  for (const auto& w : cfg_.camera_blackouts) {
    if (t >= w.first && t < w.second) {  // blind camera: an empty frame
      f.depth.width = K.width;
      f.depth.height = K.height;
      f.depth.meters.assign(static_cast<size_t>(K.width) * K.height, 0.f);
      return f;
    }
  }

  const RayScene scene(*building_, st.cabin_z, st.door_closed, oc, cfg_.depth_max + 1.0);

  // Depth image.
  f.depth.width = K.width;
  f.depth.height = K.height;
  f.depth.meters.assign(static_cast<size_t>(K.width) * K.height, 0.f);
#pragma omp parallel for schedule(static)
  for (int v = 0; v < K.height; ++v) {
    for (int u = 0; u < K.width; ++u) {
      const Vector3d dc((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0);
      const double norm = dc.norm();
      const Vector3d dw = Rwc * (dc / norm);
      const auto hit = scene.raycast(oc, dw, cfg_.depth_max * norm);
      if (!hit) continue;
      const double z = *hit / norm;
      Rng rng(stream_seed(cfg_.seed, 3, (static_cast<uint64_t>(idx) << 20) ^ (static_cast<uint64_t>(v) * K.width + u)));
      const double zn = z + cfg_.depth_coeff * z * z * rng.normal();
      if (zn > 0.1 && zn < cfg_.depth_max) f.depth.meters[static_cast<size_t>(v) * K.width + u] = static_cast<float>(zn);
    }
  }

  // Keypoints: visible landmarks, strongest first (strength is a fixed per-landmark hash).
  struct Obs { float strength; size_t lm; float u, v; };
  std::vector<Obs> obs;
  const auto& lms = building_->landmarks();
  for (size_t li = 0; li < lms.size(); ++li) {
    const auto& lm = lms[li];
    if (lm.door && !st.door_closed) continue;
    const Vector3d pw = building_->landmark_position(lm, st.cabin_z);
    const Vector3d pc = T_cw.transformFrom(pw);
    if (pc.z() < 0.3 || pc.z() > cfg_.depth_max) continue;
    if (lm.max_dist > 0 && pc.norm() > lm.max_dist) continue;  // fine texture is only resolvable close up
    const double u = K.fx * pc.x() / pc.z() + K.cx, v = K.fy * pc.y() / pc.z() + K.cy;
    if (u < 4 || v < 4 || u > K.width - 5 || v > K.height - 5) continue;
    if (!scene.visible(oc, pw)) continue;
    const float strength = static_cast<float>(Rng(hash_combine(cfg_.seed, 55, li)).uniform());
    obs.push_back({strength, li, static_cast<float>(u), static_cast<float>(v)});
  }
  std::sort(obs.begin(), obs.end(), [](const Obs& a, const Obs& b) { return a.strength > b.strength; });
  if (static_cast<int>(obs.size()) > cfg_.max_keypoints) obs.resize(cfg_.max_keypoints);
  Rng rng(stream_seed(cfg_.seed, 4, idx));
  for (const auto& o : obs) {
    strataslam::RgbKeypoint kp;
    kp.u = o.u + static_cast<float>(cfg_.pixel_sigma * rng.normal());
    kp.v = o.v + static_cast<float>(cfg_.pixel_sigma * rng.normal());
    kp.response = o.strength;
    kp.descriptor = lms[o.lm].descriptor;
    for (int b = 0; b < cfg_.desc_flip_bits; ++b) {
      const uint32_t bit = rng.below(256);
      kp.descriptor[bit / 8] ^= static_cast<uint8_t>(1u << (bit % 8));
    }
    f.keypoints.push_back(kp);
  }
  const int n_out = static_cast<int>(cfg_.outlier_ratio * obs.size());
  for (int i = 0; i < n_out; ++i) {
    strataslam::RgbKeypoint kp;
    kp.u = static_cast<float>(rng.uniform(4, K.width - 5));
    kp.v = static_cast<float>(rng.uniform(4, K.height - 5));
    kp.response = static_cast<float>(rng.uniform(0, 0.3));
    for (auto& byte : kp.descriptor) byte = static_cast<uint8_t>(rng.below(256));
    f.keypoints.push_back(kp);
  }
  return f;
}

strataslam::LidarScan Simulator::lidar(size_t idx) const {
  const double t0 = idx / cfg_.lidar_hz;
  const int cols = cfg_.lidar_columns, beams = cfg_.lidar_beams;
  strataslam::LidarScan scan;
  scan.stamp = stamp_at(t0);
  const double sweep = 1.0 / cfg_.lidar_hz;
  std::vector<std::vector<std::pair<Eigen::Vector3f, float>>> col_pts(cols);
  const TrajectorySample mid = trajectory_->state_at(t0 + 0.5 * sweep);
  const Vector3d center = trajectory_->pose_at(t0 + 0.5 * sweep).translation();
  // Cabin height/door state are taken per column below; the scene prefilter uses the middle state.
  (void)mid;
#pragma omp parallel for schedule(static)
  for (int c = 0; c < cols; ++c) {
    const double dt = sweep * c / cols;
    const TrajectorySample st = trajectory_->state_at(t0 + dt);
    const gtsam::Pose3 T_ws = trajectory_->pose_at(t0 + dt) * cfg_.extrinsics.T_base_lidar;
    const RayScene scene(*building_, st.cabin_z, st.door_closed, T_ws.translation(), cfg_.lidar_max_range + 1.0);
    const Eigen::Matrix3d R = T_ws.rotation().matrix();
    const double az = 2 * M_PI * c / cols;
    for (int b = 0; b < beams; ++b) {
      const double el = (beams == 1 ? 0.0 : (-0.5 + static_cast<double>(b) / (beams - 1)) * cfg_.lidar_fov_deg) * M_PI / 180.0;
      const Vector3d ds(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
      const auto hit = scene.raycast(T_ws.translation(), R * ds, cfg_.lidar_max_range);
      if (!hit || *hit < cfg_.lidar_min_range) continue;
      Rng rng(stream_seed(cfg_.seed, 5, (static_cast<uint64_t>(idx) << 24) ^ (static_cast<uint64_t>(c) * beams + b)));
      const double r = *hit + cfg_.lidar_range_sigma * rng.normal();
      const Vector3d p = r * ds;
      col_pts[c].push_back({p.cast<float>(), static_cast<float>(dt)});
    }
  }
  (void)center;
  for (const auto& col : col_pts) {
    for (const auto& pt : col) {
      scan.points.push_back(pt.first);
      scan.time_offsets_s.push_back(pt.second);
    }
  }
  return scan;
}

}  // namespace strataslam_sim
