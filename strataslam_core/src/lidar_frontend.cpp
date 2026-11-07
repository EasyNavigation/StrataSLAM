#include "strataslam/frontend/lidar_frontend.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <deque>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#pragma GCC diagnostic pop

#include "strataslam/geometry/gravity.hpp"
#include "strataslam/mapping/voxel_grid.hpp"

namespace strataslam {

namespace {

// ---- Registration whose result depends neither on the run nor on the number of threads (ADR-0048)
// ----
//
// small_gicp's OpenMP helpers hand out work dynamically (`schedule(guided)`): its reduction sums
// each thread's share in whatever order the threads took them, and its parallel downsampling orders
// the output points by which thread finished first. Floating-point sums in a different order give a
// different pose, and over a long run a different map (S11: 5 mm between two runs with 6 threads).
// Here the downsampling is sequential (as small_gicp's own one-thread path), the normals are
// computed per point (each one independent of the others), and the reduction splits the factors
// into a fixed number of contiguous chunks, summed in chunk order: the threads only decide who
// computes a chunk, never how the sums are grouped. One thread or twenty give the same bits.

constexpr int kReductionChunks = 16;

struct ChunkedReduction {
  int num_threads = 1;

  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree,
            typename CorrespondenceRejector, typename Factor>
  std::tuple<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>, double> linearize(
      const TargetPointCloud& target, const SourcePointCloud& source, const TargetTree& target_tree,
      const CorrespondenceRejector& rejector, const Eigen::Isometry3d& T,
      std::vector<Factor>& factors) const {
    std::vector<Eigen::Matrix<double, 6, 6>> Hs(kReductionChunks,
                                                Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> bs(kReductionChunks,
                                                Eigen::Matrix<double, 6, 1>::Zero());
    std::vector<double> es(kReductionChunks, 0.0);
    const std::int64_t n = static_cast<std::int64_t>(factors.size());
#pragma omp parallel for num_threads(num_threads) schedule(static, 1)
    for (int c = 0; c < kReductionChunks; ++c) {
      const std::int64_t begin = n * c / kReductionChunks, end = n * (c + 1) / kReductionChunks;
      for (std::int64_t i = begin; i < end; ++i) {
        Eigen::Matrix<double, 6, 6> H;
        Eigen::Matrix<double, 6, 1> b;
        double e;
        if (!factors[i].linearize(target, source, target_tree, T, i, rejector, &H, &b, &e))
          continue;
        Hs[c] += H;
        bs[c] += b;
        es[c] += e;
      }
    }
    for (int c = 1; c < kReductionChunks; ++c) {
      Hs[0] += Hs[c];
      bs[0] += bs[c];
      es[0] += es[c];
    }
    return {Hs[0], bs[0], es[0]};
  }

  template <typename TargetPointCloud, typename SourcePointCloud, typename Factor>
  double error(const TargetPointCloud& target, const SourcePointCloud& source,
               const Eigen::Isometry3d& T, std::vector<Factor>& factors) const {
    std::vector<double> es(kReductionChunks, 0.0);
    const std::int64_t n = static_cast<std::int64_t>(factors.size());
#pragma omp parallel for num_threads(num_threads) schedule(static, 1)
    for (int c = 0; c < kReductionChunks; ++c) {
      const std::int64_t begin = n * c / kReductionChunks, end = n * (c + 1) / kReductionChunks;
      for (std::int64_t i = begin; i < end; ++i) es[c] += factors[i].error(target, source, T);
    }
    double sum = 0.0;
    for (double e : es) sum += e;
    return sum;
  }
};

using Cloud = small_gicp::PointCloud;
using Tree = small_gicp::KdTree<Cloud>;

/// Downsampled cloud with normals and covariances, and its kd-tree (deterministic, see above).
std::pair<std::shared_ptr<Cloud>, std::shared_ptr<Tree>> preprocess(const Cloud& raw, double leaf,
                                                                    int neighbors, int threads) {
  auto ds = small_gicp::voxelgrid_sampling(raw, leaf);
  auto tree = std::make_shared<Tree>(ds);
  small_gicp::estimate_normals_covariances_omp(*ds, *tree, neighbors, std::max(1, threads));
  return {ds, tree};
}

small_gicp::RegistrationResult align_gicp(const Cloud& target, const Cloud& source,
                                          const Tree& target_tree, const Eigen::Isometry3d& init,
                                          double max_correspondence, int max_iterations,
                                          int threads) {
  small_gicp::Registration<small_gicp::GICPFactor, ChunkedReduction> registration;
  registration.reduction.num_threads = std::max(1, threads);
  registration.rejector.max_dist_sq = max_correspondence * max_correspondence;
  registration.criteria.translation_eps = 1e-4;
  registration.criteria.rotation_eps = 0.01 * M_PI / 180.0;
  registration.optimizer.max_iterations = max_iterations;
  return registration.align(target, source, target_tree, init);
}

/// The same with the rotation held at the initial guess's: the translation only.
small_gicp::RegistrationResult align_gicp_translation(const Cloud& target, const Cloud& source,
                                                      const Tree& target_tree,
                                                      const Eigen::Isometry3d& init,
                                                      double max_correspondence, int max_iterations,
                                                      int threads) {
  small_gicp::Registration<small_gicp::GICPFactor, ChunkedReduction, small_gicp::RestrictDoFFactor>
      registration;
  registration.general_factor.set_rotation_mask(Eigen::Array3d::Zero());
  registration.reduction.num_threads = std::max(1, threads);
  registration.rejector.max_dist_sq = max_correspondence * max_correspondence;
  registration.criteria.translation_eps = 1e-4;
  registration.criteria.rotation_eps = 0.01 * M_PI / 180.0;
  registration.optimizer.max_iterations = max_iterations;
  return registration.align(target, source, target_tree, init);
}

/// Whether 95 % of the points lie within `range` of the sensor.
bool confined_scan(const std::vector<Eigen::Vector3f>& pts, double range) {
  if (range <= 0 || pts.size() < 100) return false;
  std::vector<float> r(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) r[i] = pts[i].norm();
  const auto k = r.begin() + static_cast<std::ptrdiff_t>(0.95 * static_cast<double>(r.size()));
  std::nth_element(r.begin(), k, r.end());
  return *k < range;
}

std::vector<Eigen::Vector4d> to_homogeneous(const std::vector<Eigen::Vector3f>& pts) {
  std::vector<Eigen::Vector4d> out(pts.size());
  for (size_t i = 0; i < pts.size(); ++i)
    out[i] = Eigen::Vector4d(pts[i].x(), pts[i].y(), pts[i].z(), 1.0);
  return out;
}

Eigen::Isometry3d to_iso(const gtsam::Pose3& p) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = p.rotation().matrix();
  T.translation() = p.translation();
  return T;
}

gtsam::Pose3 from_iso(const Eigen::Isometry3d& T) {
  return gtsam::Pose3(gtsam::Rot3(T.rotation()), Eigen::Vector3d(T.translation()));
}

/// Normalized translation-Hessian spectrum: eigenvalues divided by the number of correspondences,
/// reported as (min eigenvalue / max eigenvalue). A well-constrained room is ~1e-1, a corridor
/// ~1e-3.
void spectrum(const Eigen::Matrix<double, 6, 6>& H, double* min_over_max, double* cond) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(
      H.block<3, 3>(3, 3));  // small_gicp ordering: [rot, trans]
  const double lo = std::max(0.0, es.eigenvalues()(0)), hi = std::max(1e-12, es.eigenvalues()(2));
  *min_over_max = lo / hi;
  *cond = lo > 0 ? hi / lo : 1e12;
}

}  // namespace

struct LidarFrontend::Impl {
  LidarFrontendConfig cfg;
  gtsam::Pose3 T_base_lidar;
  struct Keyframe {
    gtsam::Pose3 T_odom_lidar;
    std::vector<Eigen::Vector3f> cloud;  // lidar frame, downsampled
  };
  std::deque<Keyframe> keyframes;
  small_gicp::PointCloud::Ptr map_cloud;
  std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>> map_tree;
  bool initialized = false;
  gtsam::Pose3 T_odom_base;         // last pose
  gtsam::Pose3 last_keyframe_pose;  // T_odom_lidar of the last keyframe
  gtsam::Pose3 velocity;            // body-frame delta per scan (constant velocity model)
  std::deque<gtsam::Pose3>
      recent;  // last base poses, to estimate the velocity over a longer baseline
  Timestamp last_stamp = 0;
  TrackingQuality last_quality;
  std::vector<Eigen::Vector3f> last_kf_cloud;
  int lost_count = 0;
  bool force_kf = false;
  bool confined = false;        // the last scan saw a confined space (kept across restarts)
  Timestamp left_confined = 0;  // when the scans last came out of a confined space

  void rebuild_map(const gtsam::Pose3& T_odom_lidar_now) {
    std::vector<Eigen::Vector3f> pts;
    for (const auto& kf : keyframes) {
      if ((kf.T_odom_lidar.translation() - T_odom_lidar_now.translation()).norm() > cfg.map_radius)
        continue;
      const auto w = transform_points(kf.cloud, kf.T_odom_lidar.matrix());
      pts.insert(pts.end(), w.begin(), w.end());
    }
    const auto ds = voxel_downsample(pts, cfg.map_voxel);
    const auto h = to_homogeneous(ds);
    small_gicp::PointCloud raw(h);
    auto res = preprocess(raw, cfg.map_voxel * 0.5, cfg.num_neighbors, cfg.threads);
    map_cloud = res.first;
    map_tree = res.second;
  }
};

LidarFrontend::LidarFrontend(const LidarFrontendConfig& config, const gtsam::Pose3& T_base_lidar)
    : impl_(std::make_unique<Impl>()) {
  impl_->cfg = config;
  impl_->T_base_lidar = T_base_lidar;
}
LidarFrontend::~LidarFrontend() = default;

TrackingQuality LidarFrontend::quality() const { return impl_->last_quality; }
const std::vector<Eigen::Vector3f>& LidarFrontend::last_keyframe_cloud() const {
  return impl_->last_kf_cloud;
}

void LidarFrontend::request_keyframe() { impl_->force_kf = true; }

void LidarFrontend::reset(const gtsam::Pose3& T_init) {
  auto& s = *impl_;
  s.keyframes.clear();
  s.map_cloud.reset();
  s.map_tree.reset();
  s.initialized = false;
  s.T_odom_base = T_init;
  s.velocity = gtsam::Pose3();
  s.recent.clear();
  s.recent.push_back(T_init);
  s.lost_count = 0;
  s.last_quality = TrackingQuality();
  s.last_kf_cloud.clear();
}

LidarFrontend::RegistrationOutput LidarFrontend::register_clouds(
    const std::vector<Eigen::Vector3f>& target, const std::vector<Eigen::Vector3f>& source,
    const gtsam::Pose3& init, const LidarFrontendConfig& cfg) {
  RegistrationOutput out;
  out.T_target_source = init;
  if (target.size() < 50 || source.size() < 50) {
    out.quality.state = TrackingState::Lost;
    return out;
  }
  const auto t_h = to_homogeneous(voxel_downsample(target, cfg.scan_voxel));
  const auto s_h = to_homogeneous(voxel_downsample(source, cfg.scan_voxel));
  small_gicp::PointCloud t_raw(t_h), s_raw(s_h);
  auto tp = preprocess(t_raw, cfg.scan_voxel * 0.5, cfg.num_neighbors, cfg.threads);
  auto sp = preprocess(s_raw, cfg.scan_voxel * 0.5, cfg.num_neighbors, cfg.threads);
  const auto r = align_gicp(*tp.first, *sp.first, *tp.second, to_iso(init), cfg.max_correspondence,
                            cfg.max_iterations, cfg.threads);
  out.T_target_source = from_iso(r.T_target_source);
  out.converged = r.converged;
  out.hessian = r.H;
  const double n_src = static_cast<double>(sp.first->size());
  out.quality.inliers = static_cast<int>(r.num_inliers);
  out.quality.fitness = n_src > 0 ? r.num_inliers / n_src : 0.0;
  out.mean_error = r.num_inliers ? r.error / r.num_inliers : 0.0;
  Eigen::Matrix<double, 6, 6> Hn = r.H / std::max<double>(1.0, r.num_inliers);
  spectrum(Hn, &out.quality.min_eigenvalue, &out.quality.hessian_condition);
  out.quality.state = out.quality.fitness < cfg.min_fitness ? TrackingState::Lost
                      : out.quality.min_eigenvalue < cfg.degeneracy_threshold
                          ? TrackingState::Degraded
                          : TrackingState::Good;
  return out;
}

namespace {

/// The sensor pose with the base's roll and pitch bounded by gravity (`up_base`, from the IMU):
/// within `max_tilt_error_deg` they are the registration's own, beyond it the smallest rotation
/// brings them back to the bound; heading and sensor position (what the registration measures)
/// kept.
gtsam::Pose3 level_lidar(const gtsam::Pose3& T_ol, const gtsam::Pose3& T_base_lidar,
                         const Eigen::Vector3d& up_base, double max_tilt_error_deg) {
  const gtsam::Rot3 R_base = T_ol.rotation() * T_base_lidar.rotation().inverse();
  const Eigen::Vector3d up_now = R_base.matrix().transpose() * Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d up_imu = up_base.normalized();
  const double angle = std::acos(std::clamp(up_now.dot(up_imu), -1.0, 1.0));
  const double bound = max_tilt_error_deg * M_PI / 180.0;
  if (angle <= bound) return T_ol;
  const Eigen::Vector3d axis = up_imu.cross(up_now).normalized();
  const Eigen::Vector3d up_target = Eigen::AngleAxisd(bound, axis) * up_imu;
  return gtsam::Pose3(level_to(R_base, up_target) * T_base_lidar.rotation(), T_ol.translation());
}

}  // namespace

FrontendResult LidarFrontend::process(const SensorFrame& frame) {
  auto& s = *impl_;
  FrontendResult res;
  if (!frame.lidar) return res;
  const LidarScan& scan = *frame.lidar;

  // Range filter (+ constant-velocity deskew to the scan-start frame).
  const bool can_deskew = s.cfg.deskew && !scan.time_offsets_s.empty() && s.last_stamp > 0;
  const double dt_scan = s.last_stamp > 0 ? (frame.stamp - s.last_stamp) * 1e-9 : 0.1;
  gtsam::Vector6 twist = gtsam::Vector6::Zero();  // per-second body twist in the lidar frame
  if (can_deskew && dt_scan > 1e-3) {
    const gtsam::Pose3 vel_lidar = s.T_base_lidar.inverse() * s.velocity * s.T_base_lidar;
    twist = gtsam::Pose3::Logmap(vel_lidar) / dt_scan;
  }
  std::vector<Eigen::Vector3f> pts;
  pts.reserve(scan.points.size());
  for (size_t i = 0; i < scan.points.size(); ++i) {
    const float r = scan.points[i].norm();
    if (r < s.cfg.min_range || r > s.cfg.max_range) continue;
    if (can_deskew && i < scan.time_offsets_s.size()) {
      // The point was measured in the sensor frame at t_i; express it in the frame at scan start.
      const gtsam::Pose3 T_start_i = gtsam::Pose3::Expmap(twist * scan.time_offsets_s[i]);
      const Eigen::Vector3d pv = scan.points[i].cast<double>();
      pts.push_back(T_start_i.transformFrom(pv).cast<float>());
    } else {
      pts.push_back(scan.points[i]);
    }
  }
  // A confined space (with hysteresis: in once the scan is within the range, out once it sees twice
  // as far, since the door of a cabin, half open, still shows mostly the cabin and the registration
  // turns wrong there) is sampled twice as finely: a cabin of 1.5 m is about a hundred cells of
  // 0.4 m, and below that the scan is not tracked at all.
  res.confined = confined_scan(pts, s.confined ? 2.0 * s.cfg.confined_range : s.cfg.confined_range);
  if (s.confined && !res.confined) s.left_confined = frame.stamp;
  s.confined = res.confined;
  const bool gyro_led = res.confined && frame.gyro_delta;
  const auto cloud = voxel_downsample(pts, gyro_led ? 0.5 * s.cfg.scan_voxel : s.cfg.scan_voxel);
  if (cloud.size() < 100) {
    s.last_quality.state = TrackingState::Lost;
    res.quality = s.last_quality;
    return res;
  }

  const gtsam::Pose3 T_lidar_base = s.T_base_lidar.inverse();
  if (!s.initialized) {
    // Started some scans after the restart (they were not tracked): turned by the gyroscope since.
    if (gyro_led)
      s.T_odom_base = s.T_odom_base * gtsam::Pose3(*frame.gyro_delta, gtsam::Point3(0, 0, 0));
    const gtsam::Pose3 T_ol = s.T_odom_base * s.T_base_lidar;
    Impl::Keyframe kf{T_ol, voxel_downsample(pts, s.cfg.keyframe_cloud_voxel)};
    s.keyframes.push_back(kf);
    s.rebuild_map(T_ol);
    s.initialized = true;
    s.last_keyframe_pose = T_ol;
    s.last_stamp = frame.stamp;
    s.last_kf_cloud = kf.cloud;
    s.last_quality = TrackingQuality();
    res.valid = true;
    res.is_keyframe = true;
    res.T_odom_base = s.T_odom_base;
    res.quality = s.last_quality;
    return res;
  }

  // Predict with the constant-velocity model, then register against the local map. In a confined
  // space the rotation is the gyroscope's and only the translation is registered.
  const auto gyro_pred = [&] {
    return s.T_odom_base * gtsam::Pose3(*frame.gyro_delta, s.velocity.translation()) *
           s.T_base_lidar;
  };
  const gtsam::Pose3 pred_lidar =
      gyro_led ? gyro_pred() : s.T_odom_base * s.velocity * s.T_base_lidar;
  const auto h = to_homogeneous(cloud);
  small_gicp::PointCloud raw(h);
  auto sp = preprocess(raw, s.cfg.scan_voxel * 0.5, s.cfg.num_neighbors, s.cfg.threads);
  auto r = gyro_led ? align_gicp_translation(*s.map_cloud, *sp.first, *s.map_tree,
                                             to_iso(pred_lidar), s.cfg.max_correspondence,
                                             s.cfg.max_iterations, s.cfg.threads)
                    : align_gicp(*s.map_cloud, *sp.first, *s.map_tree, to_iso(pred_lidar),
                                 s.cfg.max_correspondence, s.cfg.max_iterations, s.cfg.threads);
  // Just out of a confined space the local map is still mostly the cabin, and a registration can
  // turn by tens of degrees in one scan at the door. Turned away from the gyroscope by far more
  // than the registration's noise (10 deg and half the turn), the scan takes the gyroscope's
  // rotation.
  bool rotation_from_gyro = gyro_led;
  constexpr Timestamp kAfterConfined = 10'000'000'000;  // ns
  if (!gyro_led && frame.gyro_delta && s.left_confined > 0 &&
      frame.stamp - s.left_confined < kAfterConfined) {
    const gtsam::Rot3 turn = s.T_odom_base.rotation().between(
        (from_iso(r.T_target_source) * s.T_base_lidar.inverse()).rotation());
    const double off = gtsam::Rot3::Logmap(frame.gyro_delta->between(turn)).norm();
    if (off > 10.0 * M_PI / 180.0 + 0.5 * gtsam::Rot3::Logmap(*frame.gyro_delta).norm()) {
      r = align_gicp_translation(*s.map_cloud, *sp.first, *s.map_tree, to_iso(gyro_pred()),
                                 s.cfg.max_correspondence, s.cfg.max_iterations, s.cfg.threads);
      rotation_from_gyro = true;
    }
  }

  TrackingQuality q;
  const double n_src = static_cast<double>(sp.first->size());
  q.inliers = static_cast<int>(r.num_inliers);
  q.fitness = n_src > 0 ? r.num_inliers / n_src : 0.0;
  Eigen::Matrix<double, 6, 6> Hn = r.H / std::max<double>(1.0, r.num_inliers);
  spectrum(Hn, &q.min_eigenvalue, &q.hessian_condition);
  q.state = q.fitness < s.cfg.min_fitness                   ? TrackingState::Lost
            : q.min_eigenvalue < s.cfg.degeneracy_threshold ? TrackingState::Degraded
                                                            : TrackingState::Good;
  s.last_quality = q;
  res.quality = q;
  s.last_stamp = frame.stamp;

  if (q.state == TrackingState::Lost) {
    ++s.lost_count;
    return res;  // no pose: the caller decides (floor manager / session reset)
  }
  s.lost_count = 0;
  gtsam::Pose3 T_ol = from_iso(r.T_target_source);
  if (rotation_from_gyro)  // the rotation was held: keep it a rotation to the last digit
    T_ol = gtsam::Pose3(gtsam::Rot3::ClosestTo(T_ol.rotation().matrix()), T_ol.translation());
  if (q.state == TrackingState::Degraded && s.cfg.pin_degenerate) {
    // Along weak directions registration is noise: keep the prediction in those directions.
    const Eigen::Matrix3d Ht = Hn.block<3, 3>(3, 3);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Ht);
    const Eigen::Vector3d dt_world = T_ol.translation() - pred_lidar.translation();
    // Weak direction in the lidar frame -> world.
    const Eigen::Vector3d weak = pred_lidar.rotation().matrix() * es.eigenvectors().col(0);
    const Eigen::Vector3d fixed = dt_world - weak * weak.dot(dt_world);
    T_ol = gtsam::Pose3(T_ol.rotation(), Eigen::Vector3d(pred_lidar.translation() + fixed));
  }
  if (frame.up_base)
    T_ol = level_lidar(T_ol, s.T_base_lidar, *frame.up_base, s.cfg.max_tilt_error_deg);
  const gtsam::Pose3 new_base = T_ol * T_lidar_base;
  res.delta = s.T_odom_base.between(new_base);
  s.recent.push_back(new_base);
  if (s.recent.size() > 4) s.recent.pop_front();
  // Velocity over up to 3 scans: registration jitter is not extrapolated into the next prediction.
  const size_t n_base = s.recent.size() - 1;
  s.velocity =
      n_base == 0
          ? res.delta
          : gtsam::Pose3::Expmap(gtsam::Pose3::Logmap(s.recent.front().between(s.recent.back())) /
                                 static_cast<double>(n_base));
  s.T_odom_base = new_base;
  res.T_odom_base = new_base;
  res.valid = true;

  const gtsam::Pose3 rel = s.last_keyframe_pose.between(T_ol);
  const double rot = gtsam::Rot3::Logmap(rel.rotation()).norm() * 180.0 / M_PI;
  if (rel.translation().norm() > s.cfg.keyframe_translation || rot > s.cfg.keyframe_rotation_deg ||
      s.force_kf) {
    s.force_kf = false;
    Impl::Keyframe kf{T_ol, voxel_downsample(pts, s.cfg.keyframe_cloud_voxel)};
    s.keyframes.push_back(kf);
    while (static_cast<int>(s.keyframes.size()) > s.cfg.map_max_keyframes) s.keyframes.pop_front();
    s.rebuild_map(T_ol);
    s.last_keyframe_pose = T_ol;
    s.last_kf_cloud = kf.cloud;
    res.is_keyframe = true;
  }
  return res;
}

}  // namespace strataslam
