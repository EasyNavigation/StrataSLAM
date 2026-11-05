#include "strataslam/frontend/visual_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace strataslam {

int hamming_distance(const Descriptor& a, const Descriptor& b) {
  uint64_t x[4], y[4];
  std::memcpy(x, a.data(), 32);
  std::memcpy(y, b.data(), 32);
  int d = 0;
  for (int i = 0; i < 4; ++i) d += __builtin_popcountll(x[i] ^ y[i]);
  return d;
}

VisualObservation make_observation(const RgbdFrame& frame, const BackprojectionConfig& cfg) {
  VisualObservation obs;
  obs.keypoints = frame.keypoints;
  obs.p_cam.resize(frame.keypoints.size());
  for (size_t i = 0; i < frame.keypoints.size(); ++i) {
    obs.p_cam[i] = backproject(frame.depth, frame.K, frame.keypoints[i].u, frame.keypoints[i].v, cfg);
  }
  return obs;
}

namespace {
cv::Mat camera_matrix(const CameraIntrinsics& K) {
  return (cv::Mat_<double>(3, 3) << K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1);
}

void pose_to_cv(const gtsam::Pose3& T, cv::Mat* rvec, cv::Mat* tvec) {
  const Eigen::Vector3d w = gtsam::Rot3::Logmap(T.rotation());
  *rvec = (cv::Mat_<double>(3, 1) << w.x(), w.y(), w.z());
  *tvec = (cv::Mat_<double>(3, 1) << T.x(), T.y(), T.z());
}

gtsam::Pose3 pose_from_cv(const cv::Mat& rvec, const cv::Mat& tvec) {
  const Eigen::Vector3d w(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
  return gtsam::Pose3(gtsam::Rot3::Expmap(w), Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)));
}
}  // namespace

namespace {
/// One RANSAC + LM attempt. Solutions that put the inliers behind the camera (mirror solutions
/// that OpenCV's reprojection test cannot tell apart) are rejected by the final re-evaluation.
PnPResult pnp_attempt(const std::vector<PnPCorrespondence>& corr, const std::vector<cv::Point3d>& obj,
                      const std::vector<cv::Point2d>& img, const CameraIntrinsics& K, const gtsam::Pose3* init, int flags,
                      double reproj_px, int iterations) {
  PnPResult res;
  const cv::Mat Km = camera_matrix(K);
  cv::Mat rvec, tvec;
  const bool guess = init != nullptr;
  if (guess) pose_to_cv(*init, &rvec, &tvec);
  std::vector<int> inl;
  bool ok = false;
  try {
    ok = cv::solvePnPRansac(obj, img, Km, cv::noArray(), rvec, tvec, guess, iterations, static_cast<float>(reproj_px), 0.999, inl, flags);
  } catch (const cv::Exception&) {
    ok = false;
  }
  if (!ok || inl.size() < 6) return res;
  std::vector<cv::Point3d> o2;
  std::vector<cv::Point2d> i2;
  for (int i : inl) { o2.push_back(obj[i]); i2.push_back(img[i]); }
  const cv::Mat rvec_ransac = rvec.clone(), tvec_ransac = tvec.clone();
  try {
    cv::solvePnPRefineLM(o2, i2, Km, cv::noArray(), rvec, tvec);
  } catch (const cv::Exception&) {
    // The refinement failed (degenerate inlier set): keep the RANSAC pose, which it may have modified.
    rvec = rvec_ransac;
    tvec = tvec_ransac;
  }
  res.T_cam_obj = pose_from_cv(rvec, tvec);
  for (size_t i = 0; i < corr.size(); ++i) {
    const Eigen::Vector3d pc = res.T_cam_obj.transformFrom(corr[i].point);
    if (pc.z() <= 0.05) continue;
    const double u = K.fx * pc.x() / pc.z() + K.cx, v = K.fy * pc.y() / pc.z() + K.cy;
    if (std::hypot(u - corr[i].pixel.x(), v - corr[i].pixel.y()) < reproj_px) res.inliers.push_back(static_cast<int>(i));
  }
  res.ok = res.inliers.size() >= 6;
  return res;
}
}  // namespace

std::optional<gtsam::Pose3> verify_loop_pnp(const std::vector<PnPCorrespondence>& corr, const CameraIntrinsics& K,
                                            int min_inliers, double reprojection_px, int iterations) {
  if (static_cast<int>(corr.size()) < min_inliers) return std::nullopt;
  const PnPResult pnp = solve_pnp(corr, K, nullptr, reprojection_px, iterations);
  if (!pnp.ok || static_cast<int>(pnp.inliers.size()) < min_inliers) return std::nullopt;
  return pnp.T_cam_obj;
}

PnPResult solve_pnp(const std::vector<PnPCorrespondence>& corr, const CameraIntrinsics& K,
                    const gtsam::Pose3* init, double reproj_px, int iterations) {
  if (corr.size() < 6) return PnPResult();
  std::vector<cv::Point3d> obj(corr.size());
  std::vector<cv::Point2d> img(corr.size());
  for (size_t i = 0; i < corr.size(); ++i) {
    obj[i] = {corr[i].point.x(), corr[i].point.y(), corr[i].point.z()};
    img[i] = {corr[i].pixel.x(), corr[i].pixel.y()};
  }
  PnPResult best;
  const auto consider = [&](const PnPResult& r) {
    if (r.ok && r.inliers.size() > best.inliers.size()) best = r;
  };
  if (init) consider(pnp_attempt(corr, obj, img, K, init, cv::SOLVEPNP_ITERATIVE, reproj_px, iterations));
  // Accept the guided result when it explains most of the data; otherwise try other strategies.
  if (best.ok && best.inliers.size() >= corr.size() / 2) return best;
  consider(pnp_attempt(corr, obj, img, K, nullptr, cv::SOLVEPNP_ITERATIVE, reproj_px, iterations));
  if (!best.ok) consider(pnp_attempt(corr, obj, img, K, nullptr, cv::SOLVEPNP_SQPNP, reproj_px, iterations));
  return best;
}

gtsam::Pose3 refine_rgbd(const std::vector<RgbdMatch>& matches, const CameraIntrinsics& K, const gtsam::Pose3& init,
                         double pixel_sigma, double depth_sigma_coeff, double depth_sigma_floor, int iterations) {
  gtsam::Pose3 T = init;
  for (int it = 0; it < iterations; ++it) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
    for (const auto& m : matches) {
      Eigen::Matrix<double, 3, 6> Hp;
      const Eigen::Vector3d pc = T.transformFrom(m.point, Hp);
      if (pc.z() < 0.05) continue;
      // Reprojection residual.
      const double iz = 1.0 / pc.z();
      Eigen::Matrix<double, 2, 3> Jp;
      Jp << K.fx * iz, 0, -K.fx * pc.x() * iz * iz, 0, K.fy * iz, -K.fy * pc.y() * iz * iz;
      const Eigen::Vector2d r(K.fx * pc.x() * iz + K.cx - m.pixel.x(), K.fy * pc.y() * iz + K.cy - m.pixel.y());
      const Eigen::Matrix<double, 2, 6> J = Jp * Hp / pixel_sigma;
      H += J.transpose() * J;
      b -= J.transpose() * (r / pixel_sigma);
      if (m.depth) {
        const double sz = depth_sigma_coeff * (*m.depth) * (*m.depth) + depth_sigma_floor;
        const Eigen::Matrix<double, 1, 6> Jd = Hp.row(2) / sz;
        H += Jd.transpose() * Jd;
        b -= Jd.transpose() * ((pc.z() - *m.depth) / sz);
      }
    }
    const Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);
    if (!dx.allFinite()) break;
    T = T.retract(dx);
    if (dx.norm() < 1e-7) break;
  }
  return T;
}

struct VisualTracker::Impl {
  struct Landmark {
    Eigen::Vector3d p;       // odom frame
    Descriptor desc;
    int last_kf;
    int n_obs = 1;  // measurements averaged into `p`
  };
  VisualTrackerConfig cfg;
  std::vector<Landmark> map;
  bool init = false;
  int kf_index = 0;
  gtsam::Pose3 last_kf_pose;
  int last_kf_inliers = 0;
  std::vector<KeypointUse> use;      // per keypoint of the last tracked observation
  std::vector<int> matched_landmark; // landmark per keypoint (-1 if none)
};

VisualTracker::VisualTracker(const VisualTrackerConfig& c) : impl_(std::make_unique<Impl>()) { impl_->cfg = c; }
VisualTracker::~VisualTracker() = default;

void VisualTracker::reset() {
  auto& s = *impl_;
  s.map.clear();
  s.init = false;
  s.kf_index = 0;
  s.last_kf_inliers = 0;
  s.use.clear();
  s.matched_landmark.clear();
}
bool VisualTracker::initialized() const { return impl_->init; }
const std::vector<KeypointUse>& VisualTracker::last_keypoint_use() const { return impl_->use; }
size_t VisualTracker::map_size() const { return impl_->map.size(); }

void VisualTracker::initialize(const CameraIntrinsics&, const VisualObservation& obs, const gtsam::Pose3& T) {
  auto& s = *impl_;
  reset();
  s.init = true;
  s.last_kf_pose = T;
  s.use.assign(obs.keypoints.size(), KeypointUse::Unmatched);
  s.matched_landmark.assign(obs.keypoints.size(), -1);
  insert_keyframe(obs, T);
  s.last_kf_inliers = static_cast<int>(s.map.size());
}

VisualTracker::TrackResult VisualTracker::track(const CameraIntrinsics& K, const VisualObservation& obs,
                                                const gtsam::Pose3& pred, bool widen) {
  auto& s = *impl_;
  TrackResult out;
  out.T_odom_cam = pred;
  const size_t nk = obs.keypoints.size();
  s.use.assign(nk, KeypointUse::Unmatched);
  s.matched_landmark.assign(nk, -1);
  if (!s.init || nk < 10 || s.map.empty()) return out;

  // Spatial grid over keypoints for windowed search.
  const double cell = std::max(8.0, widen ? s.cfg.wide_radius_px : s.cfg.match_radius_px);
  std::unordered_map<int64_t, std::vector<int>> grid;
  // Packed through uint64_t: shifting a negative int64_t left is undefined (UBSan aborted the scenarios on keypoints
  // left of or above the image, cx or cy = -1; audit AUD-047). The bits are the same.
  const auto cell_key = [&](int cx, int cy) {
    return static_cast<int64_t>((static_cast<uint64_t>(static_cast<uint32_t>(cx)) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(cy)));
  };
  for (size_t i = 0; i < nk; ++i) {
    grid[cell_key(static_cast<int>(obs.keypoints[i].u / cell), static_cast<int>(obs.keypoints[i].v / cell))].push_back(static_cast<int>(i));
  }
  const auto guided = [&](const gtsam::Pose3& T_odom_cam, double radius, std::vector<PnPCorrespondence>* corr,
                          std::vector<std::pair<int, int>>* pairs) {
    const gtsam::Pose3 T_cam_odom = T_odom_cam.inverse();
    std::vector<int> best_for_kp(nk, -1);
    std::vector<int> best_dist_for_kp(nk, 1 << 20);
    std::vector<int> best_lm_for_kp(nk, -1);
    for (size_t li = 0; li < s.map.size(); ++li) {
      const Eigen::Vector3d pc = T_cam_odom.transformFrom(s.map[li].p);
      if (pc.z() < 0.2 || pc.z() > s.cfg.depth.depth_max * 1.5) continue;
      const double u = K.fx * pc.x() / pc.z() + K.cx, v = K.fy * pc.y() / pc.z() + K.cy;
      if (u < -radius || v < -radius || u > K.width + radius || v > K.height + radius) continue;
      int best = -1, best_d = 1 << 20, second_d = 1 << 20;
      const int cx0 = static_cast<int>((u - radius) / cell), cx1 = static_cast<int>((u + radius) / cell);
      const int cy0 = static_cast<int>((v - radius) / cell), cy1 = static_cast<int>((v + radius) / cell);
      for (int cy = cy0; cy <= cy1; ++cy) {
        for (int cx = cx0; cx <= cx1; ++cx) {
          const auto it = grid.find(cell_key(cx, cy));
          if (it == grid.end()) continue;
          for (int ki : it->second) {
            if (std::hypot(obs.keypoints[ki].u - u, obs.keypoints[ki].v - v) > radius) continue;
            const int d = hamming_distance(s.map[li].desc, obs.keypoints[ki].descriptor);
            if (d < best_d) { second_d = best_d; best_d = d; best = ki; }
            else if (d < second_d) second_d = d;
          }
        }
      }
      if (best < 0 || best_d > s.cfg.max_hamming) continue;
      if (second_d < (1 << 20) && best_d > s.cfg.ratio * second_d) continue;
      // Each keypoint keeps its best landmark.
      if (best_d < best_dist_for_kp[best]) {
        best_dist_for_kp[best] = best_d;
        best_lm_for_kp[best] = static_cast<int>(li);
      }
    }
    for (size_t ki = 0; ki < nk; ++ki) {
      if (best_lm_for_kp[ki] < 0) continue;
      corr->push_back({s.map[best_lm_for_kp[ki]].p, Eigen::Vector2d(obs.keypoints[ki].u, obs.keypoints[ki].v)});
      pairs->push_back({static_cast<int>(ki), best_lm_for_kp[ki]});
    }
  };

  std::vector<PnPCorrespondence> corr;
  std::vector<std::pair<int, int>> pairs;
  guided(pred, widen ? s.cfg.wide_radius_px : s.cfg.match_radius_px, &corr, &pairs);
  if (static_cast<int>(corr.size()) < 2 * s.cfg.min_inliers && !widen) {
    corr.clear();
    pairs.clear();
    guided(pred, s.cfg.wide_radius_px, &corr, &pairs);
  }
  if (static_cast<int>(corr.size()) < s.cfg.min_inliers) {
    // Global (windowless) matching against the whole local map as a last resort.
    corr.clear();
    pairs.clear();
    std::vector<int> best_lm(nk, -1), best_d(nk, 1 << 20), second_d(nk, 1 << 20);
    for (size_t li = 0; li < s.map.size(); ++li) {
      for (size_t ki = 0; ki < nk; ++ki) {
        const int d = hamming_distance(s.map[li].desc, obs.keypoints[ki].descriptor);
        if (d < best_d[ki]) { second_d[ki] = best_d[ki]; best_d[ki] = d; best_lm[ki] = static_cast<int>(li); }
        else if (d < second_d[ki]) second_d[ki] = d;
      }
    }
    for (size_t ki = 0; ki < nk; ++ki) {
      if (best_lm[ki] < 0 || best_d[ki] > s.cfg.max_hamming || best_d[ki] > s.cfg.ratio * second_d[ki]) continue;
      corr.push_back({s.map[best_lm[ki]].p, Eigen::Vector2d(obs.keypoints[ki].u, obs.keypoints[ki].v)});
      pairs.push_back({static_cast<int>(ki), best_lm[ki]});
    }
  }
  out.matches = static_cast<int>(corr.size());
  for (const auto& p : pairs) s.use[p.first] = KeypointUse::Rejected;  // until PnP keeps it
  if (out.matches < s.cfg.min_inliers) return out;

  const gtsam::Pose3 init_cam_odom = pred.inverse();
  PnPResult pnp = solve_pnp(corr, K, &init_cam_odom, s.cfg.ransac_reprojection_px, s.cfg.ransac_iterations);
  if (!pnp.ok) pnp = solve_pnp(corr, K, nullptr, s.cfg.ransac_reprojection_px, s.cfg.ransac_iterations);
  if (!pnp.ok) return out;
  out.inliers = static_cast<int>(pnp.inliers.size());
  gtsam::Pose3 T_cam_obj = pnp.T_cam_obj;
  if (s.cfg.use_depth_in_refinement && pnp.inliers.size() >= 12) {
    std::vector<RgbdMatch> rm;
    for (int i : pnp.inliers) {
      const auto& c = corr[i];
      RgbdMatch m;
      m.point = c.point;
      m.pixel = c.pixel;
      const auto& p_cam = obs.p_cam[pairs[i].first];
      m.depth = p_cam ? std::optional<double>(p_cam->z()) : std::nullopt;
      rm.push_back(m);
    }
    T_cam_obj = refine_rgbd(rm, K, T_cam_obj, s.cfg.pixel_sigma, s.cfg.depth_sigma_coeff, s.cfg.depth_sigma_floor, 8);
  }
  out.T_odom_cam = T_cam_obj.inverse();
  out.ok = out.inliers >= s.cfg.min_inliers;
  if (!out.ok) return out;
  for (int i : pnp.inliers) {
    s.use[pairs[i].first] = KeypointUse::Inlier;
    s.matched_landmark[pairs[i].first] = pairs[i].second;
    auto& lm = s.map[pairs[i].second];
    lm.last_kf = s.kf_index;  // keeps recently seen landmarks alive
    // Running average of the landmark position over successive depth measurements (bounded
    // memory), which averages out the depth noise frozen into a single-shot triangulation.
    const auto& p_cam = obs.p_cam[pairs[i].first];
    if (s.cfg.refine_landmarks && p_cam && lm.n_obs < s.cfg.landmark_max_obs) {
      const Eigen::Vector3d meas = out.T_odom_cam.transformFrom(*p_cam);
      lm.p = (lm.p * lm.n_obs + meas) / (lm.n_obs + 1);
      ++lm.n_obs;
    }
  }
  const gtsam::Pose3 rel = s.last_kf_pose.between(out.T_odom_cam);
  const double rot = gtsam::Rot3::Logmap(rel.rotation()).norm() * 180.0 / M_PI;
  out.want_keyframe = rel.translation().norm() > s.cfg.keyframe_translation || rot > s.cfg.keyframe_rotation_deg ||
                      out.inliers < s.cfg.keyframe_tracked_ratio * s.last_kf_inliers;
  return out;
}

void VisualTracker::insert_keyframe(const VisualObservation& obs, const gtsam::Pose3& T_odom_cam) {
  auto& s = *impl_;
  ++s.kf_index;
  int added = 0;
  // Strongest unmatched keypoints with a valid 3D point become landmarks.
  std::vector<size_t> order(obs.keypoints.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return obs.keypoints[a].response > obs.keypoints[b].response; });
  int tracked = 0;
  for (size_t i : order) {
    if (i < s.use.size() && s.use[i] == KeypointUse::Inlier) { ++tracked; continue; }
    const auto& p_cam = obs.p_cam[i];
    if (!p_cam || added >= s.cfg.max_new_landmarks) continue;
    s.map.push_back({T_odom_cam.transformFrom(*p_cam), obs.keypoints[i].descriptor, s.kf_index});
    ++added;
  }
  s.last_kf_pose = T_odom_cam;
  s.last_kf_inliers = std::max(tracked, 1) + 0;
  // Prune landmarks not seen in the last N keyframes.
  const int oldest = s.kf_index - s.cfg.local_map_keyframes;
  s.map.erase(std::remove_if(s.map.begin(), s.map.end(), [&](const Impl::Landmark& l) { return l.last_kf < oldest; }), s.map.end());
}

}  // namespace strataslam
