#include "strataslam/frontend/depth_utils.hpp"

#include <cmath>

namespace strataslam {

std::optional<double> sample_depth(const DepthImage& d, double u, double v,
                                   const BackprojectionConfig& cfg) {
  const int iu = static_cast<int>(std::lround(u)), iv = static_cast<int>(std::lround(v));
  if (iu < 0 || iv < 0 || iu >= d.width || iv >= d.height) return std::nullopt;
  const double z = d.at(iu, iv);
  if (!(z > cfg.depth_min && z < cfg.depth_max)) return std::nullopt;
  for (int dv = -1; dv <= 1; ++dv) {
    for (int du = -1; du <= 1; ++du) {
      const int nu = iu + du, nv = iv + dv;
      if ((du == 0 && dv == 0) || nu < 0 || nv < 0 || nu >= d.width || nv >= d.height) continue;
      const double zn = d.at(nu, nv);
      if (zn <= 0) continue;
      if (std::abs(zn - z) > cfg.discontinuity_ratio * z) return std::nullopt;
    }
  }
  return z;
}

std::optional<Eigen::Vector3d> backproject(const DepthImage& d, const CameraIntrinsics& K, double u,
                                           double v, const BackprojectionConfig& cfg) {
  const auto z = sample_depth(d, u, v, cfg);
  if (!z) return std::nullopt;
  return Eigen::Vector3d((u - K.cx) / K.fx * *z, (v - K.cy) / K.fy * *z, *z);
}

std::vector<Eigen::Vector3f> depth_to_cloud(const DepthImage& d, const CameraIntrinsics& K,
                                            int stride, const BackprojectionConfig& cfg) {
  std::vector<Eigen::Vector3f> out;
  for (int v = stride / 2; v < d.height; v += stride) {
    for (int u = stride / 2; u < d.width; u += stride) {
      const float z = d.at(u, v);
      if (!(z > cfg.depth_min && z < cfg.depth_max)) continue;
      out.emplace_back(static_cast<float>((u - K.cx) / K.fx * z),
                       static_cast<float>((v - K.cy) / K.fy * z), z);
    }
  }
  return out;
}

}  // namespace strataslam
