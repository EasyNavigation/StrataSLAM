#include "strataslam/mapping/colorize.hpp"

#include <cmath>

namespace strataslam {

std::vector<std::array<uint8_t, 3>> colorize_points(const std::vector<Eigen::Vector3f>& pts,
                                                    const gtsam::Pose3& T_cam_lidar,
                                                    const CameraIntrinsics& K, const RgbImage& rgb,
                                                    const DepthImage* depth,
                                                    double occlusion_tolerance, double max_depth,
                                                    std::array<uint8_t, 3> unknown,
                                                    std::vector<bool>* colored) {
  std::vector<std::array<uint8_t, 3>> out(pts.size(), unknown);
  if (colored) colored->assign(pts.size(), false);
  if (rgb.empty() || rgb.width != K.width || rgb.height != K.height) return out;
  for (size_t i = 0; i < pts.size(); ++i) {
    const Eigen::Vector3d pc = T_cam_lidar.transformFrom(Eigen::Vector3d(pts[i].cast<double>()));
    if (pc.z() < 0.1 || pc.z() > max_depth) continue;
    const int u = static_cast<int>(std::lround(K.fx * pc.x() / pc.z() + K.cx)),
              v = static_cast<int>(std::lround(K.fy * pc.y() / pc.z() + K.cy));
    if (u < 0 || v < 0 || u >= rgb.width || v >= rgb.height) continue;
    if (depth && depth->width == rgb.width && depth->height == rgb.height) {
      const float d = depth->at(u, v);
      if (d > 0 && pc.z() > d + occlusion_tolerance) continue;  // hidden behind a nearer surface
    }
    const size_t idx = (static_cast<size_t>(v) * rgb.width + u) * 3;
    out[i] = {rgb.data[idx], rgb.data[idx + 1], rgb.data[idx + 2]};
    if (colored) (*colored)[i] = true;
  }
  return out;
}

}  // namespace strataslam
