#include "strataslam/mapping/colorize.hpp"

#include <cmath>

#include "strataslam/mapping/voxel_grid.hpp"

namespace strataslam {

std::vector<std::array<uint8_t, 3>> colorize_points(
    const std::vector<Eigen::Vector3f>& pts, const gtsam::Pose3& T_cam_lidar,
    const CameraIntrinsics& K, const RgbImage& rgb, const DepthImage* depth,
    double occlusion_tolerance, double max_depth, std::array<uint8_t, 3> unknown,
    std::vector<bool>* colored, bool require_surface) {
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
      // On the measured surface only: not where the image saw nothing, nor in front of what it saw.
      if (require_surface && (!(d > 0) || std::abs(pc.z() - d) > occlusion_tolerance)) continue;
    } else if (require_surface) {
      continue;
    }
    const size_t idx = (static_cast<size_t>(v) * rgb.width + u) * 3;
    out[i] = {rgb.data[idx], rgb.data[idx + 1], rgb.data[idx + 2]};
    if (colored) (*colored)[i] = true;
  }
  return out;
}

void colorize_from_views(const std::vector<Eigen::Vector3f>& points,
                         const std::vector<ColorView>& views, double max_depth,
                         double surface_tolerance, std::vector<std::array<uint8_t, 3>>* rgb,
                         std::vector<bool>* seen) {
  if (rgb->size() != points.size() || seen->size() != points.size()) return;
  for (const ColorView& v : views) {
    std::vector<size_t> todo;
    for (size_t i = 0; i < points.size(); ++i)
      if (!(*seen)[i]) todo.push_back(i);
    if (todo.empty()) return;
    if (!v.frame) continue;
    std::vector<Eigen::Vector3f> sub(todo.size());
    for (size_t k = 0; k < todo.size(); ++k) sub[k] = points[todo[k]];
    std::vector<bool> got;
    const auto c = colorize_points(sub, v.T_cam_points, v.frame->K, v.frame->rgb, &v.frame->depth,
                                   surface_tolerance, max_depth, {128, 128, 128}, &got,
                                   /*require_surface=*/true);
    for (size_t k = 0; k < todo.size(); ++k) {
      if (!got[k]) continue;
      (*rgb)[todo[k]] = c[k];
      (*seen)[todo[k]] = true;
    }
  }
}

RgbdFrame decimate_frame(const RgbdFrame& f, int n) {
  if (n <= 1) return f;
  RgbdFrame out;
  out.stamp = f.stamp;
  out.K = f.K;
  out.K.width = f.K.width / n;
  out.K.height = f.K.height / n;
  // Pixel (u', v') is pixel (n u', n v') of the original: u = n u' maps fx u/z + cx to u'.
  out.K.fx = f.K.fx / n;
  out.K.fy = f.K.fy / n;
  out.K.cx = f.K.cx / n;
  out.K.cy = f.K.cy / n;
  if (f.rgb.width == f.K.width && f.rgb.height == f.K.height && !f.rgb.empty()) {
    out.rgb.width = out.K.width;
    out.rgb.height = out.K.height;
    out.rgb.data.resize(static_cast<size_t>(out.rgb.width) * out.rgb.height * 3);
    for (int v = 0; v < out.rgb.height; ++v)
      for (int u = 0; u < out.rgb.width; ++u)
        for (int c = 0; c < 3; ++c)
          out.rgb.data[(static_cast<size_t>(v) * out.rgb.width + u) * 3 + c] =
              f.rgb.data[(static_cast<size_t>(v * n) * f.rgb.width + u * n) * 3 + c];
  }
  if (f.depth.width == f.K.width && f.depth.height == f.K.height && !f.depth.meters.empty()) {
    out.depth.width = out.K.width;
    out.depth.height = out.K.height;
    out.depth.meters.resize(static_cast<size_t>(out.depth.width) * out.depth.height);
    for (int v = 0; v < out.depth.height; ++v)
      for (int u = 0; u < out.depth.width; ++u)
        out.depth.meters[static_cast<size_t>(v) * out.depth.width + u] = f.depth.at(u * n, v * n);
  }
  return out;
}

PointCloud depth_map_cloud(const RgbdFrame& f, int stride, const BackprojectionConfig& cfg,
                           const gtsam::Pose3& T_out_camera) {
  PointCloud out;
  const auto cam = depth_to_cloud(f.depth, f.K, stride, cfg);
  if (!f.rgb.empty() && f.rgb.width == f.K.width && f.rgb.height == f.K.height)
    out.colors = colorize_points(cam, gtsam::Pose3(), f.K, f.rgb);  // each point's own pixel
  out.points = transform_points(cam, T_out_camera.matrix());
  return out;
}

}  // namespace strataslam
