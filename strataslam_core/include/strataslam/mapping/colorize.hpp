#pragma once

#include <gtsam/geometry/Pose3.h>

#include <array>
#include <vector>

#include "strataslam/frontend/depth_utils.hpp"
#include "strataslam/io/point_cloud.hpp"
#include "strataslam/sensors/measurements.hpp"

namespace strataslam {

/// Colors LiDAR points with a camera image. `T_cam_lidar` maps lidar-frame points to the optical
/// camera frame. Points behind the camera, outside the image or farther than `max_depth` keep
/// `unknown`. When a depth image is given, points that lie clearly behind the measured surface
/// (occluded from the camera) are skipped as well.
std::vector<std::array<uint8_t, 3>> colorize_points(
    const std::vector<Eigen::Vector3f>& points_lidar, const gtsam::Pose3& T_cam_lidar,
    const CameraIntrinsics& K, const RgbImage& rgb, const DepthImage* depth = nullptr,
    double occlusion_tolerance = 0.3, double max_depth = 20.0,
    std::array<uint8_t, 3> unknown = {128, 128, 128}, std::vector<bool>* colored = nullptr,
    bool require_surface = false);

/// One more image to colour points with: the camera frame and where the points' frame is seen from
/// (`T_cam_points`: the points' frame -> the optical camera frame).
struct ColorView {
  gtsam::Pose3 T_cam_points;
  const RgbdFrame* frame = nullptr;
};

/// The points not yet `seen` take their colour from the first view, in the given order, that sees
/// them on the surface it measured: a valid depth at their pixel within `surface_tolerance` of
/// their own, and no farther than `max_depth` (where the depth image still tells an occlusion).
/// `rgb` and `seen` are updated in place (ADR-0069: a keyframe's points behind or beside its own
/// camera, coloured by the images of the keyframes around it).
void colorize_from_views(const std::vector<Eigen::Vector3f>& points,
                         const std::vector<ColorView>& views, double max_depth,
                         double surface_tolerance, std::vector<std::array<uint8_t, 3>>* rgb,
                         std::vector<bool>* seen);

/// The frame kept 1/n in each direction (every n-th pixel of the colour and the depth image, the
/// intrinsics to match), for colouring at a quarter of the memory with n = 2 (ADR-0069). No
/// keypoints: it is for colours. n <= 1 gives the frame as it is.
RgbdFrame decimate_frame(const RgbdFrame& f, int n);

/// The depth image of an RGB-D frame as a cloud for the map (ADR-0068): every `stride` pixels with
/// a valid depth (`cfg`), moved by `T_out_camera` (optical camera frame -> the frame wanted), each
/// point with the colour of its pixel when the frame has a colour image of its size (no colours
/// otherwise).
PointCloud depth_map_cloud(const RgbdFrame& f, int stride, const BackprojectionConfig& cfg,
                           const gtsam::Pose3& T_out_camera);

}  // namespace strataslam
