#pragma once

#include <gtsam/geometry/Pose3.h>

#include <array>
#include <vector>

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
    std::array<uint8_t, 3> unknown = {128, 128, 128}, std::vector<bool>* colored = nullptr);

}  // namespace strataslam
