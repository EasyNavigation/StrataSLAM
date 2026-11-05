#pragma once

#include <gtsam/geometry/Rot3.h>

#include <Eigen/Core>
#include <optional>
#include <vector>

#include "strataslam/sensors/measurements.hpp"

namespace strataslam {

struct GravityFromImu {
  Eigen::Vector3d up_body;  ///< unit vector pointing up, in the IMU/base frame
  double accel_norm = 0;    ///< m/s^2, mean specific force magnitude
};

/// Gravity direction from a static window of IMU samples (specific force points up at rest).
/// Returns nullopt unless the window is static: gyro below `max_gyro` and |a| within `accel_tol` of
/// g.
std::optional<GravityFromImu> estimate_gravity_from_imu(const std::vector<ImuSample>& samples,
                                                        double max_gyro = 0.05,
                                                        double accel_tol = 0.5);

/// Roll/pitch alignment: the rotation (zero yaw) that takes `up_body` to world +z.
gtsam::Rot3 rotation_from_up(const Eigen::Vector3d& up_body);

/// `R_world_body` with its roll and pitch replaced so that `up_body` points up, its heading kept.
gtsam::Rot3 level_to(const gtsam::Rot3& R_world_body, const Eigen::Vector3d& up_body);

/// Without an IMU: the dominant horizontal plane below the sensor (at least 0.3 m under it), found
/// by seeded RANSAC. The normal closest to `expected_up` (within `max_tilt_deg`, either sign) is
/// returned, oriented like `expected_up`. All points are in the sensor frame.
std::optional<Eigen::Vector3d> estimate_up_from_ground_plane(
    const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3d& expected_up, uint64_t seed,
    double max_tilt_deg = 25.0, double inlier_threshold = 0.04, int iterations = 300);

}  // namespace strataslam
