#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace strataslam {

enum class Platform { Handheld, Ground, Aerial };
enum class FeatureType { Orb, Xfeat };
enum class DeskewMode { Auto, True, False };
/// `3d`: a 3D LiDAR's point cloud. `2d`: a planar laser scanner's sweep (sensor_msgs/LaserScan),
/// registered in its own plane: x, y and heading only (ADR-0066).
enum class LidarType { ThreeD, Planar };
enum class OdometryFrom { Auto, Camera, Lidar, External };
enum class LoopClosureFrom { Auto, Camera, Lidar, Both };

struct RobotConfig {
  Platform platform = Platform::Handheld;
  std::string base_frame = "base_link";
  std::string odom_frame = "odom";
  std::string map_frame = "map";
};

struct CameraConfig {
  bool enabled = false;
  std::string rgb_topic;
  std::string depth_topic;
  std::string info_topic;
  double depth_min = 0.3;  // m
  double depth_max = 6.0;  // m
  FeatureType features = FeatureType::Orb;
  int max_features = 1000;
};

struct LidarConfig {
  bool enabled = false;
  std::string topic = "/points";
  LidarType type = LidarType::ThreeD;
  double min_range = 0.5;   // m
  double max_range = 50.0;  // m
  DeskewMode deskew = DeskewMode::Auto;
};

/// Continuous-time IMU noise densities (SI units).
struct ImuNoise {
  double accel_noise_density = 2.0e-3;  // m/s^2/sqrt(Hz)
  double gyro_noise_density = 1.7e-4;   // rad/s/sqrt(Hz)
  double accel_random_walk = 3.0e-3;    // m/s^3/sqrt(Hz)
  double gyro_random_walk = 2.0e-5;     // rad/s^2/sqrt(Hz)
};

struct ImuConfig {
  bool enabled = false;
  std::string topic = "/imu";
  /// Its vertical acceleration detects floor changes the odometry cannot see (elevators) and rules
  /// out that a tracking loss hid one. Off: the IMU only gives gravity (the attitude), which is
  /// what a ground robot's vibrating base allows (ADR-0032).
  bool floor_changes = true;
  ImuNoise noise;
};

struct BarometerConfig {
  bool enabled = false;
  std::string topic = "/pressure";
  std::string
      reference_topic;  ///< a fixed base station (sensor_msgs/FluidPressure): differential altitude
  double temperature_k = 288.15;
  double settle_time_s = 2.0;  ///< s for the low-pass filter to settle within 1 % after a step
};

/// The robot's own odometry (wheels, typically): the motion prior of the camera tracking, which
/// predicts each frame and carries the pose through tracking losses. Read from TF (`frame` ->
/// robot.base_frame, what most ROS robots publish) or from a nav_msgs/Odometry `topic`.
struct ExternalOdometryConfig {
  bool enabled = false;
  std::string topic;
  std::string frame;
};

struct SensorsConfig {
  CameraConfig camera;
  LidarConfig lidar;
  ImuConfig imu;
  BarometerConfig barometer;
  ExternalOdometryConfig external_odometry;
};

struct FusionConfig {
  OdometryFrom odometry_from = OdometryFrom::Auto;
  LoopClosureFrom loop_closure_from = LoopClosureFrom::Auto;
  bool colorize_cloud = true;
  bool colorize_cloud_explicit =
      false;  ///< the profile set it (the warning without a camera is only for that case)
  /// Colouring a LiDAR's points when the map is saved: the images of up to this many keyframes
  /// around are tried for the points a keyframe's own image did not see (0: its own image only).
  /// Time at save, never the trajectory (ADR-0069).
  int colorize_views = 8;
  /// Scale of the keyframes' images kept for that colouring: 1, 1/2 (a quarter of the memory),
  /// 1/3 or 1/4. A keyframe's own colours always come from its full image (ADR-0069).
  double colorize_image_scale = 1.0;
  /// Visual vocabulary for camera loop closures (`strataslam vocab train`). Empty: none, and no
  /// visual loop closure. A relative path is relative to the profile's folder when the profile is
  /// read from a file. `default`: the vocabulary shipped with strataslam_ros (trained on TUM RGB-D,
  /// indoor offices; ADR-0054).
  std::string vocabulary;
};

/// How the levels of the run are mapped.
enum class LevelMode {
  Floors,  ///< each level is a map of its own (floors), stacked by `strataslam fuse`
  Single,  ///< one map in 3D, heights kept (a level with slopes or ramps)
  Planar,  ///< one map on one plane: the base keeps its height and the floor's attitude (a flat
           ///< floor)
};

struct MultifloorConfig {
  LevelMode mode = LevelMode::Floors;
  bool enabled = true;  ///< mode == Floors (`enabled: false` in a profile is `mode: single`)
  std::optional<double> floor_height_prior = 3.0;  // m, nullopt if unknown
  bool split_on_tracking_loss = true;
  double tracking_loss_timeout = 2.0;  // s
  double min_vertical_change = 1.5;    // m
};

struct OutputConfig {
  std::string map_dir = "~/strataslam_maps/{date}_{time}";
  double voxel_size = 0.05;            // m
  double keyframe_translation = 0.25;  // m
  double keyframe_rotation_deg = 10.0;
  /// The node publishes the TF tree above the robot's odometry: map -> floor_<k> -> odom (REP-105).
  /// False when another node owns map -> odom (a localizer such as AMCL): only its own branch
  /// floor_<k> -> strataslam_odom, beside it.
  bool publish_tf = true;
};

struct RuntimeConfig {
  uint64_t seed = 42;
  int threads = 0;  // 0 = automatic
};

struct Config {
  int version = 1;
  RobotConfig robot;
  SensorsConfig sensors;
  FusionConfig fusion;
  MultifloorConfig multifloor;
  OutputConfig output;
  RuntimeConfig runtime;
};

enum class OdometrySource { Camera, Lidar, External };
enum class LoopSource { Camera, Lidar, Both };

/// Result of resolving the `auto` choices (docs: profile section 8.3).
struct ResolvedModes {
  OdometrySource odometry;
  LoopSource loop_closure;
  bool verify_loop_with_lidar;  ///< camera loop candidates verified with LiDAR registration
};

/// Resolve `auto` modes for a validated config.
ResolvedModes resolve_modes(const Config& config);

}  // namespace strataslam
