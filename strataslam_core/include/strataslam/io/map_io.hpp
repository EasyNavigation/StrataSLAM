#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "strataslam/common/result.hpp"
#include "strataslam/common/types.hpp"
#include "strataslam/floors/transition.hpp"
#include "strataslam/io/payloads.hpp"
#include "strataslam/io/point_cloud.hpp"

namespace strataslam {

struct KeyframeRecord {
  KeyframeId id = 0;
  Timestamp stamp_ns = 0;
  SessionId session = 0;
  Eigen::Vector3d t = Eigen::Vector3d::Zero();            ///< in the floor frame
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();  ///< (x, y, z, w) on disk

  bool operator==(const KeyframeRecord& o) const {
    return id == o.id && stamp_ns == o.stamp_ns && session == o.session && t == o.t &&
           q.coeffs() == o.q.coeffs();
  }
};

struct EdgeRecord {
  std::string type;  ///< e.g. "odometry", "loop", "baro"
  KeyframeId from = 0, to = 0;
  std::vector<double> measurement;
  uint32_t info_dim = 0;
  std::vector<double> information;  ///< upper triangle, row-major, info_dim*(info_dim+1)/2

  bool operator==(const EdgeRecord& o) const {
    return type == o.type && from == o.from && to == o.to && measurement == o.measurement &&
           info_dim == o.info_dim && information == o.information;
  }
};

struct FloorMetadata {
  FloorId id = 0;
  Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -1);  ///< unit vector, floor frame
  Eigen::Vector3d ground_normal = Eigen::Vector3d(0, 0, 1);
  std::optional<double> reference_pressure_pa;
  std::vector<std::string> sensors;
  std::vector<SessionId> sessions;
  double alignment_residual_rad = 0.0;  ///< gravity alignment residual at submap close
  std::optional<double> altitude_m;     ///< estimated height of the floor frame above floor 0
  /// Frame of the per-keyframe clouds (`clouds/<id>.pcd`): "base", the keyframe's base frame,
  /// placed in the floor by the keyframe's pose alone. Maps written before the field have
  /// "sensor": the frame of the sensor that took them, whose mounting the map does not record.
  std::string cloud_frame = "base";
  /// Where the odometry left the floor, as the live view draws it: T_map_floor as (yaw rad, x, y,
  /// z m) in the frame of the lowest floor of the map, gravity-aligned. `strataslam fuse` refines
  /// it (fused/floor_transforms.yaml). Absent in maps written before the field (ADR-0064).
  std::optional<Eigen::Vector4d> placement;
  /// Where the sensor of `clouds/` and that of `depth_clouds/` sit in the base frame (m): the
  /// origin of their rays, for the free space of an occupancy grid (ADR-0068). Absent in maps
  /// written before the fields, and `depth_origin` when the floor has no depth clouds.
  std::optional<Eigen::Vector3d> cloud_origin, depth_origin;

  bool operator==(const FloorMetadata& o) const {
    return id == o.id && gravity == o.gravity && ground_normal == o.ground_normal &&
           reference_pressure_pa == o.reference_pressure_pa && sensors == o.sensors &&
           sessions == o.sessions && alignment_residual_rad == o.alignment_residual_rad &&
           altitude_m == o.altitude_m && cloud_frame == o.cloud_frame && placement == o.placement &&
           cloud_origin == o.cloud_origin && depth_origin == o.depth_origin;
  }
};

struct FloorData {
  FloorMetadata metadata;
  std::vector<KeyframeRecord> keyframes;
  std::vector<EdgeRecord> edges;
  std::map<KeyframeId, PointCloud> clouds;
  /// A planar scanner's keyframes with a camera: the depth image's cloud in the keyframe's base
  /// frame, with its colours (ADR-0068). Empty otherwise; written only when present.
  std::map<KeyframeId, PointCloud> depth_clouds;
  std::map<KeyframeId, VisualPayload> visual;
  std::map<KeyframeId, LidarPayload> lidar;
  std::optional<PointCloud> floor_cloud;

  bool operator==(const FloorData& o) const {
    return metadata == o.metadata && keyframes == o.keyframes && edges == o.edges &&
           clouds == o.clouds && depth_clouds == o.depth_clouds && visual == o.visual &&
           lidar == o.lidar && floor_cloud == o.floor_cloud;
  }
};

struct MapData {
  std::string strataslam_version;
  uint32_t format_version = kMapFormatVersion;
  std::string profile_yaml;  ///< verbatim copy of the profile used for capture
  std::vector<TransitionRecord> transitions;
  std::vector<FloorData> floors;

  bool operator==(const MapData& o) const {
    return strataslam_version == o.strataslam_version && format_version == o.format_version &&
           profile_yaml == o.profile_yaml && transitions == o.transitions && floors == o.floors;
  }
};

/// Writes `map` into `dir` (created if needed). Floor ids must be >= 0 and unique. When every
/// floor with a cloud has a `placement`, also `global_cloud.pcd`: the floor clouds placed by it,
/// the building as the live view shows it at the end of the run.
Result<void> write_map(const std::string& dir, const MapData& map);
/// The floor clouds of `map` in the frame of its lowest floor, each placed by its `placement`;
/// nullopt when a floor with a cloud has none (a map written before the field).
std::optional<PointCloud> placed_global_cloud(const MapData& map);
/// Reads a map directory; unknown format versions are rejected with a clear error.
Result<MapData> read_map(const std::string& dir);

}  // namespace strataslam
