#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

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
  Eigen::Vector3d t = Eigen::Vector3d::Zero();                    ///< in the floor frame
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();          ///< (x, y, z, w) on disk

  bool operator==(const KeyframeRecord& o) const {
    return id == o.id && stamp_ns == o.stamp_ns && session == o.session && t == o.t &&
           q.coeffs() == o.q.coeffs();
  }
};

struct EdgeRecord {
  std::string type;                  ///< e.g. "odometry", "loop", "baro"
  KeyframeId from = 0, to = 0;
  std::vector<double> measurement;
  uint32_t info_dim = 0;
  std::vector<double> information;   ///< upper triangle, row-major, info_dim*(info_dim+1)/2

  bool operator==(const EdgeRecord& o) const {
    return type == o.type && from == o.from && to == o.to && measurement == o.measurement &&
           info_dim == o.info_dim && information == o.information;
  }
};

struct FloorMetadata {
  FloorId id = 0;
  Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -1);        ///< unit vector, floor frame
  Eigen::Vector3d ground_normal = Eigen::Vector3d(0, 0, 1);
  std::optional<double> reference_pressure_pa;
  std::vector<std::string> sensors;
  std::vector<SessionId> sessions;
  double alignment_residual_rad = 0.0;  ///< gravity alignment residual at submap close
  std::optional<double> altitude_m;     ///< estimated height of the floor frame above floor 0

  bool operator==(const FloorMetadata& o) const {
    return id == o.id && gravity == o.gravity && ground_normal == o.ground_normal &&
           reference_pressure_pa == o.reference_pressure_pa && sensors == o.sensors &&
           sessions == o.sessions && alignment_residual_rad == o.alignment_residual_rad && altitude_m == o.altitude_m;
  }
};

struct FloorData {
  FloorMetadata metadata;
  std::vector<KeyframeRecord> keyframes;
  std::vector<EdgeRecord> edges;
  std::map<KeyframeId, PointCloud> clouds;
  std::map<KeyframeId, VisualPayload> visual;
  std::map<KeyframeId, LidarPayload> lidar;
  std::optional<PointCloud> floor_cloud;

  bool operator==(const FloorData& o) const {
    return metadata == o.metadata && keyframes == o.keyframes && edges == o.edges &&
           clouds == o.clouds && visual == o.visual && lidar == o.lidar &&
           floor_cloud == o.floor_cloud;
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

/// Writes `map` into `dir` (created if needed). Floor ids must be >= 0 and unique.
Result<void> write_map(const std::string& dir, const MapData& map);
/// Reads a map directory; unknown format versions are rejected with a clear error.
Result<MapData> read_map(const std::string& dir);

}  // namespace strataslam
