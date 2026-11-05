#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strataslam/common/result.hpp"

namespace strataslam {

/// Version of every on-disk format written by this library (see docs/data_format.md).
constexpr uint32_t kMapFormatVersion = 1;

struct VisualKeypoint {
  float x = 0, y = 0, response = 0;
  float depth = 0;  ///< m, 0 = unknown
  bool operator==(const VisualKeypoint& o) const {
    return x == o.x && y == o.y && response == o.response && depth == o.depth;
  }
};

struct VisualPayload {
  std::vector<VisualKeypoint> keypoints;
  uint32_t descriptor_cols = 0;        ///< bytes per descriptor
  std::vector<uint8_t> descriptors;    ///< keypoints.size() * descriptor_cols bytes

  bool operator==(const VisualPayload& o) const {
    return keypoints == o.keypoints && descriptor_cols == o.descriptor_cols &&
           descriptors == o.descriptors;
  }
};

struct LidarPayload {
  uint32_t rings = 0, sectors = 0;
  std::vector<float> scan_context;  ///< rings * sectors, row-major
  std::vector<float> ring_key;      ///< rings
  uint32_t num_points = 0;          ///< points in the source scan
  float max_range = 0;              ///< m

  bool operator==(const LidarPayload& o) const {
    return rings == o.rings && sectors == o.sectors && scan_context == o.scan_context &&
           ring_key == o.ring_key && num_points == o.num_points && max_range == o.max_range;
  }
};

// Container: "ESLM" | u32 format version | u32 kind | u64 payload bytes | payload (little endian).
Result<void> write_visual_payload(const std::string& path, const VisualPayload& p);
Result<VisualPayload> read_visual_payload(const std::string& path);
Result<void> write_lidar_payload(const std::string& path, const LidarPayload& p);
Result<LidarPayload> read_lidar_payload(const std::string& path);

}  // namespace strataslam
