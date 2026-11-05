#pragma once

#include <Eigen/Core>
#include <vector>

#include "strataslam/common/types.hpp"
#include "strataslam/io/payloads.hpp"

namespace strataslam {

/// Scan Context descriptor (G. Kim and A. Kim, "Scan Context: Egocentric Spatial Descriptor for
/// Place Recognition within 3D Point Cloud Map", IROS 2018). Implemented from the paper.
struct ScanContextConfig {
  int rings = 20;
  int sectors = 60;
  double max_range = 40.0;          ///< m, radial extent of the ring bins
  double lidar_height = 2.0;        ///< m, added to z so that cell values are positive
  int num_candidates = 10;          ///< ring-key search results re-ranked with the full distance
  double distance_threshold = 0.2;  ///< accept candidates below this cosine distance
  int exclude_recent = 20;          ///< skip the most recent database entries
};

struct ScanContext {
  int rings = 0, sectors = 0;
  Eigen::MatrixXf matrix;    ///< rings x sectors, max height per polar bin (0 = empty)
  Eigen::VectorXf ring_key;  ///< occupancy ratio per ring (rotation invariant)
};

ScanContext make_scan_context(const std::vector<Eigen::Vector3f>& points,
                              const ScanContextConfig& config);

/// Cosine distance with column-shift alignment. Returns the minimum distance and the best shift
/// (columns the query must be rotated to match the candidate); the yaw is shift * 2*pi/sectors.
struct ScanContextMatch {
  double distance = 1.0;
  int shift = 0;
  double yaw = 0.0;  ///< rad, rotation of the query relative to the candidate
};
ScanContextMatch compare_scan_contexts(const ScanContext& query, const ScanContext& candidate);

/// Database restricted to one floor: the caller owns one instance per floor (gating layer 1).
class ScanContextDatabase {
 public:
  explicit ScanContextDatabase(const ScanContextConfig& config) : cfg_(config) {}

  void add(KeyframeId id, ScanContext sc);
  size_t size() const { return entries_.size(); }

  struct Candidate {
    KeyframeId id;
    ScanContextMatch match;
  };
  /// Best candidates (distance below threshold), ranked by distance. The `exclude_recent`
  /// most recent entries are skipped.
  std::vector<Candidate> query(const ScanContext& sc) const;

  static LidarPayload to_payload(const ScanContext& sc, uint32_t num_points, float max_range);
  static ScanContext from_payload(const LidarPayload& p);

 private:
  ScanContextConfig cfg_;
  struct Entry {
    KeyframeId id;
    ScanContext sc;
  };
  std::vector<Entry> entries_;
};

}  // namespace strataslam
