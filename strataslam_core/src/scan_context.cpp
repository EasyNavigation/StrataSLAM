#include "strataslam/place_recognition/scan_context.hpp"

#include <algorithm>
#include <cmath>

namespace strataslam {

ScanContext make_scan_context(const std::vector<Eigen::Vector3f>& points,
                              const ScanContextConfig& cfg) {
  ScanContext sc;
  sc.rings = cfg.rings;
  sc.sectors = cfg.sectors;
  sc.matrix = Eigen::MatrixXf::Zero(cfg.rings, cfg.sectors);
  for (const auto& p : points) {
    const double r = std::hypot(p.x(), p.y());
    if (r >= cfg.max_range || r < 1e-3) continue;
    const double az = std::atan2(p.y(), p.x()) + M_PI;  // [0, 2*pi]
    const int ring = std::min(cfg.rings - 1, static_cast<int>(r / cfg.max_range * cfg.rings));
    const int sector = std::min(cfg.sectors - 1, static_cast<int>(az / (2 * M_PI) * cfg.sectors));
    const float h = p.z() + static_cast<float>(cfg.lidar_height);
    if (h > sc.matrix(ring, sector)) sc.matrix(ring, sector) = h;
  }
  sc.ring_key.resize(cfg.rings);
  for (int i = 0; i < cfg.rings; ++i) {
    sc.ring_key(i) = static_cast<float>((sc.matrix.row(i).array() > 0).count()) / cfg.sectors;
  }
  return sc;
}

namespace {
double column_distance(const ScanContext& a, const ScanContext& b, int shift) {
  double sum = 0;
  int used = 0;
  for (int j = 0; j < a.sectors; ++j) {
    const auto ca = a.matrix.col((j + shift) % a.sectors);
    const auto cb = b.matrix.col(j);
    const float na = ca.norm(), nb = cb.norm();
    if (na < 1e-9f || nb < 1e-9f) continue;  // columns empty in either descriptor are skipped
    sum += 1.0 - ca.dot(cb) / (na * nb);
    ++used;
  }
  return used ? sum / used : 1.0;
}
}  // namespace

ScanContextMatch compare_scan_contexts(const ScanContext& q, const ScanContext& c) {
  ScanContextMatch best;
  if (q.rings != c.rings || q.sectors != c.sectors) return best;
  // Coarse search over all shifts, as in the paper (the ring key makes candidates few).
  for (int s = 0; s < q.sectors; ++s) {
    const double d = column_distance(q, c, s);
    if (d < best.distance) {
      best.distance = d;
      best.shift = s;
    }
  }
  // Query column (j + shift) matches candidate column j: the query frame is rotated by +shift.
  double yaw = -best.shift * 2.0 * M_PI / q.sectors;
  yaw = std::atan2(std::sin(yaw), std::cos(yaw));
  best.yaw = yaw;
  return best;
}

void ScanContextDatabase::add(KeyframeId id, ScanContext sc) {
  entries_.push_back({id, std::move(sc)});
}

std::vector<ScanContextDatabase::Candidate> ScanContextDatabase::query(
    const ScanContext& sc) const {
  std::vector<Candidate> out;
  const int n = static_cast<int>(entries_.size()) - cfg_.exclude_recent;
  if (n <= 0) return out;
  // Stage 1: ring-key nearest neighbours (rotation invariant).
  std::vector<std::pair<float, int>> ranked;
  ranked.reserve(n);
  for (int i = 0; i < n; ++i) ranked.push_back({(entries_[i].sc.ring_key - sc.ring_key).norm(), i});
  const int k = std::min(cfg_.num_candidates, static_cast<int>(ranked.size()));
  std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end());
  // Stage 2: full distance with column alignment.
  for (int i = 0; i < k; ++i) {
    const auto& e = entries_[ranked[i].second];
    const ScanContextMatch m = compare_scan_contexts(sc, e.sc);
    if (m.distance < cfg_.distance_threshold) out.push_back({e.id, m});
  }
  std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
    return a.match.distance < b.match.distance;
  });
  return out;
}

LidarPayload ScanContextDatabase::to_payload(const ScanContext& sc, uint32_t num_points,
                                             float max_range) {
  LidarPayload p;
  p.rings = sc.rings;
  p.sectors = sc.sectors;
  p.num_points = num_points;
  p.max_range = max_range;
  p.scan_context.resize(static_cast<size_t>(sc.rings) * sc.sectors);
  for (int i = 0; i < sc.rings; ++i)
    for (int j = 0; j < sc.sectors; ++j)
      p.scan_context[static_cast<size_t>(i) * sc.sectors + j] = sc.matrix(i, j);
  p.ring_key.assign(sc.ring_key.data(), sc.ring_key.data() + sc.rings);
  return p;
}

ScanContext ScanContextDatabase::from_payload(const LidarPayload& p) {
  ScanContext sc;
  sc.rings = static_cast<int>(p.rings);
  sc.sectors = static_cast<int>(p.sectors);
  sc.matrix.resize(p.rings, p.sectors);
  for (uint32_t i = 0; i < p.rings; ++i)
    for (uint32_t j = 0; j < p.sectors; ++j)
      sc.matrix(i, j) = p.scan_context[static_cast<size_t>(i) * p.sectors + j];
  sc.ring_key = Eigen::Map<const Eigen::VectorXf>(p.ring_key.data(), p.rings);
  return sc;
}

}  // namespace strataslam
