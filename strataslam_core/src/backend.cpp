#include "strataslam/backend/backend.hpp"

#include <cstdio>
#include <cstdlib>

namespace strataslam {

Backend::Backend(const BackendConfig& config, std::shared_ptr<Logger> logger)
    : cfg_(config), log_(std::move(logger)) {}

FloorId Backend::floor_of(KeyframeId id) const {
  const auto it = registry_.find(id);
  return it == registry_.end() ? kUnassignedFloor : it->second.floor;
}

std::vector<FloorId> Backend::floors() const {
  std::vector<FloorId> out;
  out.reserve(graphs_.size());
  for (const auto& kv : graphs_) out.push_back(kv.first);
  return out;
}

Result<void> Backend::add_keyframe(const KeyframeInput& kf, std::optional<KeyframeId> odom_from,
                                   const gtsam::Pose3& odom_rel,
                                   std::optional<KeyframeId> bridge_from) {
  if (registry_.count(kf.id)) return Error{"duplicate keyframe id"};
  registry_[kf.id] = kf;
  order_.push_back(kf.id);
  // Transition keyframes are recorded but join no graph - except in the debug control mode, which
  // emulates a system without floor-gated submaps (one graph, odometry chained through everything).
  if (kf.floor < 0 && !cfg_.debug_disable_floor_gating) return {};
  insert_into_graph(kf, odom_from, odom_rel, bridge_from);
  return {};
}

void Backend::insert_into_graph(const KeyframeInput& kf, std::optional<KeyframeId> odom_from,
                                const gtsam::Pose3& odom_rel,
                                std::optional<KeyframeId> bridge_from) {
  const FloorId g = graph_key(kf.floor);
  auto it = graphs_.find(g);
  if (it == graphs_.end()) {
    // Floor frame origin: yaw and translation of this keyframe, gravity-aligned (no roll/pitch).
    const auto R = kf.T_odom_base.rotation().matrix();
    const gtsam::Pose3 T_odom_floor(gtsam::Rot3::Yaw(std::atan2(R(1, 0), R(0, 0))),
                                    kf.T_odom_base.translation());
    it = graphs_.emplace(g, std::make_unique<FloorGraph>(g, T_odom_floor, cfg_)).first;
  }
  FloorGraph& fg = *it->second;
  const gtsam::Pose3 T_floor_base = fg.T_odom_floor().between(kf.T_odom_base);
  const double scale = kf.tracking == TrackingState::Degraded ? cfg_.degraded_scale : 1.0;
  // Only link with a keyframe that lives in this graph.
  std::optional<KeyframeId> from = (odom_from && fg.has(*odom_from)) ? odom_from : std::nullopt;
  std::optional<KeyframeId> bridge =
      (bridge_from && fg.has(*bridge_from)) ? bridge_from : std::nullopt;
  std::optional<double> baro_dz;
  if (from && kf.baro_height) {
    const auto it_prev = registry_.find(*from);
    if (it_prev != registry_.end()) {
      const std::optional<double>& prev_height = it_prev->second.baro_height;
      if (prev_height) baro_dz = *kf.baro_height - *prev_height;
    }
  }
  if (debug_channel("BACKEND"))
    log_->log(LogLevel::Debug,
              strprintf("insert kf=%lu floor=%d graph=%d session=%u odom_from=%s(%ld) in_graph=%d "
                        "bridge=%d graph_size=%zu",
                        (unsigned long)kf.id, kf.floor, g, kf.session, odom_from ? "yes" : "no",
                        odom_from ? (long)*odom_from : -1L, from.has_value(), bridge.has_value(),
                        fg.size()));
  fg.add_keyframe(kf.id, kf.stamp, kf.session, T_floor_base, from, odom_rel, scale, bridge,
                  baro_dz);
  if (kf.plane_up)
    fg.add_plane(kf.id, *kf.plane_up);
  else if (kf.up_body)
    fg.add_gravity(kf.id, *kf.up_body, cfg_.gravity_sigma);
}

Result<void> Backend::promote_to_floor(const std::vector<KeyframeId>& ids, FloorId floor,
                                       std::optional<KeyframeId> bridge_from,
                                       bool bridge_is_odometry) {
  if (floor < 0) return Error{"cannot promote keyframes to an unassigned floor"};
  std::optional<KeyframeId> prev;
  std::optional<KeyframeId> bridge = bridge_from;
  for (KeyframeId id : ids) {
    const auto it = registry_.find(id);
    if (it == registry_.end()) return Error{"promote_to_floor: unknown keyframe"};
    if (it->second.floor >= 0) return Error{"promote_to_floor: keyframe already has a floor"};
    it->second.floor = floor;
    gtsam::Pose3 rel;
    std::optional<KeyframeId> odom_from, link = prev ? std::nullopt : bridge;
    if (!prev && bridge && bridge_is_odometry && registry_.count(*bridge) &&
        registry_.at(*bridge).session == it->second.session) {
      odom_from = bridge;
      link.reset();
      rel = registry_.at(*bridge).T_odom_base.between(it->second.T_odom_base);
    }
    if (prev) {
      if (registry_.at(*prev).session == it->second.session) {
        odom_from = prev;
        rel = registry_.at(*prev).T_odom_base.between(it->second.T_odom_base);
      } else {
        // A frontend restart in between: the motion across it is unknown, so only a weak link.
        link = prev;
        rel = registry_.at(*prev).T_odom_base.between(it->second.T_odom_base);
      }
    }
    insert_into_graph(it->second, odom_from, rel, link);
    prev = id;
  }
  return {};
}

Result<void> Backend::promote_transit_before(const std::vector<KeyframeId>& ids, FloorId floor,
                                             KeyframeId next) {
  if (floor < 0) return Error{"cannot promote keyframes to an unassigned floor"};
  if (cfg_.debug_disable_floor_gating)
    return {};  // that control mode already put them in its single graph
  if (!registry_.count(next) || registry_.at(next).floor != floor)
    return Error{"promote_transit_before: the next keyframe is not on the floor"};
  KeyframeId nxt = next;
  for (auto it_id = ids.rbegin(); it_id != ids.rend(); ++it_id) {
    const auto it = registry_.find(*it_id);
    if (it == registry_.end()) return Error{"promote_transit_before: unknown keyframe"};
    if (it->second.floor >= 0) return Error{"promote_transit_before: keyframe already has a floor"};
    it->second.floor = floor;
    const KeyframeInput& n = registry_.at(nxt);
    const gtsam::Pose3 rel = n.T_odom_base.between(it->second.T_odom_base);
    if (n.session == it->second.session)
      insert_into_graph(it->second, nxt, rel, std::nullopt);
    else
      insert_into_graph(it->second, std::nullopt, rel,
                        nxt);  // a frontend restart in between: weak link only
    nxt = *it_id;
  }
  graphs_.at(graph_key(floor))->sort_by_stamp();
  return {};
}

void Backend::add_gravity(KeyframeId id, const Eigen::Vector3d& up_body, double sigma_rad) {
  const FloorId f = floor_of(id);
  if (f < 0) return;
  graphs_.at(graph_key(f))->add_gravity(id, up_body, sigma_rad);
}

Result<void> Backend::verify_loop_allowed(KeyframeId a, KeyframeId b) const {
  if (cfg_.debug_disable_floor_gating) return {};
  const FloorId fa = floor_of(a), fb = floor_of(b);
  if (fa < 0 || fb < 0) return Error{"loop closure rejected: keyframe without an assigned floor"};
  if (fa != fb) return Error{"loop closure rejected: keyframes belong to different floors"};
  return {};
}

Result<void> Backend::add_loop(KeyframeId from, KeyframeId to, const gtsam::Pose3& T_from_to) {
  if (!has_keyframe(from) || !has_keyframe(to))
    return Error{"loop closure references an unknown keyframe"};
  if (auto allowed = verify_loop_allowed(from, to); !allowed) return allowed;
  const FloorId fa = floor_of(from), fb = floor_of(to);
  // Layer 3 lives in FloorGraph::add_loop: the graph of `from` refuses a node it does not hold.
  // With the debug flag every floor shares one graph, which is how the negative control passes it.
  FloorGraph& g = *graphs_.at(graph_key(fa));
  auto r = g.add_loop(from, to, T_from_to, /*check_consistency=*/!cfg_.debug_disable_floor_gating);
  if (!r) return r;
  loops_.push_back({from, to, fa, fb});
  return {};
}

gtsam::Pose3 Backend::pose_odom(KeyframeId id) const {
  const KeyframeInput& kf = registry_.at(id);
  if (kf.floor < 0) return kf.T_odom_base;
  const FloorGraph& g = *graphs_.at(graph_key(kf.floor));
  return g.T_odom_floor() * g.pose(id);
}

const std::vector<Eigen::Vector3f>* Backend::cloud(KeyframeId id) const {
  const auto it = registry_.find(id);
  return it == registry_.end() ? nullptr : &it->second.cloud;
}

void Backend::optimize_all() {
  for (auto& kv : graphs_) kv.second->optimize_batch();
}

}  // namespace strataslam
