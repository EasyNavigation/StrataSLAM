#include "strataslam_sim/building.hpp"

#include <algorithm>
#include <cmath>

#include "strataslam_sim/rng.hpp"

namespace strataslam_sim {

using Eigen::Matrix3d;
using Eigen::Vector3d;

bool point_in_box(const Vector3d& p, const Box& b, double margin) {
  const Vector3d l = b.R.transpose() * (p - b.center);
  return std::abs(l.x()) <= b.half.x() + margin && std::abs(l.y()) <= b.half.y() + margin &&
         std::abs(l.z()) <= b.half.z() + margin;
}

bool ray_box(const Vector3d& origin, const Vector3d& dir, const Box& b, double t_min, double t_max,
             double* t) {
  const Vector3d o = b.R.transpose() * (origin - b.center);
  const Vector3d d = b.R.transpose() * dir;
  double lo = t_min, hi = t_max;
  for (int a = 0; a < 3; ++a) {
    if (std::abs(d[a]) < 1e-12) {
      if (std::abs(o[a]) > b.half[a]) return false;
      continue;
    }
    double t1 = (-b.half[a] - o[a]) / d[a], t2 = (b.half[a] - o[a]) / d[a];
    if (t1 > t2) std::swap(t1, t2);
    lo = std::max(lo, t1);
    hi = std::min(hi, t2);
    if (lo > hi) return false;
  }
  *t = lo;
  return true;
}

namespace {
constexpr double kStairX0 = 10.2, kStairX1 = 14.0;       // annex interior
constexpr double kStairAx1 = 11.8, kStairBx0 = 12.4;      // flight A: [10.2, 11.8], B: [12.4, 14.0]
constexpr double kStairY0 = -0.3, kStairLandingY = -2.46, kStairAnnexY = -4.1;
constexpr double kRiser = 0.1875;
constexpr double kRampX0 = 32.4;
constexpr double kRampAy0 = 8.4, kRampAy1 = 9.8, kRampBy0 = 10.2, kRampBy1 = 11.6;
constexpr double kElevX0 = 21.0, kElevX1 = 22.2;
}  // namespace

Building::Building(const BuildingConfig& config) : config_(config) {
  for (int k = 0; k < config_.num_floors; ++k) build_level(k);
  // Roof over the footprint and annexes.
  const double zt = config_.num_floors * config_.floor_height;
  if (config_.identical_floors && config_.num_floors > 1) {
    // Identical floors must be identical from the inside, ceiling included: the top floor's ceiling is a copy of the
    // slab that is the ceiling of every other floor (the floor slab of the level above), with the same textures.
    // Before 2026-11-04 it was a plain roof with textures of its own, so a LiDAR or a camera looking up told the top
    // floor apart and S03 tested less than it claims (audit AUD-024).
    const int top = config_.num_floors - 1;
    const double slab_mid = top * config_.floor_height - 0.5 * kSlab;
    const size_t n = boxes_.size();
    for (size_t i = 0; i < n; ++i) {
      const Box& b = boxes_[i];
      if (b.level != top || b.dynamic || std::abs(b.center.z() - slab_mid) > 1e-6 || std::abs(b.half.z() - 0.5 * kSlab) > 1e-6) continue;
      const Vector3d up(0, 0, config_.floor_height);
      add_box(b.center - b.half + up, b.center + b.half + up, config_.num_floors);
      roof_copy_of_[boxes_.size() - 1] = i;
    }
    // An atrium leaves the central void open in the slabs of the upper floors; the roof closes it (with its own
    // texture: seen from the atrium it is the roof, not a floor).
    if (config_.with_atrium) add_box({6.2, 6.2, zt - kSlab}, {25.8, 15.8, zt}, config_.num_floors);
    // The copied slab is open where the stairs, the elevator and the ramp arrive from the floor below: each annex is
    // capped above it, so the shaft ends 0.2 m higher and the copy's own faces are left as they are (a cap level
    // with them would add a second set of landmarks on the same surface).
    const auto cap = [&](double x0, double y0, double x1, double y1) { add_box({x0, y0, zt}, {x1, y1, zt + kSlab}, config_.num_floors); };
    if (config_.with_stairs) cap(9.8, -4.3, 14.2, -0.2);
    if (config_.with_elevator) cap(20.2, -3.5, 23.0, -0.2);
    if (config_.with_ramp) cap(32.0, 8.2, ramp_land_x1() + 0.2, 11.8);
  } else {
    const Vector3d lo(-0.2, -0.2, zt - kSlab), hi(kSizeX + 0.2, kSizeY + 0.2, zt);
    add_box(lo, hi, config_.num_floors);
    if (config_.with_stairs) add_box({9.8, -4.3, zt - kSlab}, {14.2, -0.2, zt}, config_.num_floors);
    if (config_.with_elevator) add_box({20.2, -3.5, zt - kSlab}, {23.0, -0.2, zt}, config_.num_floors);
    if (config_.with_ramp) add_box({32.0, 8.2, zt - kSlab}, {ramp_land_x1() + 0.2, 11.8, zt}, config_.num_floors);
  }
  anchors_.elevator_door_corners = {{kElevX0, 0.0, 0.0}, {kElevX1, 0.0, 0.0},
                                    {kElevX0, 0.0, 2.2}, {kElevX1, 0.0, 2.2}};
  for (double z : {0.0, 2.2}) {
    for (double x : {20.4, 22.8}) {
      for (double y : {-3.3, -0.2}) anchors_.elevator_shaft_corners.push_back({x, y, z});
    }
  }
  for (double z : {0.0, 2.2}) {
    for (double x : {10.2, 14.0}) {
      for (double y : {-4.1, -0.2}) anchors_.stair_shaft_corners.push_back({x, y, z});
    }
  }
  for (double z : {0.0, 2.2}) {
    for (double x : {0.0, kSizeX}) {
      for (double y : {0.0, kSizeY}) anchors_.building_corners.push_back({x, y, z});
    }
  }
  generate_landmarks();
}

void Building::add_box(const Vector3d& lo, const Vector3d& hi, int level, bool dynamic, bool door) {
  Box b;
  b.center = 0.5 * (lo + hi);
  b.half = 0.5 * (hi - lo);
  b.dynamic = dynamic;
  b.door = door;
  b.level = level;
  boxes_.push_back(b);
}

void Building::add_oriented(const Vector3d& center, const Vector3d& half, const Matrix3d& R, int level) {
  Box b;
  b.center = center;
  b.half = half;
  b.R = R;
  b.level = level;
  boxes_.push_back(b);
}

void Building::build_level(int k) {
  const double H = config_.floor_height, T = kSlab;
  const double z0 = k * H, hw = H - T;
  const size_t first = boxes_.size();
  const auto wall = [&](double x0, double x1, double y0, double y1) {
    add_box({x0, y0, z0}, {x1, y1, z0 + hw}, k);
  };
  const auto slab = [&](double x0, double x1, double y0, double y1, double zt = 0.0) {
    add_box({x0, y0, z0 + zt - T}, {x1, y1, z0 + zt}, k);
  };
  const auto lintel = [&](double x0, double x1, double y0, double y1) {
    add_box({x0, y0, z0 + 2.2}, {x1, y1, z0 + hw}, k);
  };
  const bool st = config_.with_stairs, el = config_.with_elevator, rp = config_.with_ramp;

  // Slab: four pieces around the central void; the void itself is filled at the end of the level
  // (unless this is an upper floor of a building with an atrium). Keeping box indices equal on
  // every level is what makes identical floors share textures.
  slab(0, 6.2, 0, kSizeY);
  slab(25.8, kSizeX, 0, kSizeY);
  slab(6.2, 25.8, 0, 6.2);
  slab(6.2, 25.8, 15.8, kSizeY);

  // Outer walls (south wall has door gaps for the annexes, east wall for the ramp).
  std::vector<std::pair<double, double>> south;
  double cursor = -0.2;
  std::vector<std::pair<double, double>> gaps;
  if (st) gaps.push_back({kStairX0, kStairX1});
  if (el) gaps.push_back({kElevX0, kElevX1});
  std::sort(gaps.begin(), gaps.end());
  for (const auto& g : gaps) {
    wall(cursor, g.first, -0.2, 0.0);
    lintel(g.first, g.second, -0.2, 0.0);
    cursor = g.second;
  }
  wall(cursor, kSizeX + 0.2, -0.2, 0.0);
  wall(-0.2, kSizeX + 0.2, kSizeY, kSizeY + 0.2);
  wall(-0.2, 0.0, 0.0, kSizeY);
  if (rp) {
    wall(kSizeX, kSizeX + 0.2, 0.0, kRampAy0);
    wall(kSizeX, kSizeX + 0.2, kRampBy1, kSizeY);
    lintel(kSizeX, kSizeX + 0.2, kRampAy0, kRampBy1);
  } else {
    wall(kSizeX, kSizeX + 0.2, 0.0, kSizeY);
  }
  // Central block (rooms, or atrium enclosure).
  wall(6.0, 6.2, 6.0, 16.0);
  wall(25.8, 26.0, 6.0, 16.0);
  wall(6.2, 25.8, 6.0, 6.2);
  wall(6.2, 25.8, 15.8, 16.0);

  // Pillars.
  const double pil[][2] = {{2, 8}, {2, 14}, {30, 8}, {30, 14}, {8, 2}, {16, 1.8},
                           {24, 2}, {8, 20}, {16, 20.2}, {24, 20}};
  for (const auto& p : pil) add_box({p[0] - 0.3, p[1] - 0.3, z0}, {p[0] + 0.3, p[1] + 0.3, z0 + hw}, k);

  // Furniture along walls; same placement on every floor when floors are identical.
  Rng rng(config_.identical_floors ? hash_combine(config_.seed, 777) : hash_combine(config_.seed, 777, k));
  for (int i = 0; i < 12; ++i) {
    const int side = rng.below(4);
    const double w = rng.uniform(0.6, 1.4), d = rng.uniform(0.4, 0.8), h = rng.uniform(0.8, 1.7);
    double along, cx, cy, sx, sy;
    if (side == 0 || side == 1) {  // south / north outer wall (keep clear of annex doors)
      do { along = rng.uniform(1.5, kSizeX - 1.5); } while (along > 7.5 && along < 24.5);
      cx = along; sx = w; sy = d;
      cy = side == 0 ? 0.3 + d / 2 : kSizeY - 0.3 - d / 2;
    } else {  // west / east outer wall (keep clear of ramp door)
      do { along = rng.uniform(1.5, kSizeY - 1.5); } while (along > 6.0 && along < 14.0);
      cy = along; sx = d; sy = w;
      cx = side == 2 ? 0.3 + d / 2 : kSizeX - 0.3 - d / 2;
    }
    add_box({cx - sx / 2, cy - sy / 2, z0}, {cx + sx / 2, cy + sy / 2, z0 + h}, k);
  }

  // Stairs annex: U-shaped stair, two flights of 8 steps and a mid landing.
  if (st) {
    wall(10.0, 10.2, kStairAnnexY - 0.2, -0.2);
    wall(14.0, 14.2, kStairAnnexY - 0.2, -0.2);
    wall(10.0, 14.2, kStairAnnexY - 0.2, kStairAnnexY);
    wall(11.9, 12.3, -2.6, -0.1);  // divider between flights
    for (int i = 0; i < 8; ++i) {
      const double ya = kStairY0 - kStairTread * (i + 1), yb = kStairY0 - kStairTread * i;
      add_box({kStairX0, ya, z0 + kRiser * i}, {kStairAx1, yb, z0 + kRiser * (i + 1)}, k);
      const double ja = kStairLandingY + kStairTread * i, jb = kStairLandingY + kStairTread * (i + 1);
      add_box({kStairBx0, ja, z0 + kStairRise + kRiser * i}, {kStairX1, jb, z0 + kStairRise + kRiser * (i + 1)}, k);
    }
    add_box({kStairX0, kStairAnnexY, z0 + kStairRise - T}, {kStairX1, kStairLandingY, z0 + kStairRise}, k);
    add_box({kStairX0, -0.3, z0 - T}, {kStairX1, 0.0, z0}, k);  // doorway lip
  }

  // Elevator annex: shaft walls, doorway lip; the cabin is added once below.
  if (el) {
    wall(20.2, 20.4, -3.5, -0.2);
    wall(22.8, 23.0, -3.5, -0.2);
    wall(20.2, 23.0, -3.5, -3.3);
    add_box({kElevX0, -0.3, z0 - T}, {kElevX1, 0.0, z0}, k);
  }

  // Ramp annex: U-shaped ramp, 11.3 degrees, two flights of 1.5 m rise and a landing.
  if (rp) {
    const double kRampX1 = ramp_x1(), kRampLandX1 = ramp_land_x1();
    wall(32.2, kRampLandX1 + 0.2, 8.2, 8.4);
    wall(32.2, kRampLandX1 + 0.2, 11.6, 11.8);
    wall(kRampLandX1, kRampLandX1 + 0.2, 8.2, 11.8);
    wall(32.2, kRampX1 + 0.1, 9.8, 10.2);
    const double len = std::hypot(kRampX1 - kRampX0, kStairRise), c = 1.0 / len;
    {  // flight A rises toward +x
      const Vector3d dx(kRampX1 - kRampX0, 0, kStairRise), xl = dx.normalized();
      const Vector3d yl(0, 1, 0), zl = xl.cross(yl);
      Matrix3d R; R.col(0) = xl; R.col(1) = yl; R.col(2) = zl;
      const Vector3d mid(0.5 * (kRampX0 + kRampX1), 0.5 * (kRampAy0 + kRampAy1), z0 + 0.5 * kStairRise);
      add_oriented(mid - 0.1 * zl, {0.5 * len, 0.5 * (kRampAy1 - kRampAy0), 0.1}, R, k);
    }
    {  // flight B rises toward -x
      const Vector3d xl = Vector3d(kRampX1 - kRampX0, 0, -kStairRise).normalized();
      const Vector3d yl(0, 1, 0), zl = xl.cross(yl);
      Matrix3d R; R.col(0) = xl; R.col(1) = yl; R.col(2) = zl;
      const Vector3d mid(0.5 * (kRampX0 + kRampX1), 0.5 * (kRampBy0 + kRampBy1), z0 + 1.5 * kStairRise);
      add_oriented(mid - 0.1 * zl, {0.5 * len, 0.5 * (kRampBy1 - kRampBy0), 0.1}, R, k);
    }
    (void)c;
    add_box({kRampX1, kRampAy0, z0 + kStairRise - T}, {kRampLandX1, kRampBy1, z0 + kStairRise}, k);
    add_box({32.0, kRampAy0, z0 - T}, {kRampX0, kRampBy1, z0}, k);
  }

  if (!(config_.with_atrium && k >= 1)) slab(6.2, 25.8, 6.2, 15.8);  // fill the central void

  // Ground-level extensions of the slab under the annexes (distinct texture key).
  if (k == 0) {
    add_box({9.8, -4.3, -T}, {14.2, 0.0, 0.0}, 0);
    add_box({20.2, -3.5, -T}, {23.0, 0.0, 0.0}, 0);
    add_box({32.0, 8.2, -T}, {ramp_land_x1() + 0.2, 11.8, 0.0}, 0);
    // Elevator cabin (dynamic), added once. Interior x[20.7,22.5], y[-2.0,-0.2], height 2.4.
    if (el) {
      const auto cab = [&](Vector3d lo, Vector3d hi, bool door = false) {
        add_box(lo, hi, 0, true, door);
      };
      cab({20.6, -2.1, 0.0}, {20.7, -0.2, 2.4});
      cab({22.5, -2.1, 0.0}, {22.6, -0.2, 2.4});
      cab({20.6, -2.1, 0.0}, {22.6, -2.0, 2.4});
      cab({20.6, -0.3, 0.0}, {kElevX0, -0.2, 2.4});
      cab({kElevX1, -0.3, 0.0}, {22.6, -0.2, 2.4});
      cab({20.6, -2.1, -0.1}, {22.6, -0.2, 0.0});
      cab({20.6, -2.1, 2.4}, {22.6, -0.2, 2.5});
      cab({kElevX0, -0.3, 0.0}, {kElevX1, -0.2, 2.4}, true);
    }
  }

  // Texture keys: boxes at the same index of every level share a key when floors are identical.
  box_key_.resize(boxes_.size());
  for (size_t i = first; i < boxes_.size(); ++i) {
    const uint64_t idx = i - first;
    box_key_[i] = config_.identical_floors ? hash_combine(config_.seed, idx) : hash_combine(config_.seed, idx, k + 1);
  }
}

void Building::generate_landmarks() {
  box_key_.resize(boxes_.size(), 0);
  // Roof boxes (added after the levels) get unique keys, unless they copy a slab of identical floors (its key).
  for (size_t i = 0; i < boxes_.size(); ++i) {
    if (boxes_[i].level != config_.num_floors) continue;
    const auto copy = roof_copy_of_.find(i);
    box_key_[i] = copy != roof_copy_of_.end() ? box_key_[copy->second] : hash_combine(config_.seed, 9999, i);
  }
  for (size_t bi = 0; bi < boxes_.size(); ++bi) {
    const Box& b = boxes_[bi];
    for (int face = 0; face < 6; ++face) {
      const int axis = face / 2;
      const double sgn = (face % 2) ? 1.0 : -1.0;
      const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
      const double area = 4.0 * b.half[a1] * b.half[a2];
      const bool horizontal = std::abs(b.R(2, axis)) > 0.7;  // face normal mostly vertical
      // Two texture scales: coarse landmarks seen from anywhere and fine ones seen from close by, as
      // a real detector finds features at every scale. Even small faces (stair treads) carry a few.
      for (int layer = 0; layer < 2; ++layer) {
        // The elevator interior (buttons, signage, mirror) is feature-rich in practice.
        const double density = (layer == 0 ? config_.landmark_density : config_.fine_density) * (horizontal ? 0.6 : 1.0) * (b.dynamic ? 3.0 : 1.0);
        const int n = std::max(area > 0.05 ? 3 : 0, static_cast<int>(std::round(area * density)));
        Rng rng(hash_combine(box_key_[bi], face, 17 + 1000 * layer));
        for (int i = 0; i < n; ++i) {
          Vector3d local;
          local[axis] = sgn * b.half[axis];
          local[a1] = rng.uniform(-b.half[a1], b.half[a1]);
          local[a2] = rng.uniform(-b.half[a2], b.half[a2]);
          const Vector3d p = b.center + b.R * local;
          const Vector3d probe = p + 0.02 * (b.R * Vector3d::Unit(axis) * sgn);
          bool buried = false;
          for (size_t bj = 0; bj < boxes_.size() && !buried; ++bj) {
            if (bj != bi && boxes_[bj].dynamic == b.dynamic && point_in_box(probe, boxes_[bj])) buried = true;
          }
          if (buried) continue;
          Landmark lm;
          lm.position = p;
          lm.dynamic = b.dynamic;
          lm.door = b.door;
          lm.level = b.level;
          lm.max_dist = layer == 0 ? 0.f : static_cast<float>(config_.fine_range);
          Rng dr(hash_combine(box_key_[bi], face, 1000 + i + 1000000 * layer));
          for (size_t w = 0; w < strataslam::kDescriptorBytes / 8; ++w) {
            const uint64_t v = dr.next();
            for (int byte = 0; byte < 8; ++byte) lm.descriptor[w * 8 + byte] = static_cast<uint8_t>(v >> (8 * byte));
          }
          landmarks_.push_back(lm);
        }
      }
    }
  }
}

std::optional<double> Building::stairs_surface(int level, double x, double y) const {
  const double z0 = level * config_.floor_height;
  if (x >= kStairX0 && x <= kStairAx1 && y <= 0.0 && y >= kStairAnnexY) {
    if (y >= kStairY0) return z0;
    if (y > kStairLandingY) return z0 + kRiser * (std::floor((kStairY0 - y) / kStairTread) + 1);
    return z0 + kStairRise;
  }
  if (x >= kStairBx0 && x <= kStairX1 && y <= 0.0 && y >= kStairAnnexY) {
    if (y >= kStairY0) return z0 + 2 * kStairRise;
    if (y > kStairLandingY) return z0 + kStairRise + kRiser * (std::floor((y - kStairLandingY) / kStairTread) + 1);
    return z0 + kStairRise;
  }
  if (x > kStairAx1 && x < kStairBx0 && y < kStairLandingY && y >= kStairAnnexY) return z0 + kStairRise;
  return std::nullopt;
}

std::optional<double> Building::ramp_surface(int level, double x, double y) const {
  const double z0 = level * config_.floor_height;
  if (x < 32.0 || x > ramp_land_x1() || y < kRampAy0 || y > kRampBy1) return std::nullopt;
  if (x >= ramp_x1()) return z0 + kStairRise;
  if (y <= kRampAy1) return z0 + ramp_slope() * std::max(0.0, x - kRampX0);
  if (y >= kRampBy0) return z0 + kStairRise + ramp_slope() * (ramp_x1() - std::max(x, kRampX0));
  return std::nullopt;
}

std::optional<double> Building::raycast(const Vector3d& o, const Vector3d& d, double max_range,
                                        double cabin_z, bool door_closed) const {
  double best = max_range;
  bool hit = false;
  for (const Box& b : boxes_) {
    if (b.door && !door_closed) continue;
    Box shifted;
    const Box* bp = &b;
    if (b.dynamic) {
      shifted = b;
      shifted.center.z() += cabin_z;
      bp = &shifted;
    }
    double t;
    if (ray_box(o, d, *bp, 1e-6, best, &t)) {
      best = t;
      hit = true;
    }
  }
  if (!hit) return std::nullopt;
  return best;
}

bool Building::visible(const Vector3d& origin, const Vector3d& target, double cabin_z,
                       bool door_closed) const {
  const Vector3d dv = target - origin;
  const double dist = dv.norm();
  if (dist < 1e-6) return true;
  return !raycast(origin, dv / dist, dist - 0.06, cabin_z, door_closed).has_value();
}

RayScene::RayScene(const Building& b, double cabin_z, bool door_closed, const Vector3d& center,
                   double radius) {
  for (const Box& box : b.boxes()) {
    if (box.door && !door_closed) continue;
    Box bx = box;
    if (bx.dynamic) bx.center.z() += cabin_z;
    if ((bx.center - center).norm() - bx.half.norm() > radius) continue;
    if (bx.R.isIdentity(1e-12)) aabbs_.push_back({bx.center - bx.half, bx.center + bx.half});
    else obbs_.push_back(bx);
  }
}

std::optional<double> RayScene::raycast(const Vector3d& o, const Vector3d& d, double max_range) const {
  double best = max_range;
  bool hit = false;
  const Vector3d inv(std::abs(d.x()) < 1e-12 ? 1e12 : 1.0 / d.x(), std::abs(d.y()) < 1e-12 ? 1e12 : 1.0 / d.y(),
                     std::abs(d.z()) < 1e-12 ? 1e12 : 1.0 / d.z());
  for (const auto& b : aabbs_) {
    double lo = 1e-6, hi = best;
    for (int a = 0; a < 3 && lo <= hi; ++a) {
      double t1 = (b.lo[a] - o[a]) * inv[a], t2 = (b.hi[a] - o[a]) * inv[a];
      if (t1 > t2) std::swap(t1, t2);
      lo = std::max(lo, t1);
      hi = std::min(hi, t2);
    }
    if (lo <= hi) {
      best = lo;
      hit = true;
    }
  }
  for (const auto& b : obbs_) {
    double t;
    if (ray_box(o, d, b, 1e-6, best, &t)) {
      best = t;
      hit = true;
    }
  }
  if (!hit) return std::nullopt;
  return best;
}

bool RayScene::visible(const Vector3d& origin, const Vector3d& target) const {
  const Vector3d dv = target - origin;
  const double dist = dv.norm();
  if (dist < 1e-6) return true;
  return !raycast(origin, dv / dist, dist - 0.06).has_value();
}

}  // namespace strataslam_sim
