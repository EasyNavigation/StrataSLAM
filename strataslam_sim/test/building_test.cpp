#include <gtest/gtest.h>

#include <cmath>
#include <map>

#include "strataslam_sim/building.hpp"

using namespace strataslam_sim;
using Eigen::Vector3d;

namespace {
BuildingConfig cfg(bool identical = true, bool atrium = false) {
  BuildingConfig c;
  c.num_floors = 3;
  c.identical_floors = identical;
  c.with_atrium = atrium;
  c.with_ramp = true;
  return c;
}
}  // namespace

TEST(Building, VerticalRaysHitSlabs) {
  const Building b(cfg());
  const auto down = b.raycast({16, 3, 1.2}, {0, 0, -1}, 50, 0, false);
  ASSERT_TRUE(down);
  EXPECT_NEAR(*down, 1.2, 1e-6);
  const auto up = b.raycast({16, 3, 1.2}, {0, 0, 1}, 50, 0, false);
  ASSERT_TRUE(up);
  EXPECT_NEAR(*up, 3.0 - 0.2 - 1.2, 1e-6);  // underside of the next slab
}

TEST(Building, AtriumHasAVoidThroughUpperFloors) {
  const Building plain(cfg(true, false)), atrium(cfg(true, true));
  const Vector3d o(16, 11, 3.0 + 1.5);  // inside the central block, floor 1
  const auto a = plain.raycast(o, {0, 0, -1}, 50, 0, false);
  const auto b = atrium.raycast(o, {0, 0, -1}, 50, 0, false);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  EXPECT_NEAR(*a, 1.5, 1e-6);
  EXPECT_NEAR(*b, 4.5, 1e-6);  // all the way to the ground slab
}

TEST(Building, OuterWallsAndDoorGaps) {
  const Building b(cfg());
  // East wall at x = 32 outside the ramp door; a ray at y = 4 hits it.
  const auto wall = b.raycast({20, 4, 1.2}, {1, 0, 0}, 80, 0, false);
  ASSERT_TRUE(wall);
  // Through the stairs door gap at x = 12, y = 0 into the annex: first hit is the annex back wall.
  const auto gap = b.raycast({11.0, 1.0, 1.2}, {0, -1, 0}, 80, 0, false);
  ASSERT_TRUE(gap);
  EXPECT_GT(*gap, 2.0);  // not blocked by the main south wall
}

TEST(Building, IdenticalFloorsReplicateLandmarksExactly) {
  const Building b(cfg(true));
  // Every landmark of floor 1 has an exact twin (shifted by one floor height, same descriptor) on
  // floor 0. (The top floor is excluded: its last stairs run into the roof.)
  const double H = 3.0;
  int matched = 0, total = 0;
  for (const auto& l1 : b.landmarks()) {
    if (l1.dynamic || l1.level != 1) continue;
    ++total;
    for (const auto& l0 : b.landmarks()) {
      if (l0.dynamic || l0.level != 0) continue;
      if ((l0.position + Vector3d(0, 0, H) - l1.position).norm() < 1e-9 && l0.descriptor == l1.descriptor) {
        ++matched;
        break;
      }
    }
  }
  EXPECT_GT(total, 1000);
  // The doorway lips and stair risers touching the ground-floor extensions are partly buried on
  // floor 0 only: ~0.6 % of landmarks have no twin. Everything else is replicated exactly.
  EXPECT_GE(matched, 0.99 * total);
}

TEST(Building, DistinctFloorsDifferInTextureAndFurniture) {
  const Building b(cfg(false));
  int same_descriptor = 0, total = 0;
  for (const auto& l1 : b.landmarks()) {
    if (l1.dynamic || l1.level != 2) continue;
    for (const auto& l0 : b.landmarks()) {
      if (l0.dynamic || l0.level != 1) continue;
      if ((l0.position + Vector3d(0, 0, 3) - l1.position).norm() < 1e-9) {
        ++total;
        if (l0.descriptor == l1.descriptor) ++same_descriptor;
      }
    }
  }
  EXPECT_EQ(same_descriptor, 0);
}

TEST(Building, CabinMovesAndDoorBlocks) {
  const Building b(cfg());
  // Inside the cabin looking out through the door: open door sees the lobby, closed door sees the leaf.
  const Vector3d o(21.6, -1.1, 1.2);
  const auto open = b.raycast(o, {0, 1, 0}, 50, 0, false);
  const auto closed = b.raycast(o, {0, 1, 0}, 50, 0, true);
  ASSERT_TRUE(open);
  ASSERT_TRUE(closed);
  EXPECT_NEAR(*closed, 0.8, 1e-6);  // leaf at y = -0.3
  EXPECT_GT(*open, 3.0);
  // Moving the cabin up by 3 m moves its walls with it.
  const auto ceiling = b.raycast(o + Vector3d(0, 0, 3.0), {0, 0, 1}, 50, 3.0, true);
  ASSERT_TRUE(ceiling);
  EXPECT_NEAR(*ceiling, 2.4 - 1.2, 1e-6);
}

TEST(Building, SurfaceFunctions) {
  const Building b(cfg());
  EXPECT_NEAR(*b.stairs_surface(0, 11.0, -0.1), 0.0, 1e-12);
  EXPECT_NEAR(*b.stairs_surface(0, 11.0, -0.3 - 0.27 * 3.5), 0.1875 * 4, 1e-12);
  EXPECT_NEAR(*b.stairs_surface(0, 11.0, -3.5), 1.5, 1e-12);
  EXPECT_NEAR(*b.stairs_surface(0, 13.2, -0.35), 3.0 - 0.1875 * 0 - 0.0, 1.0);  // near the top of B
  EXPECT_FALSE(b.stairs_surface(0, 5.0, 3.0).has_value());
  EXPECT_NEAR(*b.ramp_surface(0, 36.15, 9.1), 0.2 * (36.15 - 32.4), 1e-9);
  EXPECT_NEAR(*b.ramp_surface(0, 41.0, 9.5), 1.5, 1e-12);
}

TEST(Building, LandmarksAreOnSurfaces) {
  const Building b(cfg());
  int near_surface = 0, n = 0;
  for (size_t i = 0; i < b.landmarks().size(); i += 37) {
    const auto& lm = b.landmarks()[i];
    ++n;
    // The landmark must be within 3 cm of some box surface and not strictly inside any box.
    bool inside = false, on_surface = false;
    for (const auto& box : b.boxes()) {
      if (box.dynamic != lm.dynamic) continue;
      if (point_in_box(lm.position, box, 0.03) && !point_in_box(lm.position, box, -0.03)) on_surface = true;
      if (point_in_box(lm.position, box, -0.03)) inside = true;
    }
    if (on_surface && !inside) ++near_surface;
  }
  EXPECT_GT(near_surface, 0.95 * n);
}

// With identical floors the top floor's ceiling is the same slab, with the same textures, as every other floor's
// ceiling (the floor slab of the level above): a sensor looking up cannot tell the top floor apart (audit AUD-024).
TEST(Building, TheTopFloorsCeilingIsLikeEveryOtherCeiling) {
  BuildingConfig c = cfg(true);
  const Building b(c);
  const double H = c.floor_height;
  int roof = 0, twins = 0;
  for (const auto& r : b.landmarks()) {
    // The roof's upper face looks at the sky (an intermediate slab's is under the walls of the floor above): only
    // what can be seen from inside counts.
    if (r.dynamic || r.level != c.num_floors || r.position.z() > c.num_floors * H - 1e-6) continue;
    ++roof;
    for (const auto& l : b.landmarks()) {
      if (l.dynamic || l.level != c.num_floors - 1) continue;
      if ((l.position + Vector3d(0, 0, H) - r.position).norm() < 1e-9 && l.descriptor == r.descriptor) {
        ++twins;
        break;
      }
    }
  }
  EXPECT_GT(roof, 100);
  EXPECT_GE(twins, 0.99 * roof);  // as for the floors: a few landmarks on the slab's edges are buried differently
  // Floors that differ keep a roof of their own.
  BuildingConfig d = cfg(false);
  const Building bd(d);
  int roof_d = 0, twins_d = 0;
  for (const auto& r : bd.landmarks()) {
    if (r.dynamic || r.level != d.num_floors) continue;
    ++roof_d;
    for (const auto& l : bd.landmarks())
      if (!l.dynamic && l.level == d.num_floors - 1 && (l.position + Vector3d(0, 0, H) - r.position).norm() < 1e-9 && l.descriptor == r.descriptor) ++twins_d;
  }
  EXPECT_GT(roof_d, 100);
  EXPECT_EQ(twins_d, 0);
}

// The roof closes the building wherever there is a floor below it, annexes included: with identical floors it is a
// copy of the top slab, and that slab has openings where the stairs, the elevator and the ramp arrive from below.
TEST(Building, TheRoofIsClosedOverTheTopFloorAndEveryAnnex) {
  for (const bool identical : {true, false}) {
    for (const bool atrium : {false, true}) {
      BuildingConfig c = cfg(identical, atrium);
      const Building b(c);
      const double H = c.floor_height, z = (c.num_floors - 1) * H + 1.5;  // 1.5 m above the top floor
      const std::map<std::string, Vector3d> spots = {
          {"main", {16, 3, z}},          {"atrium", {16, 11, z}},     {"stairs A", {11.0, -1.5, z}},
          {"stairs B", {13.2, -1.5, z}}, {"stair landing", {12, -3.3, z}}, {"elevator", {21.6, -1.8, z}},
          {"ramp A", {34, 9.1, z}},      {"ramp B", {34, 10.9, z}},   {"ramp landing", {b.ramp_land_x1() - 1.0, 10, z}}};
      for (const auto& [name, o] : spots) {
        const auto up = b.raycast(o, {0, 0, 1}, 50, 0, false);
        EXPECT_TRUE(up && *up < H) << name << (identical ? " (identical floors)" : " (distinct floors)") << (atrium ? ", atrium" : "");
      }
    }
  }
}
