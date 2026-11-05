#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "strataslam/fusion/floor_stacking.hpp"
#include "strataslam/fusion/fusion.hpp"

using namespace strataslam;

namespace {

void wall(PointCloud& c, double x0, double y0, double x1, double y1, double z0 = 0.5,
          double z1 = 2.5) {
  const double len = std::hypot(x1 - x0, y1 - y0);
  for (double s = 0; s <= len; s += 0.05)
    for (double z = z0; z <= z1; z += 0.5)
      c.points.emplace_back(static_cast<float>(x0 + (x1 - x0) * s / len),
                            static_cast<float>(y0 + (y1 - y0) * s / len), static_cast<float>(z));
}

/// An asymmetric building: outer rectangle, an L of inner walls and a few columns.
PointCloud building() {
  PointCloud c;
  wall(c, 0, 0, 30, 0);
  wall(c, 30, 0, 30, 20);
  wall(c, 30, 20, 0, 20);
  wall(c, 0, 20, 0, 0);
  wall(c, 10, 0, 10, 12);
  wall(c, 10, 12, 22, 12);
  for (const auto& p : {Eigen::Vector2d(24, 5), Eigen::Vector2d(26, 16), Eigen::Vector2d(5, 16)})
    wall(c, p.x(), p.y(), p.x() + 0.5, p.y());
  return c;
}

PointCloud moved(
    const PointCloud& c,
    const Pose4& T_a_b) {  // the cloud that floor b would see: points of a in b's frame
  PointCloud out;
  const Pose4 inv = T_a_b.inverse();
  const double cs = std::cos(inv.theta()), sn = std::sin(inv.theta());
  for (const auto& p : c.points)
    out.points.emplace_back(static_cast<float>(cs * p.x() - sn * p.y() + inv.translation().x()),
                            static_cast<float>(sn * p.x() + cs * p.y() + inv.translation().y()),
                            p.z());
  return out;
}

}  // namespace

TEST(FloorStacking, RecoversTheRelativePoseFromADriftedInitialGuess) {
  const PointCloud a = building();
  const Pose4 truth(0.09, Eigen::Vector3d(3.0, -2.0, 4.0));
  const PointCloud b = moved(a, truth);
  const Pose4 guess(0.09 + 0.10,
                    Eigen::Vector3d(3.0 + 4.0, -2.0 - 3.0, 4.0));  // 4+3 m and 6 degrees off
  std::string why;
  const auto m = match_floor_structure(a, b, guess, StackingOptions(), &why);
  ASSERT_TRUE(m.has_value()) << why;
  EXPECT_NEAR(m->a_to_b.translation().x(), 3.0, 0.3);
  EXPECT_NEAR(m->a_to_b.translation().y(), -2.0, 0.3);
  EXPECT_NEAR(m->a_to_b.theta(), 0.09, 0.02);
  EXPECT_DOUBLE_EQ(m->a_to_b.translation().z(), 4.0);  // the height is the caller's
}

TEST(FloorStacking, TheOtherWayRoundGivesTheInverse) {
  const PointCloud a = building();
  const Pose4 truth(-0.07, Eigen::Vector3d(-5.0, 2.0, 0.0));
  const PointCloud b = moved(a, truth);
  const auto m = match_floor_structure(b, a, truth.inverse());
  ASSERT_TRUE(m.has_value());
  const Pose4 err = m->a_to_b * truth;
  EXPECT_LT(err.translation().head<2>().norm(), 0.3);
  EXPECT_LT(std::abs(err.theta()), 0.02);
}

TEST(FloorStacking, FloorsThatDoNotStackAreNotMatched) {
  PointCloud other;  // a different layout
  wall(other, 0, 0, 12, 0);
  wall(other, 12, 0, 12, 25);
  wall(other, 3, 8, 3, 25);
  wall(other, 6, 3, 9, 20);
  std::string why;
  EXPECT_FALSE(match_floor_structure(building(), other, Pose4(0, Eigen::Vector3d(0, 0, 3)),
                                     StackingOptions(), &why)
                   .has_value());
  EXPECT_FALSE(why.empty());
}

TEST(FloorStacking, ARepetitiveLayoutIsAmbiguous) {
  PointCloud corridor;  // two long walls and identical columns every 4 m
  wall(corridor, 0, 0, 60, 0);
  wall(corridor, 0, 6, 60, 6);
  for (double x = 2; x < 60; x += 4) wall(corridor, x, 3, x + 0.5, 3);
  PointCloud
      middle;  // what the other floor saw of it: a stretch of 40 m, which fits at every column
  for (const auto& p : corridor.points)
    if (p.x() >= 10 && p.x() <= 50) middle.points.push_back(p);
  std::string why;
  const auto m = match_floor_structure(corridor, middle, Pose4(0, Eigen::Vector3d(0, 0, 3)),
                                       StackingOptions(), &why);
  EXPECT_FALSE(m.has_value());
  EXPECT_NE(why.find("ambiguous"), std::string::npos) << why;
}

TEST(FloorStacking, TooLittleStructureIsRefused) {
  PointCloud tiny;
  wall(tiny, 0, 0, 3, 0);
  std::string why;
  EXPECT_FALSE(
      match_floor_structure(tiny, building(), Pose4(), StackingOptions(), &why).has_value());
  EXPECT_NE(why.find("too little"), std::string::npos) << why;
}

TEST(FloorStacking, FloorsAndCeilingsAreNotStructure) {
  PointCloud a = building(), b = moved(a, Pose4(0.0, Eigen::Vector3d(2, 1, 3)));
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> ux(-5, 35), uy(-5, 25);
  for (int i = 0; i < 40000; ++i) {  // a floor at z = 0 and a ceiling at z = 3.4 on both clouds
    b.points.emplace_back(ux(rng), uy(rng), 0.0f);
    b.points.emplace_back(ux(rng), uy(rng), 3.4f);
  }
  const auto m = match_floor_structure(a, b, Pose4(0.02, Eigen::Vector3d(1.5, 1.5, 3)));
  ASSERT_TRUE(m.has_value());
  EXPECT_NEAR(m->a_to_b.translation().x(), 2.0, 0.3);
}

TEST(SameLevel, AFloorSplitByDriftIsACandidateAndAFloorAboveIsNot) {
  // The same structure seen as two floors: 1.6 m apart (a drift of the odometry in z) or a storey
  // up (4 m).
  const PointCloud a = building();
  const Pose4 split(0.05, Eigen::Vector3d(1.0, -1.0, 1.6));
  std::string why;
  const auto c = same_level_candidate(2, a, 3, moved(a, split), split, FusionOptions(), &why);
  ASSERT_TRUE(c.has_value()) << why;
  EXPECT_EQ(c->a, 2);
  EXPECT_EQ(c->b, 3);
  EXPECT_DOUBLE_EQ(c->dz, 1.6);
  EXPECT_GE(c->score, FusionOptions().same_level_min_score);
  const Pose4 storey(0.05, Eigen::Vector3d(1.0, -1.0, 4.0));
  EXPECT_FALSE(
      same_level_candidate(2, a, 3, moved(a, storey), storey, FusionOptions(), &why).has_value());
  EXPECT_EQ(why, "heights too far apart");
}

TEST(SameLevel, FloorsWithDifferentStructureAreNotCandidates) {
  PointCloud other;  // a long corridor, nothing like the building
  wall(other, 0, 0, 60, 0);
  wall(other, 0, 3, 60, 3);
  const Pose4 near(0.0, Eigen::Vector3d(0, 0, 1.0));
  EXPECT_FALSE(same_level_candidate(0, building(), 1, other, near).has_value());
}
