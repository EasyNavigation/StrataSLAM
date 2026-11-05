#include <gtest/gtest.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/factorTesting.h>

#include <filesystem>
#include <fstream>
#include <random>

#include "strataslam/fusion/factors4.hpp"
#include "strataslam/fusion/fusion.hpp"
#include "strataslam/fusion/registration4.hpp"

using namespace strataslam;
using Eigen::Vector3d;
using gtsam::Symbol;

namespace {
double pose4_err(const Pose4& a, const Pose4& b) {
  return std::max(std::abs(Pose4::WrapAngle(a.theta() - b.theta())), (a.translation() - b.translation()).norm());
}
}  // namespace

// ---------------- factors ----------------

TEST(FusionFactors, ZeroErrorAtExactMeasurementAndCorrectJacobians) {
  const Pose4 Xa(0.4, Vector3d(1, 2, 3)), Xb(-1.9, Vector3d(-2, 0.5, 6.1));
  gtsam::Values v;
  v.insert(Symbol('f', 1), Xa);
  v.insert(Symbol('f', 2), Xb);
  const auto n4 = gtsam::noiseModel::Isotropic::Sigma(4, 0.05);
  const auto n1 = gtsam::noiseModel::Isotropic::Sigma(1, 0.05);

  TransitionFactor tf_exact(Symbol('f', 1), Symbol('f', 2), Xa.between(Xb), n4);
  EXPECT_LT(tf_exact.unwhitenedError(v).norm(), 1e-12);
  TransitionFactor tf(Symbol('f', 1), Symbol('f', 2), Pose4(0.9, Vector3d(0.3, -1, 2.5)), n4);
  EXPECT_TRUE(gtsam::internal::testFactorJacobians("TransitionFactor", tf, v, 1e-5, 1e-6));  // exact even at non-zero error
  CrossFloorMatchFactor cf(Symbol('f', 1), Symbol('f', 2), Pose4(-0.3, Vector3d(1, 1, 1)), n4);
  EXPECT_TRUE(gtsam::internal::testFactorJacobians("CrossFloorMatchFactor", cf, v, 1e-5, 1e-6));

  const double dz = Xb.translation().z() - Xa.translation().z();
  BaroHeightFactor bf_exact(Symbol('f', 1), Symbol('f', 2), dz, n1);
  FloorHeightPrior hp_exact(Symbol('f', 1), Symbol('f', 2), dz, n1);
  EXPECT_NEAR(bf_exact.unwhitenedError(v)(0), 0.0, 1e-12);
  EXPECT_NEAR(hp_exact.unwhitenedError(v)(0), 0.0, 1e-12);
  BaroHeightFactor bf(Symbol('f', 1), Symbol('f', 2), dz - 0.7, n1);
  FloorHeightPrior hp(Symbol('f', 1), Symbol('f', 2), 2.2, n1);
  EXPECT_TRUE(gtsam::internal::testFactorJacobians("BaroHeightFactor", bf, v, 1e-5, 1e-6));
  EXPECT_TRUE(gtsam::internal::testFactorJacobians("FloorHeightPrior", hp, v, 1e-5, 1e-6));

  // Anchors: the same physical points expressed in each floor frame, with and without z.
  const std::vector<Vector3d> world{{1, 2, 0.5}, {4, -1, 2.2}, {-2, 3, 1.1}, {0.3, 0.7, 1.9}};
  std::vector<Vector3d> pa, pb;
  for (const auto& w : world) {
    pa.push_back(Xa.inverse().act(w));
    pb.push_back(Xb.inverse().act(w));
  }
  for (bool use_z : {true, false}) {
    AnchorFactor exact(Symbol('f', 1), Symbol('f', 2), pa, pb, use_z, 0.03);
    EXPECT_LT(exact.unwhitenedError(v).norm(), 1e-12);
    auto pb_off = pb;
    for (auto& p : pb_off) p += Vector3d(0.2, -0.1, 0.4);
    AnchorFactor off(Symbol('f', 1), Symbol('f', 2), pa, pb_off, use_z, 0.03);
    EXPECT_TRUE(gtsam::internal::testFactorJacobians("AnchorFactor", off, v, 1e-5, 1e-6));
    EXPECT_EQ(off.unwhitenedError(v).size(), (use_z ? 3 : 2) * 4);
  }
}

// ---------------- registration ----------------

TEST(Registration4, ClosedFormRecoversAPose4FromFourNoiseFreePoints) {
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> u(-5, 5);
  const Pose4 X(-2.1, Vector3d(3, -4, 1.5));
  std::vector<PointPair> pairs;
  for (int i = 0; i < 4; ++i) {
    const Vector3d p(u(rng), u(rng), u(rng));
    pairs.push_back({p, X.act(p)});
  }
  EXPECT_LT(pose4_err(align_pose4_closed_form(pairs), X), 1e-9);
  // Planar variant: z is not compared (the translation along z is the mean height difference).
  const Pose4 R = align_pose4_closed_form(pairs, nullptr, false);
  EXPECT_LT(std::abs(Pose4::WrapAngle(R.theta() - X.theta())), 1e-9);
  EXPECT_LT((R.translation() - X.translation()).norm(), 1e-9);
}

TEST(Registration4, GncRecoversTheTransformWithFiftyPercentOutliers) {
  for (unsigned seed = 1; seed <= 20; ++seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-5, 5);
    std::normal_distribution<double> n(0, 0.01);
    const Pose4 X(std::uniform_real_distribution<double>(-3, 3)(rng), Vector3d(u(rng), u(rng), u(rng)));
    std::vector<PointPair> pairs;
    for (int i = 0; i < 40; ++i) {
      const Vector3d p(u(rng), u(rng), u(rng));
      Vector3d q = X.act(p) + Vector3d(n(rng), n(rng), n(rng));
      if (i % 2 == 0) q = Vector3d(u(rng), u(rng), u(rng)) * 2;  // 50 % gross outliers
      pairs.push_back({p, q});
    }
    const auto r = align_pose4_gnc(pairs, 0.05);
    ASSERT_TRUE(r.converged) << seed;
    EXPECT_LT(std::abs(Pose4::WrapAngle(r.transform.theta() - X.theta())), 0.01) << seed;
    EXPECT_LT((r.transform.translation() - X.translation()).norm(), 0.03) << seed;
    EXPECT_GE(r.n_inliers, 18u);
    for (size_t i = 0; i < pairs.size(); i += 2) {
      EXPECT_FALSE(r.inlier[i]) << seed << " " << i;
    }
  }
}

// ---------------- the fusion itself ----------------

namespace {

/// Three floors with known frame placements and transition records generated from them.
struct Synthetic {
  MapData map;
  std::map<FloorId, Pose4> truth;
  Pose4 base_in_floor[3][2];  // keyframe base poses in each floor frame (kf index 0, 1)

  explicit Synthetic(double yaw1 = 0.5, double yaw2 = -0.4) {
    truth[0] = Pose4();
    truth[1] = Pose4(yaw1, Vector3d(1, 2, 3));
    truth[2] = truth[1] * Pose4(yaw2, Vector3d(0.5, -1, 3));
    for (int f = 0; f < 3; ++f) {
      FloorData fd;
      fd.metadata.id = f;
      fd.metadata.altitude_m = 3.0 * f;
      for (int k = 0; k < 2; ++k) {
        base_in_floor[f][k] = Pose4(0.3 * k, Vector3d(2.0 * k, 0.5 * k, 1.2));
        KeyframeRecord kr;
        kr.id = static_cast<KeyframeId>(10 * f + k);
        kr.t = base_in_floor[f][k].translation();
        const double th = base_in_floor[f][k].theta();
        kr.q = Eigen::Quaterniond(Eigen::AngleAxisd(th, Vector3d::UnitZ()));
        fd.keyframes.push_back(kr);
      }
      map.floors.push_back(fd);
    }
    add_transition(0, 1, TransitionType::Stairs, false);
    add_transition(1, 2, TransitionType::Elevator, true);
  }

  Pose4 map_pose(int f, int k) const { return truth.at(f) * base_in_floor[f][k]; }

  void add_transition(int a, int b, TransitionType type, bool baro) {
    TransitionRecord t;
    t.type = type;
    t.from = a;
    t.to = b;
    t.kf_before = 10 * a + 1;
    t.kf_after = 10 * b;
    const Pose4 rel = map_pose(a, 1).between(map_pose(b, 0));
    t.relative = std::array<double, 4>{rel.theta(), rel.translation().x(), rel.translation().y(), rel.translation().z()};
    t.continuous_tracking = true;
    if (baro) t.delta_z = truth.at(b).translation().z() - truth.at(a).translation().z() + 1.2 - 1.2 + (base_in_floor[a][1].translation().z() - base_in_floor[b][0].translation().z());
    t.covariance = Eigen::Matrix4d::Identity() * 1e-4;
    map.transitions.push_back(t);
  }
};

}  // namespace

TEST(Fusion, AllThreeFormulationsRecoverTheFloorTransformsFromTransitions) {
  const Synthetic s;
  for (Formulation f : {Formulation::Lie4, Formulation::Se3Soft, Formulation::Euler}) {
    FusionOptions opt;
    opt.formulation = f;
    opt.floor_height_prior = 3.0;
    const auto r = fuse(s.map, opt);
    ASSERT_TRUE(r.ok()) << to_string(f) << ": " << r.error().message;
    ASSERT_TRUE(r.value().finite());
    EXPECT_TRUE(r.value().converged);
    for (const auto& [floor, X] : s.truth) {
      EXPECT_LT(pose4_err(r.value().floor_to_map.at(floor), X), 2e-3) << to_string(f) << " floor " << floor;
    }
    EXPECT_EQ(r.value().covariance.size(), 3u);
    // The ablation compares formulations by the time of their own part: graph and optimization, a part of the whole
    // (audit AUD-053: the published time was the whole of fuse(), dominated by the stacking common to all three).
    EXPECT_GT(r.value().optimize_seconds, 0.0);
    EXPECT_LE(r.value().optimize_seconds, r.value().seconds);
  }
}

TEST(Fusion, ElevatorHeightComesFromTheBarometerFactor) {
  const Synthetic s;
  FusionOptions opt;
  const auto r = fuse(s.map, opt);
  ASSERT_TRUE(r.ok());
  bool saw_baro = false, saw_transition = false;
  for (const auto& c : r.value().constraints) {
    saw_baro = saw_baro || c.kind == "baro_height";
    saw_transition = saw_transition || c.kind == "transition";
    EXPECT_TRUE(c.accepted);
  }
  EXPECT_TRUE(saw_baro);
  EXPECT_TRUE(saw_transition);
  EXPECT_NEAR(r.value().floor_to_map.at(2).translation().z(), s.truth.at(2).translation().z(), 2e-3);
}

TEST(Fusion, GncRejectsAFalseFiveMeterConstraintAmongThreeCorrectOnes) {
  // Two floors, no transition record: four anchor sets relate them; one is shifted by 5 m.
  Synthetic s;
  s.map.transitions.clear();
  s.map.floors.resize(2);
  const Pose4 X1 = s.truth.at(1);
  FusionOptions opt;
  opt.floor_height_prior = 3.0;
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(-6, 6);
  for (int i = 0; i < 4; ++i) {
    AnchorSet a;
    a.name = "door" + std::to_string(i);
    for (int k = 0; k < 4; ++k) {
      const Vector3d w(u(rng), u(rng), 1.0 + k * 0.4);
      Vector3d p0 = w, p1 = X1.inverse().act(w + Vector3d(0, 0, 3.0));  // the structure repeats 3 m higher
      if (i == 3) p1 += Vector3d(5.0, 0.0, 0.0);                          // the false constraint
      a.points[0].push_back(p0);
      a.points[1].push_back(p1);
    }
    opt.anchors.push_back(a);
  }
  const auto r = fuse(s.map, opt);
  ASSERT_TRUE(r.ok()) << r.error().message;
  // The 5 m shift moves all four of the set's points consistently, so only the GNC stage over the
  // constraints (not the per-set robust prefilter) can discard it.
  EXPECT_LT(std::abs(Pose4::WrapAngle(r.value().floor_to_map.at(1).theta() - X1.theta())), 1e-6);
  EXPECT_NEAR(r.value().floor_to_map.at(1).translation().x(), X1.translation().x(), 1e-3);
  EXPECT_NEAR(r.value().floor_to_map.at(1).translation().y(), X1.translation().y(), 1e-3);
  EXPECT_NEAR(r.value().floor_to_map.at(1).translation().z(), 3.0, 0.05);  // from the height prior
  int rejected = 0;
  for (const auto& c : r.value().constraints) rejected += !c.accepted;
  EXPECT_EQ(rejected, 1);
}

TEST(Fusion, AFloorWithoutConstraintsIsReportedNotInvented) {
  Synthetic s;
  s.map.transitions.clear();
  const auto r = fuse(s.map, FusionOptions{});
  ASSERT_TRUE(r.ok());
  int unconnected = 0;
  for (const auto& w : r.value().warnings) unconnected += w.find("not connected") != std::string::npos;
  EXPECT_EQ(unconnected, 2);
}

TEST(Fusion, ReportAndOutputsAreWritten) {
  const Synthetic s;
  FusionOptions opt;
  opt.floor_height_prior = 3.0;
  const auto r = fuse(s.map, opt);
  ASSERT_TRUE(r.ok());
  MapData m = s.map;
  PointCloud pc;
  pc.points = {Eigen::Vector3f(1, 2, 3), Eigen::Vector3f(0, 0, 1)};
  for (auto& f : m.floors) f.floor_cloud = pc;
  const std::string dir = ::testing::TempDir() + "/strataslam_fused";
  ASSERT_TRUE(write_fusion_outputs(dir, m, r.value()).ok());
  for (const char* f : {"floor_transforms.yaml", "fusion_report.yaml", "global_cloud.pcd"}) EXPECT_TRUE(std::filesystem::exists(dir + "/" + f)) << f;
  const auto cloud = read_pcd(dir + "/global_cloud.pcd");
  ASSERT_TRUE(cloud.ok());
  EXPECT_EQ(cloud.value().points.size(), 6u);
  std::ifstream in(dir + "/fusion_report.yaml");
  const std::string text((std::istreambuf_iterator<char>(in)), {});
  EXPECT_NE(text.find("constraints:"), std::string::npos);
  EXPECT_NE(text.find("sigma:"), std::string::npos);
}

TEST(Fusion, AnchorsLoadFromYaml) {
  const std::string path = ::testing::TempDir() + "/anchors.yaml";
  std::ofstream(path) << "anchors:\n  - name: door\n    use_z: false\n    sigma: 0.02\n    points:\n      0: [[1, 2, 0], [3, 4, 0]]\n      1: [[0.5, 1, 0], [2.5, 3, 0]]\n";
  const auto a = load_anchors(path);
  ASSERT_TRUE(a.ok()) << a.error().message;
  ASSERT_EQ(a.value().size(), 1u);
  EXPECT_EQ(a.value()[0].name, "door");
  EXPECT_NEAR(a.value()[0].sigma, 0.02, 1e-12);
  EXPECT_EQ(a.value()[0].points.at(1).size(), 2u);
  EXPECT_FALSE(load_anchors(path + ".missing").ok());
}

TEST(Fusion, FormulationNamesRoundTrip) {
  for (Formulation f : {Formulation::Lie4, Formulation::Se3Soft, Formulation::Euler}) {
    Formulation back;
    ASSERT_TRUE(parse_formulation(to_string(f), &back));
    EXPECT_EQ(back, f);
  }
  Formulation x;
  EXPECT_FALSE(parse_formulation("nope", &x));
}

// ---------------- same level ----------------

namespace {

void wall_pts(PointCloud& c, double x0, double y0, double x1, double y1) {
  const double len = std::hypot(x1 - x0, y1 - y0);
  for (double s = 0; s <= len; s += 0.05)
    for (double z = 0.5; z <= 2.5; z += 0.5) c.points.emplace_back(static_cast<float>(x0 + (x1 - x0) * s / len), static_cast<float>(y0 + (y1 - y0) * s / len), static_cast<float>(z));
}

PointCloud asymmetric_building() {
  PointCloud c;
  wall_pts(c, 0, 0, 30, 0);
  wall_pts(c, 30, 0, 30, 20);
  wall_pts(c, 30, 20, 0, 20);
  wall_pts(c, 0, 20, 0, 0);
  wall_pts(c, 10, 0, 10, 12);
  wall_pts(c, 10, 12, 22, 12);
  return c;
}

/// Two floors joined by one ramp that the odometry measured as `dz` m high; floor 1 sees the same structure.
MapData two_floors(double dz) {
  MapData map;
  const Pose4 rel(0.05, Vector3d(2.0, -1.0, dz));
  const PointCloud a = asymmetric_building();
  PointCloud b;  // floor a's structure as seen from floor b's frame
  const Pose4 inv = rel.inverse();
  const double cs = std::cos(inv.theta()), sn = std::sin(inv.theta());
  for (const auto& p : a.points)
    b.points.emplace_back(static_cast<float>(cs * p.x() - sn * p.y() + inv.translation().x()), static_cast<float>(sn * p.x() + cs * p.y() + inv.translation().y()), p.z());
  for (int f = 0; f < 2; ++f) {
    FloorData fd;
    fd.metadata.id = f;
    KeyframeRecord kr;
    kr.id = static_cast<KeyframeId>(f);
    kr.t = Vector3d::Zero();
    kr.q = Eigen::Quaterniond::Identity();
    fd.keyframes.push_back(kr);
    fd.floor_cloud = f == 0 ? a : b;
    map.floors.push_back(fd);
  }
  TransitionRecord t;
  t.type = TransitionType::Ramp;
  t.from = 0;
  t.to = 1;
  t.kf_before = 0;
  t.kf_after = 1;
  t.relative = std::array<double, 4>{rel.theta(), rel.translation().x(), rel.translation().y(), rel.translation().z()};
  t.continuous_tracking = true;
  t.covariance = Eigen::Matrix4d::Identity() * 1e-4;
  map.transitions.push_back(t);
  return map;
}

}  // namespace

TEST(Fusion, ADeclaredSameLevelBringsTheTwoFloorsToTheSameHeight) {
  const MapData map = two_floors(1.7);  // the odometry drifted: the ramp reads 1.7 m, both floors are one level
  FusionOptions opt;
  opt.floor_height_prior = std::nullopt;
  opt.auto_stacking = false;
  const auto without = fuse(map, opt);
  ASSERT_TRUE(without.ok());
  EXPECT_NEAR(without.value().floor_to_map.at(1).translation().z(), 1.7, 0.05);
  opt.same_level.push_back({0, 1, 0.05});
  const auto with = fuse(map, opt);
  ASSERT_TRUE(with.ok());
  EXPECT_LT(std::abs(with.value().floor_to_map.at(1).translation().z()), 1.0);
  bool listed = false;
  for (const auto& c : with.value().constraints) listed = listed || c.label == "same level 0,1";
  EXPECT_TRUE(listed);
}

TEST(Fusion, FloorsWithCompatibleStructureAndCloseHeightsAreListedAsCandidates) {
  FusionOptions opt;
  opt.floor_height_prior = std::nullopt;
  const auto close = fuse(two_floors(1.7), opt);
  ASSERT_TRUE(close.ok());
  ASSERT_EQ(close.value().same_level_candidates.size(), 1u);
  EXPECT_EQ(close.value().same_level_candidates[0].a, 0);
  EXPECT_EQ(close.value().same_level_candidates[0].b, 1);
  EXPECT_NEAR(close.value().same_level_candidates[0].dz, 1.7, 0.05);
  EXPECT_GT(close.value().same_level_candidates[0].score, 0.9);
  EXPECT_NE(close.value().report_yaml().find("same_level_candidates"), std::string::npos);
  const auto far = fuse(two_floors(3.0), opt);  // a whole floor apart: not a candidate
  ASSERT_TRUE(far.ok());
  EXPECT_TRUE(far.value().same_level_candidates.empty());
}

TEST(Fusion, ADeclaredPairIsNotListedAgainAsACandidate) {
  FusionOptions opt;
  opt.floor_height_prior = std::nullopt;
  opt.same_level.push_back({0, 1, 0.05});
  const auto r = fuse(two_floors(1.7), opt);
  ASSERT_TRUE(r.ok());
  EXPECT_TRUE(r.value().same_level_candidates.empty());
}
