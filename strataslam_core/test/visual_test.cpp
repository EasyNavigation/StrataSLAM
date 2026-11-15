#include <gtest/gtest.h>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <random>

#include "strataslam/frontend/depth_utils.hpp"
#include "strataslam/frontend/feature_extractor.hpp"
#include "strataslam/frontend/visual_tracker.hpp"
#include "strataslam/pipeline/slam_system.hpp"
#include "strataslam/place_recognition/bow.hpp"

using namespace strataslam;
using Eigen::Vector3d;

namespace {

CameraIntrinsics intrinsics() {
  CameraIntrinsics K;
  K.width = 320;
  K.height = 240;
  K.fx = K.fy = 228.5;
  K.cx = 159.5;
  K.cy = 119.5;
  return K;
}

DepthImage flat_depth(float z) {
  DepthImage d;
  d.width = 320;
  d.height = 240;
  d.meters.assign(320 * 240, z);
  return d;
}

}  // namespace

// ---------------- depth back-projection ----------------

TEST(Backprojection, RangeLimitsAreEnforced) {
  const BackprojectionConfig cfg;  // [0.3, 6.0]
  EXPECT_TRUE(sample_depth(flat_depth(2.0f), 100, 100, cfg).has_value());
  EXPECT_FALSE(sample_depth(flat_depth(0.2f), 100, 100, cfg).has_value());
  EXPECT_FALSE(sample_depth(flat_depth(6.5f), 100, 100, cfg).has_value());
  EXPECT_FALSE(sample_depth(flat_depth(0.0f), 100, 100, cfg).has_value());
  EXPECT_FALSE(sample_depth(flat_depth(2.0f), -3, 100, cfg).has_value());
  EXPECT_FALSE(sample_depth(flat_depth(2.0f), 100, 500, cfg).has_value());
}

TEST(Backprojection, DiscontinuitiesAreDiscarded) {
  BackprojectionConfig cfg;
  DepthImage d = flat_depth(2.0f);
  // A step edge at u = 150: left of it 2.0 m, right 3.0 m (50 % jump, far above 5 %).
  for (int v = 0; v < 240; ++v)
    for (int u = 150; u < 320; ++u) d.meters[v * 320 + u] = 3.0f;
  EXPECT_FALSE(sample_depth(d, 149, 100, cfg).has_value());
  EXPECT_FALSE(sample_depth(d, 150, 100, cfg).has_value());
  EXPECT_TRUE(sample_depth(d, 100, 100, cfg).has_value());
  EXPECT_TRUE(sample_depth(d, 250, 100, cfg).has_value());
  // A 3 % step is within tolerance.
  DepthImage e = flat_depth(2.0f);
  for (int v = 0; v < 240; ++v)
    for (int u = 150; u < 320; ++u) e.meters[v * 320 + u] = 2.06f;
  EXPECT_TRUE(sample_depth(e, 149, 100, cfg).has_value());
}

TEST(Backprojection, ProducesTheCorrectCameraFramePoint) {
  const auto K = intrinsics();
  const auto p = backproject(flat_depth(4.0f), K, 259.5, 119.5, BackprojectionConfig{});
  ASSERT_TRUE(p);
  EXPECT_NEAR(p->x(), 100.0 / 228.5 * 4.0, 1e-9);
  EXPECT_NEAR(p->y(), 0.0, 1e-9);
  EXPECT_NEAR(p->z(), 4.0, 1e-9);
  const auto cloud = depth_to_cloud(flat_depth(4.0f), K, 8, BackprojectionConfig{});
  EXPECT_EQ(cloud.size(), 40u * 30u);
}

// ---------------- PnP on synthetic correspondences ----------------

TEST(Pnp, RecoversPoseWithOutliersAndPixelNoise) {
  const auto K = intrinsics();
  double worst_t = 0, worst_r = 0, sum_t2 = 0, sum_r2 = 0;
  const int n_seeds = 50;
  for (int seed = 1; seed <= n_seeds; ++seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1, 1);
    std::normal_distribution<double> n(0, 1.0);  // 1 px noise
    const gtsam::Pose3 T_cam_world(gtsam::Rot3::RzRyRx(0.1 * u(rng), 0.3 * u(rng), 0.3 * u(rng)),
                                   Vector3d(u(rng), u(rng), 0.5 * u(rng)));
    std::vector<PnPCorrespondence> corr;
    while (corr.size() < 500) {  // the simulator emits up to 500 keypoints per frame
      // A landmark in front of the camera, 2-6 m away, inside the field of view.
      const Vector3d pc(u(rng) * 2.0, u(rng) * 1.4, 2.0 + 4.0 * (0.5 + 0.5 * u(rng)));
      const double px = K.fx * pc.x() / pc.z() + K.cx, py = K.fy * pc.y() / pc.z() + K.cy;
      if (px < 5 || px > 315 || py < 5 || py > 235) continue;
      PnPCorrespondence c;
      c.point = T_cam_world.transformTo(pc);
      c.pixel = Eigen::Vector2d(px + n(rng), py + n(rng));
      if (corr.size() % 5 == 0)
        c.pixel = Eigen::Vector2d(5 + 310 * (0.5 + 0.5 * u(rng)),
                                  5 + 230 * (0.5 + 0.5 * u(rng)));  // 20 % outliers
      corr.push_back(c);
    }
    const gtsam::Pose3 init =
        T_cam_world.compose(gtsam::Pose3(gtsam::Rot3::Yaw(0.05), Vector3d(0.2, -0.1, 0.1)));
    const auto r = solve_pnp(corr, K, &init, 3.0, 200);
    ASSERT_TRUE(r.ok) << seed;
    const gtsam::Pose3 err = r.T_cam_obj.between(T_cam_world);
    if (std::getenv("STRATASLAM_PNP_DEBUG"))
      std::printf("seed %d t=%.4f r=%.4f inl=%zu\n", seed, err.translation().norm(),
                  gtsam::Rot3::Logmap(err.rotation()).norm() * 180 / M_PI, r.inliers.size());
    worst_t = std::max(worst_t, err.translation().norm());
    sum_t2 += err.translation().squaredNorm();
    sum_r2 += std::pow(gtsam::Rot3::Logmap(err.rotation()).norm() * 180 / M_PI, 2);
    worst_r = std::max(worst_r, gtsam::Rot3::Logmap(err.rotation()).norm() * 180 / M_PI);
    EXPECT_GE(r.inliers.size(), 350u);
  }
  // RMS over 50 random geometries/noise draws: translation below 1 cm and rotation below 0.2
  // degrees (ADR-0009), with hard bounds on every single draw (2 cm, 0.5 degrees).
  EXPECT_LT(std::sqrt(sum_t2 / n_seeds), 0.01);
  EXPECT_LT(worst_t, 0.02) << "worst translation error " << worst_t;
  EXPECT_LT(std::sqrt(sum_r2 / n_seeds), 0.2);
  EXPECT_LT(worst_r, 0.5) << "worst rotation error " << worst_r;
}

TEST(Pnp, TooFewCorrespondencesFailCleanly) {
  const auto r = solve_pnp({}, intrinsics(), nullptr, 3.0, 100);
  EXPECT_FALSE(r.ok);
}

TEST(VisualLoop, CandidateBelowThirtyPnpInliersIsRejectedWithTheDefaultTuning) {
  // Spec 9.7: a visual loop needs PnP with at least 30 inliers. Perfect correspondences, so PnP
  // itself always succeeds: only the threshold decides.
  const auto K = intrinsics();
  const SystemTuning tuning;
  EXPECT_EQ(tuning.visual_loop_min_inliers, 30);
  const gtsam::Pose3 T_cam_world(gtsam::Rot3::Yaw(0.2), Vector3d(0.3, -0.1, 0.2));
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(-1, 1);
  auto make = [&](size_t n) {
    std::vector<PnPCorrespondence> corr;
    while (corr.size() < n) {
      const Vector3d pc(u(rng) * 2.0, u(rng) * 1.4, 3.0 + u(rng));
      const double px = K.fx * pc.x() / pc.z() + K.cx, py = K.fy * pc.y() / pc.z() + K.cy;
      if (px < 5 || px > 315 || py < 5 || py > 235) continue;
      corr.push_back({T_cam_world.transformTo(pc), Eigen::Vector2d(px, py)});
    }
    return corr;
  };
  const double px = tuning.visual.tracker.ransac_reprojection_px;
  for (size_t n : {10u, 20u, 29u}) {
    const auto corr = make(n);
    EXPECT_FALSE(verify_loop_pnp(corr, K, tuning.visual_loop_min_inliers, px, 300))
        << n << " inliers";
    EXPECT_TRUE(verify_loop_pnp(corr, K, static_cast<int>(n), px, 300))
        << n << " inliers, threshold " << n;
  }
  const auto r = verify_loop_pnp(make(30), K, tuning.visual_loop_min_inliers, px, 300);
  ASSERT_TRUE(r);
  EXPECT_LT(r->between(T_cam_world).translation().norm(), 1e-3);
  // 30 matches of which 10 are outliers: 20 inliers, below the threshold.
  auto mixed = make(20);
  for (int i = 0; i < 10; ++i)
    mixed.push_back({Vector3d(u(rng), u(rng), 3.0 + u(rng)),
                     Eigen::Vector2d(160 + 150 * u(rng), 120 + 110 * u(rng))});
  EXPECT_FALSE(verify_loop_pnp(mixed, K, tuning.visual_loop_min_inliers, px, 300));
}

// ---------------- ORB extractor ----------------

TEST(Orb, DepthMaskKeepsKeypointsWhereThereIsADepth) {
  // A textured image whose right half has no depth (beyond the sensor's range): keypoints only on
  // the left.
  GrayImage img;
  img.width = 320;
  img.height = 240;
  img.data.resize(320 * 240);
  std::mt19937 rng(5);
  for (auto& p : img.data) p = static_cast<uint8_t>(rng() % 256);
  DepthImage depth;
  depth.width = 320;
  depth.height = 240;
  depth.meters.assign(320 * 240, 0.f);
  for (int v = 0; v < 240; ++v)
    for (int u = 0; u < 160; ++u) depth.meters[static_cast<size_t>(v * 320 + u)] = 2.0f;
  const auto mask = valid_depth_mask(depth, 0.3, 6.0);
  auto ex = make_orb_extractor(500);
  const auto all = ex->extract(img), masked = ex->extract(img, &mask);
  ASSERT_GT(masked.size(), 100u);
  for (const auto& k : masked) EXPECT_LT(k.u, 160.f);
  EXPECT_TRUE(
      std::any_of(all.begin(), all.end(), [](const RgbKeypoint& k) { return k.u > 160.f; }));
  EXPECT_EQ(valid_depth_mask(depth, 2.5, 6.0)[0], 0);  // 2 m is outside [2.5, 6]
}

TEST(Orb, ExtractsBinaryDescriptorsFromATexturedImage) {
  GrayImage img;
  img.width = 320;
  img.height = 240;
  img.data.resize(320 * 240);
  std::mt19937 rng(3);
  for (auto& p : img.data) p = static_cast<uint8_t>(rng() % 256);
  // Blocky texture so that FAST corners exist: 4x4 blocks of random grey levels.
  for (int by = 0; by < 60; ++by)
    for (int bx = 0; bx < 80; ++bx) {
      const uint8_t g = static_cast<uint8_t>(rng() % 256);
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) img.data[(by * 4 + y) * 320 + bx * 4 + x] = g;
    }
  auto ex = make_orb_extractor(300);
  EXPECT_EQ(ex->name(), "orb");
  const auto kps = ex->extract(img);
  EXPECT_GT(kps.size(), 50u);
  EXPECT_LE(kps.size(), 300u);
  for (size_t i = 1; i < kps.size(); ++i) EXPECT_GE(kps[i - 1].response, kps[i].response);
  for (const auto& k : kps) {
    EXPECT_GE(k.u, 0.f);
    EXPECT_LT(k.u, 320.f);
  }
  // Same image, same output (deterministic).
  const auto again = ex->extract(img);
  ASSERT_EQ(kps.size(), again.size());
  EXPECT_EQ(kps[0].descriptor, again[0].descriptor);
}

// ---------------- Bag of words ----------------

namespace {
std::vector<Descriptor> random_descriptors(size_t n, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::vector<Descriptor> out(n);
  for (auto& d : out)
    for (size_t w = 0; w < 4; ++w) {
      const uint64_t v = rng();
      std::memcpy(d.data() + 8 * w, &v, 8);
    }
  return out;
}
std::vector<Descriptor> noisy(const std::vector<Descriptor>& in, int flips, unsigned seed) {
  std::mt19937 rng(seed);
  auto out = in;
  for (auto& d : out)
    for (int i = 0; i < flips; ++i) {
      const unsigned bit = rng() % 256;
      d[bit / 8] ^= static_cast<uint8_t>(1u << (bit % 8));
    }
  return out;
}
}  // namespace

TEST(Bow, VocabularyTrainsAndScoresSimilarPlacesHigher) {
  BowConfig cfg;
  cfg.branching = 10;
  cfg.levels = 4;
  const auto vocab = BowVocabulary::train(random_descriptors(60000, 1), cfg);
  EXPECT_GT(vocab.words(), 3000u);
  const auto place_a = random_descriptors(300, 10), place_b = random_descriptors(300, 11);
  const BowVector a = vocab.transform(place_a), a2 = vocab.transform(noisy(place_a, 6, 5)),
                  b = vocab.transform(place_b);
  float sum = 0;
  for (const auto& p : a) sum += p.second;
  EXPECT_NEAR(sum, 1.0f, 1e-5);
  EXPECT_NEAR(BowVocabulary::score(a, a), 1.0, 1e-5);
  EXPECT_GT(BowVocabulary::score(a, a2), 2.0 * BowVocabulary::score(a, b));
  EXPECT_GT(BowVocabulary::score(a, a2), 0.2);
}

TEST(Bow, DatabaseRanksTheRevisitAndHonoursTemporalExclusion) {
  BowConfig cfg;
  cfg.branching = 10;
  cfg.levels = 4;
  auto vocab = std::make_shared<const BowVocabulary>(
      BowVocabulary::train(random_descriptors(60000, 1), cfg));
  BowDatabase db(vocab);
  std::vector<std::vector<Descriptor>> places;
  for (int i = 0; i < 10; ++i) {
    places.push_back(random_descriptors(300, 100 + i));
    db.add(i, /*session=*/0, /*stamp=*/static_cast<Timestamp>(i) * 1000000000LL,
           vocab->transform(places.back()));
  }
  const BowVector q = vocab->transform(noisy(places[3], 6, 9));
  const Timestamp now = 30LL * 1000000000LL;
  const auto res = db.query(q, 0, now, 10.0, 0.05, 5);
  ASSERT_FALSE(res.empty());
  EXPECT_EQ(res.front().id, 3u);
  // A query only 1 s after entry 3 in the same session must not return it...
  const auto recent = db.query(q, 0, 4LL * 1000000000LL, 10.0, 0.05, 5);
  for (const auto& c : recent) EXPECT_NE(c.id, 3u);
  // ...but from another session it is always eligible.
  const auto other = db.query(q, 1, 4LL * 1000000000LL, 10.0, 0.05, 5);
  ASSERT_FALSE(other.empty());
  EXPECT_EQ(other.front().id, 3u);
}

TEST(Bow, VocabularySaveLoadRoundTripAndVersionCheck) {
  BowConfig cfg;
  cfg.branching = 6;
  cfg.levels = 2;
  const auto vocab = BowVocabulary::train(random_descriptors(5000, 2), cfg);
  const std::string path = ::testing::TempDir() + "/vocab.bin";
  ASSERT_TRUE(vocab.save(path).ok());
  const auto back = BowVocabulary::load(path);
  ASSERT_TRUE(back.ok());
  EXPECT_TRUE(back.value() == vocab);
  const auto d = random_descriptors(50, 7);
  EXPECT_EQ(vocab.transform(d), back.value().transform(d));
  // Corrupt the version.
  {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(4);
    const uint32_t v = 99;
    f.write(reinterpret_cast<const char*>(&v), 4);
  }
  const auto bad = BowVocabulary::load(path);
  ASSERT_FALSE(bad.ok());
  EXPECT_NE(bad.error().message.find("unsupported vocabulary version 99"), std::string::npos);
  EXPECT_FALSE(BowVocabulary::load(path + ".missing").ok());
}

TEST(Bow, TrainingIsDeterministic) {
  BowConfig cfg;
  cfg.branching = 6;
  cfg.levels = 2;
  const auto data = random_descriptors(3000, 4);
  EXPECT_TRUE(BowVocabulary::train(data, cfg) == BowVocabulary::train(data, cfg));
}

// ---------------- Tracker: what each keypoint was used for ----------------

TEST(Tracker, EachKeypointIsReportedAsInlierRejectedOrUnmatched) {
  const CameraIntrinsics K = intrinsics();
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> ux(-2.0, 2.0), uy(-1.5, 1.5), uz(3.0, 6.0);
  const size_t n = 100;
  std::vector<Vector3d> pts(n);
  for (auto& p : pts) p = Vector3d(ux(rng), uy(rng), uz(rng));
  const auto desc = random_descriptors(n + 10, 11);
  const auto observe = [&](const gtsam::Pose3& T_odom_cam, size_t i, double du) {
    const Vector3d c = T_odom_cam.transformTo(pts[i]);
    RgbKeypoint k;
    k.u = static_cast<float>(K.fx * c.x() / c.z() + K.cx + du);
    k.v = static_cast<float>(K.fy * c.y() / c.z() + K.cy);
    k.response = 1.f;
    k.descriptor = desc[i];
    return std::make_pair(k, std::optional<Vector3d>(c));
  };
  VisualObservation first;
  for (size_t i = 0; i < n; ++i) {
    auto [k, c] = observe(gtsam::Pose3(), i, 0.0);
    first.keypoints.push_back(k);
    first.p_cam.push_back(c);
  }
  VisualTracker tracker{VisualTrackerConfig()};
  EXPECT_TRUE(tracker.last_keypoint_use().empty());
  tracker.initialize(K, first, gtsam::Pose3());
  ASSERT_EQ(tracker.last_keypoint_use().size(), n);

  // Second frame: 80 points where they should be, 10 shifted 12 px (matched, but PnP outliers) and
  // 10 keypoints never seen before.
  const gtsam::Pose3 moved(gtsam::Rot3(), gtsam::Point3(0.05, 0.0, 0.02));
  VisualObservation second;
  for (size_t i = 0; i < 90; ++i) {
    auto [k, c] = observe(moved, i, i < 80 ? 0.0 : 12.0);
    second.keypoints.push_back(k);
    second.p_cam.push_back(c);
  }
  for (size_t j = 0; j < 10; ++j) {
    auto [k, c] = observe(moved, 90 + j, 0.0);
    k.descriptor = desc[n + j];
    second.keypoints.push_back(k);
    second.p_cam.push_back(c);
  }
  const auto r = tracker.track(K, second, moved, false);
  ASSERT_TRUE(r.ok);
  const auto& use = tracker.last_keypoint_use();
  ASSERT_EQ(use.size(), second.keypoints.size());
  EXPECT_EQ(std::count(use.begin(), use.begin() + 80, KeypointUse::Inlier), 80);
  EXPECT_EQ(std::count(use.begin() + 80, use.begin() + 90, KeypointUse::Rejected), 10);
  EXPECT_EQ(std::count(use.begin() + 90, use.end(), KeypointUse::Unmatched), 10);
  EXPECT_EQ(r.inliers, 80);
}

// The same scene, the same motion and the same calibration error, seen by a VGA camera and by
// one with twice its focal length (a 720p OAK-D Lite): the error covers twice the pixels, and the
// pixel thresholds of the tracking grow with the focal length, so both keep every point. With the
// thresholds of the VGA camera 40 % of the 720p points fell outside RANSAC's 3 px.
TEST(Tracker, ALongerFocalLengthTracksAsWellAsTheReferenceOne) {
  for (const double f : {kReferenceFocalPx, 2 * kReferenceFocalPx}) {
    CameraIntrinsics K;
    K.width = static_cast<int>(1.2 * f);
    K.height = static_cast<int>(0.9 * f);
    K.fx = K.fy = f;
    K.cx = 0.6 * f;
    K.cy = 0.45 * f;
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> ux(-2.0, 2.0), uy(-1.5, 1.5), uz(3.0, 6.0);
    std::normal_distribution<double> error(0.0, f / kReferenceFocalPx);  // px
    const size_t n = 100;
    std::vector<Vector3d> pts(n);
    for (auto& p : pts) p = Vector3d(ux(rng), uy(rng), uz(rng));
    const auto desc = random_descriptors(n, 13);
    const auto observe = [&](const gtsam::Pose3& T_odom_cam, double sigma) {
      VisualObservation o;
      std::normal_distribution<double> e(0.0, sigma);
      for (size_t i = 0; i < n; ++i) {
        const Vector3d c = T_odom_cam.transformTo(pts[i]);
        RgbKeypoint k;
        k.u = static_cast<float>(K.fx * c.x() / c.z() + K.cx + e(rng));
        k.v = static_cast<float>(K.fy * c.y() / c.z() + K.cy + e(rng));
        k.response = 1.f;
        k.descriptor = desc[i];
        o.keypoints.push_back(k);
        o.p_cam.push_back(c);
      }
      return o;
    };
    VisualTracker tracker{VisualTrackerConfig()};
    tracker.initialize(K, observe(gtsam::Pose3(), 0.0), gtsam::Pose3());
    const gtsam::Pose3 moved(gtsam::Rot3::Yaw(0.01), gtsam::Point3(0.05, 0.0, 0.02));
    const auto r = tracker.track(K, observe(moved, error.stddev()), moved, false);
    ASSERT_TRUE(r.ok) << "focal " << f;
    EXPECT_GE(r.inliers, 95) << "focal " << f;
  }
}

TEST(Tracker, SystemReportsKeypointUseOnlyWithCameraOdometry) {
  Config lidar;
  lidar.sensors.lidar.enabled = true;
  lidar.sensors.camera.enabled = true;
  lidar.sensors.camera.depth_min = 0.3;
  lidar.sensors.camera.depth_max = 6.0;
  lidar.fusion.odometry_from = OdometryFrom::Lidar;
  const SlamSystem with_lidar(lidar, Extrinsics());
  EXPECT_FALSE(
      with_lidar.keypoint_use().has_value());  // the camera keypoints only serve loop closures
  Config camera = lidar;
  camera.sensors.lidar.enabled = false;
  camera.fusion.odometry_from = OdometryFrom::Camera;
  const SlamSystem with_camera(camera, Extrinsics());
  ASSERT_TRUE(with_camera.keypoint_use().has_value());
  EXPECT_TRUE(with_camera.keypoint_use()->empty());  // no frame yet
  EXPECT_FALSE(with_camera.baro_height().has_value());
  EXPECT_EQ(with_camera.floor_altitude(0), 0.0);
  EXPECT_FALSE(with_camera.floor_altitude(1).has_value());
  EXPECT_FALSE(with_camera.floor_altitude(-1).has_value());
}

// ---------------- The RGB-D pipeline, end to end ----------------

namespace {

/// A closed room as an RGB-D camera sees it at the feature level, as the simulator models one:
/// landmarks with fixed descriptors on its walls, floor and ceiling, each observed with pixel noise
/// and a few flipped descriptor bits, strongest first up to a cap; the depth image is cast against
/// the room's box (convex, so every landmark in front of the camera is visible).
class Room {
 public:
  Room() {
    std::mt19937 rng(21);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (int axis = 0; axis < 3; ++axis)
      for (const double side : {lo_[axis], hi_[axis]}) {
        const int a = (axis + 1) % 3, b = (axis + 2) % 3;
        const double area = (hi_[a] - lo_[a]) * (hi_[b] - lo_[b]);
        for (int i = 0; i < static_cast<int>(kPerSquareMetre * area); ++i) {
          Vector3d p;
          p[axis] = side;
          p[a] = lo_[a] + unit(rng) * (hi_[a] - lo_[a]);
          p[b] = lo_[b] + unit(rng) * (hi_[b] - lo_[b]);
          points_.push_back(p);
          strength_.push_back(static_cast<float>(unit(rng)));
        }
      }
    descriptors_ = random_descriptors(points_.size(), 22);
  }

  RgbdFrame frame(const gtsam::Pose3& T_room_cam, Timestamp stamp, std::mt19937& rng) const {
    RgbdFrame f;
    f.stamp = stamp;
    f.K = intrinsics();
    const CameraIntrinsics& K = f.K;
    f.depth.width = K.width;
    f.depth.height = K.height;
    f.depth.meters.assign(static_cast<size_t>(K.width) * K.height, 0.f);
    const Eigen::Matrix3d R = T_room_cam.rotation().matrix();
    const Vector3d& o = T_room_cam.translation();
    for (int v = 0; v < K.height; ++v)
      for (int u = 0; u < K.width; ++u) {
        // The ray (x, y, 1) in the camera: its parameter at the wall is the depth itself.
        const Vector3d d = R * Vector3d((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0);
        double z = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 3; ++i) {
          if (d[i] > 1e-12) z = std::min(z, (hi_[i] - o[i]) / d[i]);
          if (d[i] < -1e-12) z = std::min(z, (lo_[i] - o[i]) / d[i]);
        }
        f.depth.meters[static_cast<size_t>(v) * K.width + u] = static_cast<float>(z);
      }
    std::vector<std::pair<float, size_t>> seen;
    for (size_t i = 0; i < points_.size(); ++i) {
      const Vector3d c = T_room_cam.transformTo(points_[i]);
      if (c.z() < 0.3) continue;
      const double u = K.fx * c.x() / c.z() + K.cx, v = K.fy * c.y() / c.z() + K.cy;
      if (u < 4 || v < 4 || u > K.width - 5 || v > K.height - 5) continue;
      seen.emplace_back(strength_[i], i);
    }
    std::sort(seen.begin(), seen.end(), std::greater<>());
    if (seen.size() > 500) seen.resize(500);
    std::normal_distribution<double> pixel(0.0, 0.5);
    for (const auto& [strength, i] : seen) {
      const Vector3d c = T_room_cam.transformTo(points_[i]);
      RgbKeypoint k;
      k.u = static_cast<float>(K.fx * c.x() / c.z() + K.cx + pixel(rng));
      k.v = static_cast<float>(K.fy * c.y() / c.z() + K.cy + pixel(rng));
      k.response = strength;
      k.descriptor = descriptors_[i];
      for (int flip = 0; flip < 6; ++flip) {
        const unsigned bit = rng() % 256;
        k.descriptor[bit / 8] ^= static_cast<uint8_t>(1u << (bit % 8));
      }
      f.keypoints.push_back(k);
    }
    return f;
  }

 private:
  static constexpr double kPerSquareMetre = 25.0;
  Vector3d lo_{0.0, 0.0, 0.0}, hi_{8.0, 6.0, 3.0};
  std::vector<Vector3d> points_;
  std::vector<float> strength_;
  std::vector<Descriptor> descriptors_;
};

/// A hand-held walk round the room at 1.3 m, 1.5 m from its walls: straight at 0.8 m/s, turning
/// left on the spot at each corner, and on past the start, where it began 25 s before. 10 Hz.
std::vector<gtsam::Pose3> walk_round_the_room() {
  std::vector<gtsam::Pose3> out;
  Eigen::Vector2d at(1.5, 1.5);
  double yaw = 0.0;
  const auto here = [&] {
    return gtsam::Pose3(gtsam::Rot3::Yaw(yaw), gtsam::Point3(at.x(), at.y(), 1.3));
  };
  const auto go_to = [&](const Eigen::Vector2d& to) {
    const Eigen::Vector2d from = at;
    const int n = static_cast<int>(std::round((to - from).norm() / 0.08));
    for (int i = 1; i <= n; ++i) {
      at = from + (to - from) * (static_cast<double>(i) / n);
      out.push_back(here());
    }
  };
  const auto turn_left = [&] {
    for (int i = 0; i < 20; ++i) {
      yaw += M_PI / 40;
      out.push_back(here());
    }
  };
  out.push_back(here());
  go_to({6.5, 1.5});
  turn_left();
  go_to({6.5, 4.5});
  turn_left();
  go_to({1.5, 4.5});
  turn_left();
  go_to({1.5, 1.5});
  turn_left();
  go_to({4.5, 1.5});
  return out;
}

Config rgbd_only() {
  Config c;
  c.sensors.camera.enabled = true;
  c.fusion.odometry_from = OdometryFrom::Camera;
  return c;
}

SystemTuning with_vocabulary() {
  SystemTuning t;
  BowConfig cfg;
  cfg.branching = 10;
  cfg.levels = 4;
  t.vocabulary = std::make_shared<const BowVocabulary>(
      BowVocabulary::train(random_descriptors(30000, 23), cfg));
  return t;
}

constexpr Timestamp kFrameNs = 100'000'000;  // 10 Hz

}  // namespace

// The camera alone tracks the walk round the room, and back at the start the place is recognized
// and a loop closed against the keyframes of 25 s before.
TEST(RgbdSystem, AWalkRoundARoomIsTrackedAndClosedByItsVisualLoop) {
  const Room room;
  const auto walk = walk_round_the_room();
  Extrinsics ext;
  ext.T_base_camera = Extrinsics::ForwardCamera();
  SlamSystem slam(rgbd_only(), ext, with_vocabulary());
  std::mt19937 rng(24);
  for (size_t i = 0; i < walk.size(); ++i) {
    slam.add_rgbd(
        room.frame(walk[i] * ext.T_base_camera, static_cast<Timestamp>(i) * kFrameNs, rng));
    ASSERT_NE(slam.tracking_quality().state, TrackingState::Lost) << "frame " << i;
  }
  EXPECT_GT(slam.keyframes().size(), 10u);
  ASSERT_FALSE(slam.loops().empty());
  for (const auto& l : slam.loops()) {
    EXPECT_EQ(l.from_floor, 0);
    EXPECT_EQ(l.to_floor, 0);
  }
  EXPECT_EQ(slam.current_floor(), 0);
  // The odometry starts at the first pose, so the walk is compared relative to it.
  const gtsam::Pose3 truth = walk.front().between(walk.back());
  const std::optional<gtsam::Pose3> odom = slam.current_pose_odom();
  if (!odom) FAIL() << "no odometry pose";
  EXPECT_LT((odom->translation() - truth.translation()).norm(), 0.1);
  const MapData map = slam.finalize();
  ASSERT_EQ(map.floors.size(), 1u);
}

// The camera goes blind for 1 s mid-way along the far wall (empty frames, as a covered lens gives):
// tracking is lost and the frontend starts a new session that knows nothing of the room. Its first
// keyframes are relocalized on floor 0 against the keyframes seen before the loss, where the room
// puts them, and back at the start a loop joins the two sessions.
TEST(RgbdSystem, ASessionAfterABlindSecondIsRelocalizedOnItsFloor) {
  const Room room;
  const auto walk = walk_round_the_room();
  Extrinsics ext;
  ext.T_base_camera = Extrinsics::ForwardCamera();
  Config cfg = rgbd_only();
  cfg.multifloor.tracking_loss_timeout = 0.5;  // s: the blind second outlasts it
  SlamSystem slam(cfg, ext, with_vocabulary());
  std::mt19937 rng(25);
  bool lost = false;
  for (size_t i = 0; i < walk.size(); ++i) {
    const Timestamp stamp = static_cast<Timestamp>(i) * kFrameNs;
    if (i >= walk.size() / 2 && i < walk.size() / 2 + 10) {
      RgbdFrame dark;
      dark.stamp = stamp;
      dark.K = intrinsics();
      dark.depth = flat_depth(0.f);
      slam.add_rgbd(dark);
    } else {
      slam.add_rgbd(room.frame(walk[i] * ext.T_base_camera, stamp, rng));
    }
    lost = lost || slam.tracking_quality().state == TrackingState::Lost;
  }
  EXPECT_TRUE(lost);
  EXPECT_EQ(slam.current_session(), 1u);
  const SlamSystem::FloorStatus status = slam.floor_status();
  EXPECT_FALSE(status.pending);
  EXPECT_EQ(status.num_floors, 1u);
  EXPECT_EQ(slam.current_floor(), 0);
  std::map<KeyframeId, SessionId> session;
  for (const auto& kf : slam.keyframes()) session[kf.id] = kf.session;
  size_t between_sessions = 0;
  for (const auto& l : slam.loops()) between_sessions += session.at(l.from) != session.at(l.to);
  EXPECT_GE(between_sessions, 1u);
  // Every keyframe of the new session joined floor 0, where the room puts it.
  size_t placed = 0;
  for (const auto& [stamp, T_floor_base] : slam.floor_path(0)) {
    const size_t i = static_cast<size_t>(stamp / kFrameNs);
    if (i < walk.size() / 2) continue;
    const gtsam::Pose3 truth = walk.front().between(walk[i]);
    EXPECT_LT((T_floor_base.translation() - truth.translation()).norm(), 0.15) << "frame " << i;
    ++placed;
  }
  EXPECT_GT(placed, 10u);
}

// The camera goes blind for the last turn, on the spot where the walk began, and the session that
// follows starts where the turn did, a quarter turn off. Back where the walk began the place is
// recognized: the relocalization, not the restarted odometry, places the session's keyframes, and
// they are where the room puts them.
TEST(RgbdSystem, ASessionAfterABlindTurnIsRelocalizedWhereTheWalkBegan) {
  const Room room;
  const auto walk = walk_round_the_room();
  Extrinsics ext;
  ext.T_base_camera = Extrinsics::ForwardCamera();
  Config cfg = rgbd_only();
  cfg.multifloor.tracking_loss_timeout = 0.5;  // s: the 2 s turn outlasts it
  SlamSystem slam(cfg, ext, with_vocabulary());
  std::mt19937 rng(25);
  bool lost = false;
  for (size_t i = 0; i < walk.size(); ++i) {
    const Timestamp stamp = static_cast<Timestamp>(i) * kFrameNs;
    const bool blind =
        i > walk.size() / 2 && (walk[i].translation() - walk.front().translation()).norm() < 1e-6;
    if (blind) {
      RgbdFrame dark;
      dark.stamp = stamp;
      dark.K = intrinsics();
      dark.depth = flat_depth(0.f);
      slam.add_rgbd(dark);
    } else {
      slam.add_rgbd(room.frame(walk[i] * ext.T_base_camera, stamp, rng));
    }
    lost = lost || slam.tracking_quality().state == TrackingState::Lost;
  }
  EXPECT_TRUE(lost);
  EXPECT_GT(slam.current_session(), 0u);
  const SlamSystem::FloorStatus status = slam.floor_status();
  EXPECT_FALSE(status.pending);
  EXPECT_EQ(status.num_floors, 1u);
  EXPECT_EQ(slam.current_floor(), 0);
  std::map<KeyframeId, SessionId> session;
  for (const auto& kf : slam.keyframes()) session[kf.id] = kf.session;
  size_t between_sessions = 0;
  for (const auto& l : slam.loops()) between_sessions += session.at(l.from) != session.at(l.to);
  EXPECT_GE(between_sessions, 1u);
  size_t placed = 0;
  for (const auto& [stamp, T_floor_base] : slam.floor_path(0)) {
    const size_t i = static_cast<size_t>(stamp / kFrameNs);
    if (i <= walk.size() / 2 || (walk[i].translation() - walk.front().translation()).norm() < 1e-6)
      continue;
    const gtsam::Pose3 truth = walk.front().between(walk[i]);
    EXPECT_LT((T_floor_base.translation() - truth.translation()).norm(), 0.15) << "frame " << i;
    ++placed;
  }
  EXPECT_GT(placed, 3u);
}
