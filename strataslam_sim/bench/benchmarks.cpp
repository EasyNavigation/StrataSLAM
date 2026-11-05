// Performance benchmarks (spec 12.8). Targets: ORB with 1000 features at 640x480 <= 33 ms per
// frame, LiDAR frontend with a 64-beam sensor <= 50 ms per scan, iSAM2 update <= 20 ms mean per
// keyframe.
#include <benchmark/benchmark.h>

#include <random>

#include "strataslam/backend/backend.hpp"
#include "strataslam/frontend/feature_extractor.hpp"
#include "strataslam/frontend/lidar_frontend.hpp"
#include "strataslam/frontend/visual_frontend.hpp"
#include "strataslam_sim/scenarios.hpp"

namespace {

/// Textured 640x480 image (random grey blocks) so that FAST finds plenty of corners.
strataslam::GrayImage textured_image() {
  strataslam::GrayImage img;
  img.width = 640;
  img.height = 480;
  img.data.resize(640 * 480);
  std::mt19937 rng(5);
  for (int by = 0; by < 120; ++by)
    for (int bx = 0; bx < 160; ++bx) {
      const uint8_t g = static_cast<uint8_t>(rng() % 256);
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) img.data[(by * 4 + y) * 640 + bx * 4 + x] = g;
    }
  return img;
}

void BM_OrbExtract_640x480_1000(benchmark::State& state) {
  const auto img = textured_image();
  auto orb = strataslam::make_orb_extractor(1000);
  // Warm-up outside the timed loop: OpenCV's first call in a process initialises its backends
  // (about 2 s here), which a single measured iteration would report as the cost of an extraction.
  size_t n = orb->extract(img).size();
  for (auto _ : state) n = orb->extract(img).size();
  state.counters["features"] = static_cast<double>(n);
}
BENCHMARK(BM_OrbExtract_640x480_1000)->Unit(benchmark::kMillisecond);

void BM_VisualTracking(benchmark::State& state) {
  auto spec = strataslam_sim::make_scenario("S01");
  strataslam_sim::Simulator sim(spec.building, spec.trajectory, spec.sim);
  std::vector<strataslam::RgbdFrame> frames;
  for (size_t i = 0; i < 120; ++i) frames.push_back(sim.rgbd(i));
  strataslam::VisualFrontendConfig cfg;
  for (auto _ : state) {
    state.PauseTiming();
    strataslam::VisualFrontend fe(cfg, spec.sim.extrinsics.T_base_camera);
    fe.reset(sim.gt_pose(frames.front().stamp));
    state.ResumeTiming();
    for (const auto& f : frames) {
      strataslam::SensorFrame sf;
      sf.stamp = f.stamp;
      sf.rgbd = &f;
      benchmark::DoNotOptimize(fe.process(sf));
    }
  }
  state.SetItemsProcessed(state.iterations() * frames.size());
  state.counters["time_per_frame"] = benchmark::Counter(
      static_cast<double>(frames.size()),
      benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_VisualTracking)->Unit(benchmark::kMillisecond)->Iterations(3);

void BM_LidarFrontend_64beams(benchmark::State& state) {
  auto spec = strataslam_sim::make_scenario("S02");
  spec.sim.lidar_beams = 64;
  spec.sim.lidar_columns = 360;
  strataslam_sim::Simulator sim(spec.building, spec.trajectory, spec.sim);
  std::vector<strataslam::LidarScan> scans;
  for (size_t i = 0; i < 60; ++i) scans.push_back(sim.lidar(i));
  strataslam::LidarFrontendConfig cfg;
  cfg.threads = 4;
  for (auto _ : state) {
    state.PauseTiming();
    strataslam::LidarFrontend fe(cfg, spec.sim.extrinsics.T_base_lidar);
    fe.reset(sim.gt_pose(scans.front().stamp));
    state.ResumeTiming();
    for (const auto& s : scans) {
      strataslam::SensorFrame sf;
      sf.stamp = s.stamp;
      sf.lidar = &s;
      benchmark::DoNotOptimize(fe.process(sf));
    }
  }
  state.SetItemsProcessed(state.iterations() * scans.size());
  state.counters["time_per_scan"] = benchmark::Counter(
      static_cast<double>(scans.size()),
      benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_LidarFrontend_64beams)->Unit(benchmark::kMillisecond)->Iterations(2);

/// Mean cost of inserting a keyframe (odometry factor + iSAM2 update) in a growing chain with
/// periodic loops.
void BM_Isam2KeyframeUpdate(benchmark::State& state) {
  const int n = 400;
  for (auto _ : state) {
    strataslam::Backend be{strataslam::BackendConfig{}};
    gtsam::Pose3 pose(gtsam::Rot3(), gtsam::Point3(0, 0, 1.2)), prev = pose;
    std::mt19937 rng(1);
    std::normal_distribution<double> noise(0, 0.01);
    for (int i = 0; i < n; ++i) {
      const gtsam::Pose3 step(gtsam::Rot3::Yaw(0.02), gtsam::Point3(0.25, 0, 0));
      pose = pose * step;
      strataslam::KeyframeInput kf;
      kf.id = static_cast<strataslam::KeyframeId>(i);
      kf.stamp = i * 100000000LL;
      kf.floor = 0;
      kf.T_odom_base =
          pose.compose(gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(noise(rng), noise(rng), 0)));
      std::optional<strataslam::KeyframeId> from;
      gtsam::Pose3 rel;
      if (i > 0) {
        from = static_cast<strataslam::KeyframeId>(i - 1);
        rel = prev.between(pose);
      }
      be.add_keyframe(kf, from, rel, std::nullopt);
      prev = pose;
    }
  }
  state.counters["time_per_keyframe"] =
      benchmark::Counter(static_cast<double>(n), benchmark::Counter::kIsIterationInvariantRate |
                                                     benchmark::Counter::kInvert);
}
BENCHMARK(BM_Isam2KeyframeUpdate)->Unit(benchmark::kMillisecond)->Iterations(3);

}  // namespace

BENCHMARK_MAIN();
