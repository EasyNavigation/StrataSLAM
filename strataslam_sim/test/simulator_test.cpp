#include <gtest/gtest.h>

#include <bitset>
#include <cmath>

#include "strataslam/sensors/imu_preintegration.hpp"
#include "strataslam_sim/simulator.hpp"

using namespace strataslam_sim;
using Eigen::Vector3d;

namespace {

struct Fixture {
  std::shared_ptr<Building> b;
  std::shared_ptr<Trajectory> t;
  Fixture(bool lidar = false, bool elevator = false) {
    BuildingConfig bc;
    bc.num_floors = 2;
    b = std::make_shared<Building>(bc);
    TrajectoryConfig tc;
    TrajectoryBuilder tb(*b, tc, 0, {3, 3}, 0.0);
    tb.hold(2.0).walk({{29, 3}, {29, 19}});
    if (elevator) tb.walk({{21.6, 3.0}, {21.6, 0.5}, {21.6, -1.1}}).turn_in_place(M_PI / 2).elevator(0, 1);
    t = std::make_shared<Trajectory>(tb.build());
    (void)lidar;
  }
};

int hamming(const strataslam::Descriptor& a, const strataslam::Descriptor& b) {
  int d = 0;
  for (size_t i = 0; i < a.size(); ++i) d += std::bitset<8>(a[i] ^ b[i]).count();
  return d;
}

}  // namespace

TEST(Simulator, ImuAtRestMeasuresGravityUpAndBiasIsSmall) {
  Fixture f;
  SimConfig sc;
  sc.accel_bias_init = 0.0;
  sc.gyro_bias_init = 0.0;
  Simulator sim(f.b, f.t, sc);
  Vector3d sum = Vector3d::Zero(), gsum = Vector3d::Zero();
  const int n = 300;  // 1.5 s while stationary
  for (int i = 0; i < n; ++i) {
    const auto m = sim.imu(i);
    sum += m.accel;
    gsum += m.gyro;
  }
  const Vector3d mean = sum / n;
  EXPECT_NEAR(mean.norm(), 9.80665, 0.1);
  EXPECT_GT(mean.z(), 9.5);  // specific force points up in the base frame
  EXPECT_LT((gsum / n).norm(), 0.01);
}

TEST(Simulator, ImuDoubleIntegrationTracksTrajectory) {
  Fixture f;
  SimConfig sc;
  sc.accel_bias_init = 0.0;
  Simulator sim(f.b, f.t, sc);
  // Integrate world acceleration (using true orientation) for 6 s of walking.
  const double dt = 1.0 / sc.imu_hz;
  Vector3d v = f.t->samples()[0].v, p = f.t->samples()[0].p;
  const int n = 6 * 200;
  for (int i = 0; i < n; ++i) {
    const auto m = sim.imu(i);
    const auto s = f.t->state_at(i * dt);
    const Vector3d a_world = s.R * m.accel - Vector3d(0, 0, sc.gravity);
    p += v * dt + 0.5 * a_world * dt * dt;
    v += a_world * dt;
  }
  const Vector3d truth = f.t->state_at(n * dt).p;
  EXPECT_LT((p - truth).norm(), 0.5);
}

TEST(Simulator, BarometerMatchesHypsometricHeight) {
  Fixture f(false, true);
  SimConfig sc;
  sc.baro = true;
  Simulator sim(f.b, f.t, sc);
  const size_t n = sim.count(SensorEvent::Type::Baro);
  // Compare first and last 2 s of readings: heights differ by ~3 m through the elevator ride.
  const auto avg = [&](size_t a, size_t b) {
    double s = 0;
    for (size_t i = a; i < b; ++i) s += sim.baro(i).pressure_pa;
    return s / (b - a);
  };
  const double p0 = avg(0, 40), p1 = avg(n - 40, n);
  const double dh = (287.05 * sc.temperature_k / sc.gravity) * std::log(p0 / p1);
  EXPECT_NEAR(dh, 3.0, 0.15);
}

TEST(Simulator, CameraProducesConsistentKeypointsAndDepth) {
  Fixture f;
  SimConfig sc;
  sc.pixel_sigma = 0.0;
  Simulator sim(f.b, f.t, sc);
  const auto frame = sim.rgbd(5);
  EXPECT_GT(frame.keypoints.size(), 80u);
  EXPECT_EQ(frame.depth.width, 320);
  // Backproject keypoints with depth and compare with landmarks: the nearest landmark must be close
  // and have a similar descriptor (a few flipped bits).
  const auto K = frame.K;
  const gtsam::Pose3 T_wc = sim.gt_pose(frame.stamp) * sc.extrinsics.T_base_camera;
  int good = 0, tested = 0;
  for (const auto& kp : frame.keypoints) {
    const float z = frame.depth.at(static_cast<int>(std::round(kp.u)), static_cast<int>(std::round(kp.v)));
    if (z <= 0) continue;
    const Vector3d pc((kp.u - K.cx) / K.fx * z, (kp.v - K.cy) / K.fy * z, z);
    const Vector3d pw = T_wc.transformFrom(pc);
    ++tested;
    for (const auto& lm : f.b->landmarks()) {
      if (hamming(lm.descriptor, kp.descriptor) <= 12 && (lm.position - pw).norm() < 0.25) {
        ++good;
        break;
      }
    }
  }
  EXPECT_GT(tested, 60);
  EXPECT_GT(good, 0.8 * tested);
}

TEST(Simulator, LidarScanHasRangesAndMotionDistortionTimes) {
  Fixture f;
  SimConfig sc;
  sc.camera = false;
  sc.lidar = true;
  Simulator sim(f.b, f.t, sc);
  const auto scan = sim.lidar(30);  // t = 3 s, walking
  EXPECT_GT(scan.points.size(), 1500u);
  EXPECT_EQ(scan.points.size(), scan.time_offsets_s.size());
  float tmax = 0;
  for (size_t i = 0; i < scan.points.size(); ++i) {
    const float r = scan.points[i].norm();
    ASSERT_GE(r, 0.45f);
    ASSERT_LE(r, 60.5f);
    tmax = std::max(tmax, scan.time_offsets_s[i]);
  }
  EXPECT_NEAR(tmax, 0.1f, 0.01f);
}

TEST(Simulator, ElevatorCabinSeenFromInsideIsStaticToLidar) {
  Fixture f(true, true);
  SimConfig sc;
  sc.camera = false;
  sc.lidar = true;
  Simulator sim(f.b, f.t, sc);
  // Pick a scan in the middle of the ride (door closed): every point is within the cabin (< 3 m).
  // A time in the middle of the ride: door closed and the cabin at cruising speed.
  double t_ride = 0;
  for (const auto& smp : f.t->samples()) {
    if (smp.door_closed && std::abs(smp.v.z()) > 0.9) { t_ride = smp.t; break; }
  }
  ASSERT_GT(t_ride, 0.0);
  const auto scan = sim.lidar(static_cast<size_t>(t_ride * sc.lidar_hz));
  ASSERT_GT(scan.points.size(), 500u);
  float rmax = 0;
  for (const auto& p : scan.points) rmax = std::max(rmax, p.norm());
  EXPECT_LT(rmax, 3.0f);
}

TEST(Simulator, EventsSortedAndDeterministic) {
  Fixture f;
  SimConfig sc;
  sc.baro = true;
  sc.lidar = true;
  sc.external_odom = true;
  Simulator a(f.b, f.t, sc), b(f.b, f.t, sc);
  const auto& ev = a.events();
  for (size_t i = 1; i < ev.size(); ++i) ASSERT_LE(ev[i - 1].stamp, ev[i].stamp);
  EXPECT_EQ(a.imu(123).accel, b.imu(123).accel);
  EXPECT_EQ(a.baro(7).pressure_pa, b.baro(7).pressure_pa);
  EXPECT_EQ(a.rgbd(3).keypoints.size(), b.rgbd(3).keypoints.size());
  EXPECT_EQ(a.rgbd(3).keypoints[5].descriptor, b.rgbd(3).keypoints[5].descriptor);
  EXPECT_EQ(a.lidar(2).points.size(), b.lidar(2).points.size());
  EXPECT_EQ(a.odom(10).pose.translation(), b.odom(10).pose.translation());
  SimConfig other = sc;
  other.seed = 7;
  Simulator c(f.b, f.t, other);
  EXPECT_NE(a.imu(123).accel, c.imu(123).accel);
}

TEST(Simulator, ExternalOdometryDriftsSlowly) {
  Fixture f;
  SimConfig sc;
  sc.external_odom = true;
  Simulator sim(f.b, f.t, sc);
  const size_t n = sim.count(SensorEvent::Type::Odom);
  const gtsam::Pose3 gt0 = f.t->pose_at(0);
  const gtsam::Pose3 gt1 = f.t->pose_at((n - 1) / sc.odom_hz);
  const gtsam::Pose3 rel_gt = gt0.between(gt1);
  const gtsam::Pose3 rel_odom = sim.odom(n - 1).pose;
  const double traveled = rel_gt.translation().norm();
  EXPECT_LT((rel_odom.translation() - rel_gt.translation()).norm(), 0.1 * traveled + 0.1);
}

TEST(Simulator, PreintegrationPredictsTheStateTwoSecondsAhead) {
  Fixture f;
  SimConfig sc;
  sc.accel_bias_init = 0.0;
  sc.gyro_bias_init = 0.0;
  Simulator sim(f.b, f.t, sc);
  // Window t in [3, 5] s while walking: start from the true state, integrate the simulated IMU.
  const size_t i0 = 3 * 200, i1 = 5 * 200;
  strataslam::ImuNoise noise;
  strataslam::ImuPreintegrator pim(noise);
  std::vector<strataslam::ImuSample> batch;
  for (size_t i = i0 + 1; i <= i1; ++i) batch.push_back(sim.imu(i));
  pim.add_batch(batch, sim.imu(i0).stamp);
  EXPECT_NEAR(pim.delta_t(), 2.0, 1e-9);
  const auto s0 = f.t->state_at(i0 / sc.imu_hz), s1 = f.t->state_at(i1 / sc.imu_hz);
  const gtsam::NavState start(f.t->pose_at(i0 / sc.imu_hz), s0.v);
  const gtsam::NavState pred = pim.predict(start);
  EXPECT_LT((pred.pose().translation() - s1.p).norm(), 0.12);
  EXPECT_LT((pred.velocity() - s1.v).norm(), 0.15);
  const gtsam::Rot3 err = pred.pose().rotation().between(f.t->pose_at(i1 / sc.imu_hz).rotation());
  EXPECT_LT(gtsam::Rot3::Logmap(err).norm() * 180 / M_PI, 0.5);
  // Resetting discards the integration.
  pim.reset(gtsam::imuBias::ConstantBias());
  EXPECT_EQ(pim.delta_t(), 0.0);
}
