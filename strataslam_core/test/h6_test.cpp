#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/factorTesting.h>

#include <cmath>
#include <random>

#include "strataslam/backend/factors.hpp"
#include "strataslam/floors/floor_manager.hpp"
#include "strataslam/geometry/gravity.hpp"
#include "strataslam/pipeline/slam_system.hpp"
#include "strataslam/sensors/barometer.hpp"

using namespace strataslam;
using Eigen::Vector3d;

// ---------------- barometer ----------------

TEST(Barometer, EqualPressureGivesZeroHeight) {
  EXPECT_DOUBLE_EQ(hypsometric_height(101325.0, 101325.0, 288.15), 0.0);
}

TEST(Barometer, KnownHeightIsRecoveredToOneMicrometre) {
  const double T = 288.15, p_ref = 101325.0;
  const double p = p_ref * std::exp(-kStandardGravity * 10.0 / (kDryAirGasConstant * T));
  EXPECT_NEAR(hypsometric_height(p_ref, p, T), 10.0, 1e-6);
  const double p_down = p_ref * std::exp(+kStandardGravity * 3.0 / (kDryAirGasConstant * T));
  EXPECT_NEAR(hypsometric_height(p_ref, p_down, T), -3.0, 1e-6);
}

TEST(Barometer, FilterSettlesInTheConfiguredTimeAfterAOneHectopascalStep) {
  const double T = 288.15, settle = 3.0, p0 = 101325.0;
  BaroFilter f(T, settle);
  Timestamp t = 0;
  const double dt = 0.05;
  for (int i = 0; i < 40; ++i, t += static_cast<Timestamp>(dt * 1e9)) f.update(t, p0);
  EXPECT_NEAR(*f.height(), 0.0, 1e-9);
  const double target = hypsometric_height(p0, p0 - 100.0, T);  // 1 hPa lower: ~8.4 m higher
  EXPECT_GT(target, 8.0);
  double at_settle = 0, prev = 0;
  for (int i = 0; i <= static_cast<int>(settle / dt) + 2;
       ++i, t += static_cast<Timestamp>(dt * 1e9)) {
    const double h = f.update(t, p0 - 100.0);
    EXPECT_GE(h, prev - 1e-12);  // monotone, no overshoot
    EXPECT_LE(h, target + 1e-9);
    prev = h;
    if (i == static_cast<int>(settle / dt)) at_settle = h;
  }
  EXPECT_NEAR(at_settle, target, 0.012 * target);      // within ~1 % at the settling time
  EXPECT_LT(std::abs(prev - target), 0.012 * target);  // still within ~1 % two samples later
}

// First-order response: 63.2 % of a step at t = tau (= settle time / ln 100), nothing of it
// instantly, and white pressure noise attenuated as a low-pass filter must (audit AUD-012: a filter
// removed passed every test).
TEST(Barometer, StepResponseFollowsTheTimeConstantAndNoiseIsAttenuated) {
  const double T = 288.15, settle = 3.0, p0 = 101325.0, dt = 0.01;
  const double tau = settle / std::log(100.0);
  BaroFilter f(T, settle);
  Timestamp t = 0;
  for (int i = 0; i < 100; ++i, t += static_cast<Timestamp>(dt * 1e9)) f.update(t, p0);
  const double target = hypsometric_height(p0, p0 - 100.0, T);
  const double first = f.update(t, p0 - 100.0);
  EXPECT_LT(first,
            0.05 * target);  // one 10 ms sample after the step: about 1.5 %, not the whole step
  const int k_tau = static_cast<int>(std::lround(tau / dt));
  double h = first;
  for (int i = 1; i <= k_tau; ++i) {
    t += static_cast<Timestamp>(dt * 1e9);
    h = f.update(t, p0 - 100.0);
  }
  EXPECT_NEAR(h / target, 1.0 - std::exp(-k_tau * dt / tau), 0.01);

  BaroFilter g(T, settle);
  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, 2.0);  // Pa, about 0.17 m
  double sum_raw = 0, sum2_raw = 0, sum_f = 0, sum2_f = 0;
  int n = 0;
  t = 0;
  for (int i = 0; i < 4000; ++i, t += static_cast<Timestamp>(0.04 * 1e9)) {
    const double p = p0 + noise(rng);
    const double hf = g.update(t, p);
    if (i < 500) continue;
    const double hr = hypsometric_height(p0, p, T);
    sum_raw += hr;
    sum2_raw += hr * hr;
    sum_f += hf;
    sum2_f += hf * hf;
    ++n;
  }
  const double sd_raw = std::sqrt(sum2_raw / n - std::pow(sum_raw / n, 2));
  const double sd_f = std::sqrt(sum2_f / n - std::pow(sum_f / n, 2));
  EXPECT_LT(sd_f, 0.3 * sd_raw);  // a first-order filter at 25 Hz with tau 0.65 s: about 0.18 x
}

TEST(Barometer, BaseStationCancelsWeatherDrift) {
  const double T = 288.15;
  BaroFilter plain(T, 0.1), diff(T, 0.1);
  Timestamp t = 0;
  double h_plain = 0, h_diff = 0;
  const double base0 = 101325.0;
  diff.set_base_station(base0);
  plain.update(t, base0);
  diff.update(t, base0);
  // The weather lowers the pressure by 50 Pa for everybody; the rover does not move.
  for (int i = 1; i < 100; ++i) {
    t += 50000000;
    diff.set_base_station(base0 - 50.0);
    h_plain = plain.update(t, base0 - 50.0);
    h_diff = diff.update(t, base0 - 50.0);
  }
  EXPECT_GT(std::abs(h_plain), 3.0);
  EXPECT_LT(std::abs(h_diff), 0.05);
}

// ---------------- BaroDeltaZFactor ----------------

TEST(BaroDeltaZFactor, ZeroErrorAndCorrectJacobians) {
  using gtsam::Symbol;
  const gtsam::Pose3 Xi(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.7), Vector3d(1, 2, 3));
  const gtsam::Pose3 Xj(gtsam::Rot3::RzRyRx(-0.3, 0.1, 1.9), Vector3d(-2, 0.5, 4.4));
  gtsam::Values v;
  v.insert(Symbol('x', 1), Xi);
  v.insert(Symbol('x', 2), Xj);
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(1, 0.05);
  BaroDeltaZFactor exact(Symbol('x', 1), Symbol('x', 2), 1.4, noise);
  EXPECT_NEAR(exact.unwhitenedError(v)(0), 0.0, 1e-12);  // (4.4 - 3) - 1.4
  BaroDeltaZFactor off(Symbol('x', 1), Symbol('x', 2), 1.0, noise);
  EXPECT_NEAR(off.unwhitenedError(v)(0), 0.4, 1e-12);
  EXPECT_TRUE(gtsam::internal::testFactorJacobians("BaroDeltaZFactor", off, v, 1e-5, 1e-6));
  EXPECT_TRUE(gtsam::internal::testFactorJacobians("BaroDeltaZFactor", exact, v, 1e-5, 1e-6));
}

// ---------------- gravity ----------------

TEST(Gravity, ImuAtRestGivesTheUpDirection) {
  std::mt19937 rng(1);
  std::normal_distribution<double> n(0, 0.03);
  const gtsam::Rot3 R_wb = gtsam::Rot3::RzRyRx(0.05, -0.08, 0.4);  // body tilted a few degrees
  std::vector<ImuSample> s;
  for (int i = 0; i < 200; ++i) {
    ImuSample m;
    m.stamp = i * 5000000LL;
    m.accel = R_wb.matrix().transpose() * Vector3d(0, 0, kStandardGravity) +
              Vector3d(n(rng), n(rng), n(rng));
    m.gyro = Vector3d(0.001, -0.002, 0.0005) + Vector3d(n(rng), n(rng), n(rng)) * 0.05;
    s.push_back(m);
  }
  const auto g = estimate_gravity_from_imu(s);
  ASSERT_TRUE(g.has_value());
  const Vector3d truth = R_wb.matrix().transpose() * Vector3d::UnitZ();
  EXPECT_LT(std::acos(std::min(1.0, g->up_body.dot(truth))) * 180 / M_PI, 0.5);
  EXPECT_NEAR(g->accel_norm, kStandardGravity, 0.05);
  // The alignment rotation takes the measured up to world +z.
  EXPECT_LT((rotation_from_up(g->up_body).matrix() * g->up_body - Vector3d::UnitZ()).norm(), 1e-9);
}

TEST(Gravity, MovingOrWrongUnitsAreRejected) {
  std::vector<ImuSample> moving;
  for (int i = 0; i < 100; ++i) {
    ImuSample m;
    m.accel = Vector3d(0, 0, kStandardGravity);
    m.gyro = Vector3d(0, 0, 0.5);  // turning
    moving.push_back(m);
  }
  EXPECT_FALSE(estimate_gravity_from_imu(moving).has_value());
  std::vector<ImuSample> g_units;
  for (int i = 0; i < 100; ++i) {
    ImuSample m;
    m.accel = Vector3d(0, 0, 1.0);  // in g, not m/s^2
    g_units.push_back(m);
  }
  EXPECT_FALSE(estimate_gravity_from_imu(g_units).has_value());
  EXPECT_FALSE(estimate_gravity_from_imu({}).has_value());
}

TEST(Gravity, RotationFromUpHandlesEveryDirection) {
  for (const Vector3d& u :
       {Vector3d(0, 0, 1), Vector3d(0, 0, -1), Vector3d(1, 0, 0), Vector3d(0.3, -0.5, 0.8)}) {
    const Vector3d un = u.normalized();
    EXPECT_LT((rotation_from_up(un).matrix() * un - Vector3d::UnitZ()).norm(), 1e-9);
  }
}

TEST(Gravity, GroundPlaneRansacFindsTheTiltOfARoom) {
  // A room seen by a sensor tilted 4 degrees in roll and 3 in pitch: points in the sensor frame.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(0, 1);
  const gtsam::Rot3 R_ws = gtsam::Rot3::RzRyRx(0.07, -0.05, 0.3);  // sensor -> world
  std::vector<Eigen::Vector3f> pts;
  const auto add_plane = [&](const Vector3d& o, const Vector3d& a, const Vector3d& b, int n) {
    for (int i = 0; i < n; ++i) {
      const Vector3d w = o + u(rng) * a + u(rng) * b + Vector3d(0, 0, 0);
      pts.push_back((R_ws.matrix().transpose() * (w - Vector3d(0, 0, 1.2)))
                        .cast<float>());  // sensor at 1.2 m height
    }
  };
  add_plane({-4, -3, 0}, {8, 0, 0}, {0, 6, 0}, 6000);    // floor
  add_plane({-4, -3, 2.8}, {8, 0, 0}, {0, 6, 0}, 3000);  // ceiling
  add_plane({-4, -3, 0}, {8, 0, 0}, {0, 0, 2.8}, 3000);  // wall
  add_plane({-4, -3, 0}, {0, 6, 0}, {0, 0, 2.8}, 3000);  // wall
  add_plane({4, -3, 0}, {0, 6, 0}, {0, 0, 2.8}, 3000);   // wall
  const Vector3d truth =
      R_ws.matrix().transpose() * Vector3d::UnitZ();  // world up in the sensor frame
  const auto up = estimate_up_from_ground_plane(pts, Vector3d::UnitZ(), 1234);
  ASSERT_TRUE(up.has_value());
  EXPECT_LT(std::acos(std::min(1.0, up->dot(truth))) * 180 / M_PI, 1.0);
  // Deterministic for a fixed seed.
  EXPECT_EQ(*up, *estimate_up_from_ground_plane(pts, Vector3d::UnitZ(), 1234));
  // No dominant horizontal plane (a pure wall cloud): rejected.
  std::vector<Eigen::Vector3f> wall;
  for (int i = 0; i < 2000; ++i)
    wall.emplace_back(2.0f, static_cast<float>(u(rng) * 4 - 2), static_cast<float>(u(rng) * 2 - 1));
  EXPECT_FALSE(estimate_up_from_ground_plane(wall, Vector3d::UnitZ(), 1).has_value());
}

// A LiDAR in a corridor sees mostly walls: the floor is about a tenth of the points (S02, seed
// 7017, in the audit AUD-022). With RANSAC over every point, three floor points were rarely drawn,
// and a slab of wall above the sensor, tilted 10 degrees, passed as the ground: the odometry then
// climbed 5 cm per metre. The ground is only looked for below the sensor now. Whatever the seed,
// the answer is right within a degree or there is none.
TEST(Gravity, GroundPlaneIsFoundBelowTheSensorWhenTheFloorIsASmallPartOfTheScan) {
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(0, 1);
  std::normal_distribution<double> noise(0, 0.02);
  std::vector<Eigen::Vector3f> pts;
  const auto add = [&](const Vector3d& o, const Vector3d& a, const Vector3d& b, int n) {
    for (int i = 0; i < n; ++i) {
      const Vector3d w = o + u(rng) * a + u(rng) * b + Vector3d(noise(rng), noise(rng), noise(rng));
      pts.push_back((w - Vector3d(0, 0, 1.2)).cast<float>());  // level sensor at 1.2 m
    }
  };
  add({-1.5, -10, 0}, {3, 0, 0}, {0, 20, 0}, 600);     // floor of a 3 m wide corridor: about 10 %
  add({-1.5, -10, 0}, {0, 20, 0}, {0, 0, 2.8}, 2400);  // the two long walls
  add({1.5, -10, 0}, {0, 20, 0}, {0, 0, 2.8}, 2400);
  add({-1.5, -10, 2.8}, {3, 0, 0}, {0, 20, 0}, 400);  // ceiling
  int found = 0;
  for (uint64_t seed = 0; seed < 60; ++seed) {
    const auto up = estimate_up_from_ground_plane(pts, Vector3d::UnitZ(), seed);
    if (!up) continue;
    ++found;
    EXPECT_LT(std::acos(std::min(1.0, up->dot(Vector3d::UnitZ()))) * 180 / M_PI, 1.0)
        << "seed " << seed;
  }
  EXPECT_GE(found, 54);  // and the floor is found almost always
}

// The first LiDAR scan levels the odometry frame however small the tilt. The 0.43 degrees here are
// the simulator's start attitude (wobble: roll 0.118, pitch 0.414 at t = 0); one cloud measures
// them within a few hundredths of a degree. Rounding them to level, as a 0.5-degree threshold at
// start-up did, left that error in every keyframe: S13 lost 0.14 m of floor height on average over
// 20 seeds and S07 3 of them (audit FIXES, AUD-022).
TEST(Gravity, TheFirstScanLevelsTheOdometryEvenForASmallTilt) {
  const double roll = 0.118 * M_PI / 180, pitch = 0.414 * M_PI / 180;
  const gtsam::Rot3 R_wb = gtsam::Rot3::Ypr(0.3, pitch, roll);  // base -> world, with some heading
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(0, 1);
  std::normal_distribution<double> noise(0, 0.01);
  LidarScan scan;
  const auto add = [&](const Vector3d& o, const Vector3d& a, const Vector3d& b, int n) {
    for (int i = 0; i < n; ++i) {
      const Vector3d w = o + u(rng) * a + u(rng) * b + Vector3d(noise(rng), noise(rng), noise(rng));
      scan.points.push_back(
          (R_wb.matrix().transpose() * (w - Vector3d(0, 0, 1.2))).cast<float>());  // sensor = base
    }
  };
  add({-4, -3, 0}, {8, 0, 0}, {0, 6, 0}, 4000);    // floor
  add({-4, -3, 2.8}, {8, 0, 0}, {0, 6, 0}, 2000);  // ceiling
  add({-4, -3, 0}, {8, 0, 0}, {0, 0, 2.8}, 2000);  // walls
  add({-4, 3, 0}, {8, 0, 0}, {0, 0, 2.8}, 2000);
  add({-4, -3, 0}, {0, 6, 0}, {0, 0, 2.8}, 2000);
  add({4, -3, 0}, {0, 6, 0}, {0, 0, 2.8}, 2000);
  scan.stamp = 1'000'000'000LL;
  Config cfg;
  cfg.sensors.lidar.enabled = true;
  cfg.fusion.odometry_from = OdometryFrom::Lidar;
  SlamSystem slam(cfg, Extrinsics());
  slam.add_lidar(scan);
  const auto pose = slam.current_pose_odom();
  ASSERT_TRUE(pose.has_value());
  const Vector3d up_est =
      pose->rotation().matrix().transpose() * Vector3d::UnitZ();  // world up seen from the base
  const Vector3d up_true = R_wb.matrix().transpose() * Vector3d::UnitZ();
  EXPECT_LT(std::acos(std::min(1.0, up_est.dot(up_true))) * 180 / M_PI, 0.1);
}

// ---------------- floor manager with IMU / barometer profiles ----------------

namespace {
struct ElevatorRig {
  FloorManager mgr;
  Timestamp t = 1'000'000'000'000LL;
  std::vector<FloorEvent> events;
  bool tracking =
      true;  ///< false: the odometry is lost (an elevator door closing is where it breaks)
  explicit ElevatorRig(FloorManagerConfig c = {}) : mgr(c) {}
  void step(double az, std::optional<double> baro) {
    FloorSample s;
    s.stamp = t;
    s.tracking_ok = tracking;
    s.position = Vector3d(0, 0, 1.2);  // the odometry sees the cabin standing still
    s.baro_height = baro;
    s.imu_vertical_accel = az;
    const FloorEvent e = mgr.update(s);
    if (e.kind != FloorEvent::Kind::None) events.push_back(e);
    t += 100000000;
  }
  /// Elevator ride: a up for ta, cruise tc, a down for ta (the spec's pulse - cruise - pulse).
  void ride(double a, double ta, double tc, double bias, bool with_baro, double noise_sigma = 0.0) {
    std::mt19937 rng(11);
    std::normal_distribution<double> n(0, noise_sigma > 0 ? noise_sigma : 1.0);
    double z = 0, v = 0;
    const double dt = 0.1;
    const double total = 2 * ta + tc;
    for (double tt = 0; tt < total; tt += dt) {
      const double acc = tt < ta ? a : (tt < ta + tc ? 0.0 : -a);
      v += acc * dt;
      z += v * dt;
      step(acc + bias + (noise_sigma > 0 ? n(rng) : 0.0),
           with_baro ? std::optional<double>(z) : std::nullopt);
    }
    last_z = z;
  }
  void rest(double s, double z, double bias, bool with_baro) {
    for (double tt = 0; tt < s; tt += 0.1)
      step(bias, with_baro ? std::optional<double>(z) : std::nullopt);
  }
  const FloorEvent* ended() const {
    for (const auto& e : events)
      if (e.kind == FloorEvent::Kind::TransitionEnded) return &e;
    return nullptr;
  }
  double last_z = 0;
};
}  // namespace

TEST(FloorManagerImu, ElevatorProfileWithBarometerAndImuIsAnElevator) {
  ElevatorRig r;
  r.rest(30, 0.0, 0.02, true);
  r.ride(0.5, 2.0, 1.0, 0.02, true);
  r.rest(6, r.last_z, 0.02, true);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_EQ(e->to, 1);
  EXPECT_NEAR(e->delta_z, 3.0, 0.05);
  EXPECT_TRUE(e->needs_frontend_reset);
}

TEST(FloorManagerImu, ElevatorWithoutBarometerIsMeasuredByTheImuAlone) {
  ElevatorRig r;
  r.rest(30, 0.0, 0.03, false);  // long rest so that the vertical accelerometer bias is learned
  r.ride(0.5, 2.0, 1.0, 0.03, false, 0.05);
  r.rest(8, 0.0, 0.03, false);
  int started = 0, unobserved = 0;
  for (const auto& e : r.events) {
    started += e.kind == FloorEvent::Kind::TransitionStarted;
    unobserved += e.kind == FloorEvent::Kind::TransitionStarted && e.vertical_unobserved;
  }
  EXPECT_EQ(started, 1);
  EXPECT_EQ(unobserved, 1);  // the pipeline must restart the frontend
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_EQ(e->to, 1);
  EXPECT_NEAR(e->delta_z, 3.0, 0.5);  // S06: height error <= 0.5 m
}

// The odometry breaks when the cabin door closes and the ride starts more than
// tracking_loss_timeout (2 s) later: the IMU must still open the transition (audit AUD-006: S11
// lost its third floor in 4 of 20 seeds this way).
TEST(FloorManagerImu, ElevatorIsDetectedByTheImuWhileTheOdometryIsLost) {
  ElevatorRig r;
  r.rest(30, 0.0, 0.03, false);
  r.tracking = false;
  r.rest(3.0, 0.0, 0.03, false);  // lost 3 s before the cabin accelerates
  r.ride(0.5, 2.0, 1.0, 0.03, false, 0.05);
  r.rest(2.0, 0.0, 0.03, false);
  r.tracking = true;
  r.rest(8, 0.0, 0.03, false);
  int started = 0;
  for (const auto& e : r.events)
    started += e.kind == FloorEvent::Kind::TransitionStarted && e.vertical_unobserved;
  EXPECT_EQ(started, 1);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_EQ(e->to, 1);
  EXPECT_NEAR(e->delta_z, 3.0, 0.5);
}

TEST(FloorManagerImu, WalkingBobbingDoesNotOpenATransition) {
  ElevatorRig r;
  // 2 Hz vertical oscillation of +-1.5 m/s^2 for 40 s (walking), odometry level.
  for (int i = 0; i < 400; ++i)
    r.step(1.5 * std::sin(2 * M_PI * 2.0 * i * 0.1 + 0.3), std::nullopt);
  EXPECT_EQ(r.events.size(), 0u);
  EXPECT_EQ(r.mgr.floor(), 0);
}

TEST(FloorManagerImu, ImpulsesOfTheAccelerometerOnAParkedOrRollingRobotAreNotAnElevator) {
  // Gazebo garage: a wheel hitting an edge reads up to 79 m/s^2 for one sample at 20 Hz. The mean
  // of one second of samples would then read +2.6 m/s^2 and open an elevator; a median ignores the
  // impulse.
  ElevatorRig r;
  for (int i = 0; i < 600; ++i) r.step((i % 20 == 7) ? 79.0 : 0.0, std::nullopt);
  EXPECT_EQ(r.events.size(), 0u);
  EXPECT_EQ(r.mgr.floor(), 0);
}

TEST(FloorManagerImu, AnElevatorIsStillMeasuredWhenTheAccelerometerHasImpulses) {
  ElevatorRig r;
  r.rest(30, 0.0, 0.0, false);
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> u(0, 1);
  double v = 0;
  const double a = 0.5, dt = 0.1;
  for (double tt = 0; tt < 5.0;
       tt += dt) {  // 2 s up, 1 s cruise, 2 s down, with an impulse in 1 sample of 20
    const double acc = tt < 2.0 ? a : (tt < 3.0 ? 0.0 : -a);
    v += acc * dt;
    r.step(acc + (u(rng) < 0.05 ? 60.0 : 0.0), std::nullopt);
  }
  r.rest(8, 0.0, 0.0, false);
  const FloorEvent* e = r.ended();
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->type, TransitionType::Elevator);
  EXPECT_NEAR(e->delta_z, 3.0, 0.8);
}

TEST(FloorManagerImu, DownwardRideGoesToTheFloorBelowAndBackAgainReusesLevels) {
  ElevatorRig r;
  r.rest(30, 0.0, 0.0, false);
  r.ride(0.5, 2.0, 1.0, 0.0, false);  // up 3 m
  r.rest(8, 0.0, 0.0, false);
  ASSERT_EQ(r.mgr.floor(), 1);
  r.ride(-0.5, 2.0, 1.0, 0.0, false);  // down 3 m
  r.rest(8, 0.0, 0.0, false);
  EXPECT_EQ(r.mgr.floor(), 0);
  EXPECT_EQ(r.mgr.num_floors(), 2u);
}

// The base station's first sample arrives after the rover's: its reference is taken then, and the
// weather drift that follows is cancelled (audit A4A5-02: the correction was never applied, 16.66 m
// with -200 Pa of drift).
TEST(Barometer, ABaseStationThatStartsLateStillCancelsTheWeather) {
  const double T = 288.15, p0 = 101325.0;
  BaroFilter f(T, 0.1);
  Timestamp t = 0;
  for (int i = 0; i < 20; ++i, t += 50'000'000) f.update(t, p0);  // the rover starts alone
  f.set_base_station(p0);                                         // then the base station comes up
  double h = 0;
  for (int i = 0; i < 400; ++i, t += 50'000'000) {
    const double weather =
        -200.0 * i / 399.0;  // the pressure falls 2 hPa everywhere (about 17 m of fake climb)
    f.set_base_station(p0 + weather);
    h = f.update(t, p0 + weather);
  }
  EXPECT_NEAR(h, 0.0, 0.05);
}
