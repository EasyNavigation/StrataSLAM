#include "strataslam/pipeline/slam_system.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <optional>
#include <thread>

#include "strataslam/common/version.hpp"
#include "strataslam/geometry/gravity.hpp"
#include "strataslam/mapping/colorize.hpp"
#include "strataslam/mapping/voxel_grid.hpp"
#include "strataslam/sensors/barometer.hpp"
#include "strataslam/sensors/imu_mount.hpp"

namespace strataslam {

struct SlamSystem::Impl {
  Config cfg;
  Extrinsics ext;
  SystemTuning tuning;
  std::shared_ptr<Logger> log;
  ResolvedModes modes;

  std::unique_ptr<LidarFrontend> lidar_fe;
  std::unique_ptr<VisualFrontend> visual_fe;
  OdometryFrontend* odom_fe = nullptr;  // the frontend that provides odometry
  Backend backend;
  FloorManager mgr;
  std::unique_ptr<BaroFilter> baro_filter;
  std::deque<OdomMeasurement>
      ext_odom;  // the robot's own odometry (wheels), the last seconds of it
  std::optional<double> baro_h;
  std::deque<ImuSample> imu_buf;        // recent IMU samples (last ~3 s), corrected (sign, scale)
  std::unique_ptr<ImuMount> imu_mount;  // learns how the IMU is really mounted
  GravityTracker gravity;               // up in the base frame, from the IMU
  bool imu_checked = false;             // the start-up check of gravity against the TF has run
  bool imu_trusted = false;  // its samples are used: the TF agreed, or the mounting was learned
  bool imu_learned = false;  // the mounting was learned from the motion (no more checks)
  double imu_sign = 1.0, imu_scale = 1.0;
  Timestamp last_fe_stamp =
      0;  // stamp of the last valid frontend pose (rotations for the IMU mounting)
  Timestamp turn_start =
      0;  // the rotations are paired with the gyroscope over windows of about 1 s
  Eigen::Vector3d turn_sum = Eigen::Vector3d::Zero();
  // The base's acceleration over the same windows, for the IMU's heading: velocity at the start,
  // sum of w x v dt, sum of up dt.
  std::optional<Eigen::Vector3d> win_v0, last_v;
  std::deque<Eigen::Vector3d> recent_v;  // the last per-frame velocities: their mean is last_v
  Eigen::Vector3d win_w_x_v = Eigen::Vector3d::Zero(), win_up = Eigen::Vector3d::Zero();
  std::vector<ImuSample> imu_since_update;
  std::optional<RgbdFrame>
      last_cam;  // newest camera frame (when the odometry comes from the LiDAR)
  std::map<KeyframeId, std::shared_ptr<const RgbdFrame>> keyframe_images;  // only when colorizing
  bool gravity_done = false;
  Timestamp first_frame = 0;  // of the odometry's sensor, while waiting to level the start
  // The frame being processed, so that a frontend restart can re-level on it.
  const RgbdFrame* cur_rgbd = nullptr;
  const LidarScan* cur_scan = nullptr;
  std::vector<KeyframeInfo> kfs;
  std::vector<TransitionRecord> transitions;
  std::map<FloorId, ScanContextDatabase> sc_db;
  std::map<KeyframeId, ScanContext> sc_of;
  std::map<FloorId, BowDatabase> bow_db;
  struct VisualKf {
    std::vector<RgbKeypoint> keypoints;
    std::vector<std::optional<Eigen::Vector3d>> p_cam;
    CameraIntrinsics K;
  };
  std::map<KeyframeId, VisualKf> visual_kf;

  KeyframeId next_kf = 0;
  SessionId session = 0;
  std::optional<KeyframeId> last_kf;  // last keyframe of the current session
  std::optional<KeyframeId>
      bridge_from;  // keyframe of the target floor to link the next keyframe to
  std::map<FloorId, KeyframeId> last_kf_on_floor;
  std::vector<KeyframeId>
      transit_kfs;  // keyframes created while a transition was open (no floor yet)
  std::map<FloorId, KeyframeId> last_kf_in_graph;  // keyed like the backend's graphs (all floors
                                                   // share one in the control mode)
  std::optional<gtsam::Pose3> last_pose;
  std::optional<gtsam::Pose3> last_kf_pose;
  Timestamp lost_since = 0;
  // Last pose the frontend tracked as Good: a loss is usually preceded by a few Degraded frames
  // whose pose (in particular the attitude) is the least reliable one to restart from.
  std::optional<std::pair<Timestamp, gtsam::Pose3>> last_good;
  Timestamp last_pose_stamp = 0;  // of last_pose
  Timestamp restart_stamp = 0;    // of the frame the frontend was last restarted at
  Timestamp frame_stamp = 0;      // of the frame being processed
  // The body's attitude carried by the gyroscope alone (base frame, bias removed), every 20 ms over
  // the last 5 minutes: a frontend restarts from the last pose turned by what the body turned while
  // the odometry was blind (a person turning round in an elevator cabin: Elevator-VIGS Office1-E's
  // second floor was mapped heading back the way the first had come).
  std::deque<std::pair<Timestamp, Eigen::Quaterniond>> gyro_att;
  Eigen::Quaterniond gyro_q = Eigen::Quaterniond::Identity();
  Timestamp gyro_last = 0;
  int since_loop = 1000;
  size_t cross_floor_seen = 0;

  // A floor change in progress whose record is completed by the first keyframe on the new floor.
  struct OpenTransition {
    bool active = false;
    TransitionType type = TransitionType::Unknown;
    FloorId from = 0, to = 0;
    double dz = 0;
    bool continuous = false;
    bool dz_from_sensor =
        false;  ///< the odometry did not see the height change: use `dz` in the relative pose
    bool tracking_gap = false;  ///< tracking was lost and restarted during the climb
    std::optional<KeyframeId> kf_before;
  } open_tr;
  // A session whose floor is not known yet (after a tracking loss): placed by relocalization, or
  // on a new floor when relocalization keeps failing.
  struct Pending {
    bool active = false;
    std::vector<KeyframeId> kfs;
    FloorId from_floor = 0;
    std::optional<KeyframeId> kf_before;
    bool in_transition = false;  ///< started inside stairs or a ramp the odometry was measuring
  } pending;

  static FloorManagerConfig floor_config(const Config& c, FloorManagerConfig base) {
    base.floor_height_prior = c.multifloor.floor_height_prior;
    base.min_vertical_change = c.multifloor.min_vertical_change;
    base.tracking_loss_timeout = c.multifloor.tracking_loss_timeout;
    base.aerial_platform = c.robot.platform == Platform::Aerial;
    return base;
  }

  Impl(const Config& c, const Extrinsics& e, const SystemTuning& t, std::shared_ptr<Logger> l)
      : cfg(c),
        ext(e),
        tuning(t),
        log(std::move(l)),
        backend(t.backend, log),
        mgr(floor_config(c, t.floors)) {
    mgr.set_logger(log);
    modes = resolve_modes(cfg);
    const int threads =
        cfg.runtime.threads > 0
            ? cfg.runtime.threads
            : static_cast<int>(std::max(1u, std::thread::hardware_concurrency() / 2));
    LidarFrontendConfig lc = tuning.lidar;
    lc.min_range = cfg.sensors.lidar.min_range;
    lc.max_range = cfg.sensors.lidar.max_range;
    lc.keyframe_translation = cfg.output.keyframe_translation;
    lc.keyframe_rotation_deg = cfg.output.keyframe_rotation_deg;
    lc.threads = threads;
    lc.deskew = cfg.sensors.lidar.deskew != DeskewMode::False;
    tuning.lidar = lc;
    if (cfg.sensors.lidar.enabled) lidar_fe = std::make_unique<LidarFrontend>(lc, ext.T_base_lidar);

    VisualFrontendConfig vc = tuning.visual;
    vc.tracker.depth.depth_min = cfg.sensors.camera.depth_min;
    vc.tracker.depth.depth_max = cfg.sensors.camera.depth_max;
    vc.tracker.keyframe_translation = cfg.output.keyframe_translation;
    vc.tracker.keyframe_rotation_deg = cfg.output.keyframe_rotation_deg;
    tuning.visual = vc;
    if (cfg.sensors.camera.enabled)
      visual_fe = std::make_unique<VisualFrontend>(vc, ext.T_base_camera);

    if (cfg.sensors.imu.enabled) imu_mount = std::make_unique<ImuMount>(ext.T_base_imu.rotation());
    if (cfg.sensors.barometer.enabled)
      baro_filter = std::make_unique<BaroFilter>(cfg.sensors.barometer.temperature_k,
                                                 cfg.sensors.barometer.settle_time_s);
    odom_fe = modes.odometry == OdometrySource::Camera
                  ? static_cast<OdometryFrontend*>(visual_fe.get())
                  : static_cast<OdometryFrontend*>(lidar_fe.get());
  }

  /// The external odometry at `t`, interpolated between its two samples around it; nullopt outside
  /// them (a stamp up to kTolerance after the last sample takes the last one: the odometry may
  /// trail the camera).
  std::optional<OdomMeasurement> ext_odom_at(Timestamp t) const {
    constexpr Timestamp kTolerance = 50'000'000;  // ns
    if (ext_odom.empty() || t < ext_odom.front().stamp || t > ext_odom.back().stamp + kTolerance)
      return std::nullopt;
    if (t >= ext_odom.back().stamp) return OdomMeasurement{t, ext_odom.back().pose};
    const auto b =
        std::lower_bound(ext_odom.begin(), ext_odom.end(), t,
                         [](const OdomMeasurement& m, Timestamp v) { return m.stamp < v; });
    if (b->stamp == t || b == ext_odom.begin()) return OdomMeasurement{t, b->pose};
    const auto a = std::prev(b);
    const double alpha =
        static_cast<double>(t - a->stamp) / static_cast<double>(b->stamp - a->stamp);
    return OdomMeasurement{t, a->pose.interpolateRt(b->pose, alpha)};
  }

  FloorId key_floor(FloorId f) const { return tuning.backend.debug_disable_floor_gating ? 0 : f; }
  FloorId current_floor() const { return pending.active ? kUnassignedFloor : mgr.floor(); }

  ScanContextDatabase& sc_for(FloorId f) {
    auto it = sc_db.find(key_floor(f));
    if (it == sc_db.end())
      it = sc_db.emplace(key_floor(f), ScanContextDatabase(tuning.scan_context)).first;
    return it->second;
  }
  BowDatabase& bow_for(FloorId f) {
    auto it = bow_db.find(key_floor(f));
    if (it == bow_db.end()) it = bow_db.emplace(key_floor(f), BowDatabase(tuning.vocabulary)).first;
    return it->second;
  }

  // ---------------- planar mode ----------------

  bool planar() const { return cfg.multifloor.mode == LevelMode::Planar; }

  /// Planar mode: up in the base frame, from the IMU's gravity (a base on suspension tilts as it
  /// accelerates), else the base's own z (a level base).
  Eigen::Vector3d plane_up() const {
    if (imu_trusted)
      if (const auto u = gravity.up()) return *u;
    return Eigen::Vector3d::UnitZ();
  }

  /// Planar mode: the frontend's pose moved onto the plane of the floor (the odometry frame's z =
  /// 0, where the run started): x, y and heading kept, height and attitude imposed. The frontend's
  /// own drift in z and in roll and pitch never reaches the map.
  void to_plane(FrontendResult& r) const {
    if (!planar() || !r.valid) return;
    const gtsam::Pose3 p(level_to(r.T_odom_base.rotation(), plane_up()),
                         gtsam::Point3(r.T_odom_base.x(), r.T_odom_base.y(), 0.0));
    if (last_pose) r.delta = last_pose->between(p);
    r.T_odom_base = p;
  }

  // ---------------- IMU, gravity, barometer ----------------

  /// Up in the base frame as the platform defines it (a ground robot's base is parallel to the
  /// floor), else from the odometry's attitude: the reference for the IMU's gravity before its
  /// mounting is known.
  Eigen::Vector3d reference_up(const gtsam::Pose3& T_odom_base) const {
    if (cfg.robot.platform == Platform::Ground) return Eigen::Vector3d::UnitZ();
    return T_odom_base.rotation().matrix().transpose() * Eigen::Vector3d::UnitZ();
  }

  /// One IMU sample into the pipeline, corrected by what is known of the mounting.
  /// `track_gravity` false: a sample the gravity tracker has already had (replayed after a mounting
  /// was learned).
  void use_imu(const ImuSample& raw, bool track_gravity = true) {
    ImuSample m = raw;
    m.accel *= imu_sign * imu_scale;
    imu_buf.push_back(m);
    while (imu_buf.size() > 600) imu_buf.pop_front();
    imu_since_update.push_back(m);
    const Eigen::Matrix3d R = ext.T_base_imu.rotation().matrix();
    if (track_gravity) gravity.add(m.stamp, R * m.accel, R * m.gyro);
    integrate_gyro(m.stamp, R * m.gyro);
  }

  void integrate_gyro(Timestamp stamp, const Eigen::Vector3d& gyro_base) {
    if (stamp <= gyro_last) return;
    if (gyro_last != 0) {
      const double dt = std::min((stamp - gyro_last) * 1e-9, 0.1);
      const Eigen::Vector3d w = (gyro_base - gravity.gyro_bias()) * dt;
      const double angle = w.norm();
      if (angle > 0)
        gyro_q = (gyro_q * Eigen::Quaterniond(Eigen::AngleAxisd(angle, w / angle))).normalized();
    }
    gyro_last = stamp;
    constexpr Timestamp kStep = 20'000'000, kKeep = 300'000'000'000;
    if (gyro_att.empty() || stamp - gyro_att.back().first >= kStep)
      gyro_att.push_back({stamp, gyro_q});
    while (gyro_att.front().first < stamp - kKeep) gyro_att.pop_front();
  }

  /// What the body turned from `from` to `to` by the gyroscope (in the body frame at `from`), when
  /// its record covers them and the IMU's mounting can be trusted for it: on a ground robot, only
  /// once its heading has been learned (a TF turned half around reverses the roll and pitch rates).
  std::optional<gtsam::Rot3> gyro_turn(Timestamp from, Timestamp to) const {
    if (!imu_trusted || (cfg.robot.platform == Platform::Ground && !imu_learned))
      return std::nullopt;
    if (gyro_att.empty() || from == 0 || to <= from || from < gyro_att.front().first)
      return std::nullopt;
    const auto at = [&](Timestamp t) {
      auto it = std::upper_bound(gyro_att.begin(), gyro_att.end(), t,
                                 [](Timestamp v, const auto& e) { return v < e.first; });
      return it == gyro_att.begin() ? it->second : std::prev(it)->second;
    };
    return gtsam::Rot3((at(from).conjugate() * at(to)).toRotationMatrix());
  }

  /// After the first second: does gravity, through the TF, point up? On a ground robot it must; if
  /// it does not, the TF or the driver's sign convention is wrong and the IMU waits until its
  /// mounting is learned from the motion.
  void check_imu_against_tf() {
    const auto& recent = imu_mount->recent();
    if (recent.size() < 10 || (recent.back().stamp - recent.front().stamp) * 1e-9 < 1.0) return;
    imu_checked = true;
    const Eigen::Vector3d g = ext.T_base_imu.rotation().matrix() * imu_mount->mean_accel();
    const double angle = std::acos(std::clamp(g.normalized().z(), -1.0, 1.0)) * 180.0 / M_PI;
    imu_trusted = cfg.robot.platform != Platform::Ground || angle < 30.0;
    if (!imu_trusted) {
      log->log(LogLevel::Warn, strprintf("IMU: gravity is %.0f deg away from up by the TF (the TF "
                                         "or the driver's sign convention is "
                                         "wrong); not used until its mounting is learned from the "
                                         "motion (the robot must turn)",
                                         angle));
      return;
    }
    for (const auto& m : recent) use_imu(m);
  }

  /// The odometry turned the base between the last valid pose and this one: summed over about a
  /// second (where the rotation stands out of the per-frame noise and a small time offset between
  /// the sensors does not matter) and paired with the gyroscope for the IMU mounting. Over the same
  /// windows the base's acceleration, dv/dt + w x v in the base frame, is paired with the
  /// accelerometer for the heading of the IMU, which the turns of a ground robot do not show.
  void learn_imu_mounting(const FrontendResult& r, Timestamp stamp) {
    if (!imu_mount || imu_learned) return;
    constexpr Timestamp kWindow = 1'000'000'000;  // ns
    const Eigen::Vector3d up =
        r.T_odom_base.rotation().matrix().transpose() * Eigen::Vector3d::UnitZ();
    if (!last_fe_stamp) {
      turn_start = stamp;
      turn_sum.setZero();
      win_v0.reset();
      last_v.reset();
      recent_v.clear();
      win_w_x_v.setZero();
      win_up.setZero();
    } else {
      const Eigen::Vector3d rotvec = gtsam::Rot3::Logmap(r.delta.rotation());
      turn_sum += rotvec;
      const double dt = (stamp - last_fe_stamp) * 1e-9;
      if (dt > 1e-3) {
        const Eigen::Vector3d v = r.delta.translation() / dt;
        // The velocity at a window's edges is a mean over the last half second or so: one frame's
        // registration jitter, differentiated, would bury the robot's own acceleration.
        recent_v.push_back(v);
        if (recent_v.size() > 5) recent_v.pop_front();
        Eigen::Vector3d v_mean = Eigen::Vector3d::Zero();
        for (const auto& x : recent_v) v_mean += x;
        v_mean /= static_cast<double>(recent_v.size());
        if (!win_v0) win_v0 = last_v ? *last_v : v_mean;
        win_w_x_v += rotvec.cross(v);  // (w dt) x v
        win_up += up * dt;
        last_v = v_mean;
      }
    }
    last_fe_stamp = stamp;
    if (stamp - turn_start < kWindow) return;
    imu_mount->add_base_rotation(turn_start, stamp, turn_sum, reference_up(r.T_odom_base));
    if (win_v0 && last_v) {
      const double window_s = (stamp - turn_start) * 1e-9;
      imu_mount->add_base_motion(turn_start, stamp, (*last_v - *win_v0 + win_w_x_v) / window_s,
                                 win_up.normalized());
    }
    turn_start = stamp;
    turn_sum.setZero();
    win_v0 = last_v;
    win_w_x_v.setZero();
    win_up.setZero();
    const auto e = imu_mount->estimate();
    if (!e) return;
    if (debug_channel("IMU"))
      log->log(LogLevel::Debug,
               strprintf("IMU mounting: %.0f deg turned, motion energy %.2f, explained %.2f, "
                         "heading %.1f deg",
                         e->turned_deg, e->motion_energy, e->explained, e->yaw_deg));
    // Compared with the mounting in use (the TF's, or one learned before the heading was seen).
    const bool wrong =
        gtsam::Rot3::Logmap(ext.T_base_imu.rotation().between(e->R_base_imu)).norm() * 180.0 /
                M_PI >
            10.0 ||
        e->accel_sign != imu_sign || std::abs(e->accel_scale / imu_scale - 1.0) > 0.03;
    if (!e->yaw_observed && imu_trusted && !wrong)
      return;  // agrees so far: learning goes on until the accelerations show the heading
    imu_learned = e->yaw_observed;
    if (imu_trusted && !wrong) {
      log->log(
          LogLevel::Info,
          strprintf(
              "IMU: mounting confirmed by the motion (%.1f deg from the TF, %.0f deg of turning)",
              e->tf_error_deg, e->turned_deg));
      return;
    }
    log->log(LogLevel::Warn,
             strprintf("IMU: mounting learned from %.0f deg of turning%s: %.1f deg from the TF%s, "
                       "gravity scaled by %.3f; used from now on",
                       e->turned_deg,
                       e->yaw_observed ? " and the accelerations" : " (heading still the TF's)",
                       e->tf_error_deg,
                       e->accel_sign < 0 ? ", accelerometer reports gravity pointing down" : "",
                       e->accel_scale));
    // The gravity tracker keeps its state, turned into the new mounting, unless the accelerometer's
    // sign was wrong (its up was down) or it has none yet.
    const bool keep_gravity = imu_trusted && gravity.started() && e->accel_sign == imu_sign;
    const Eigen::Matrix3d dR =
        e->R_base_imu.matrix() * ext.T_base_imu.rotation().matrix().transpose();
    ext.T_base_imu = gtsam::Pose3(e->R_base_imu, ext.T_base_imu.translation());
    imu_sign = e->accel_sign;
    imu_scale = e->accel_scale;
    imu_buf.clear();
    imu_since_update.clear();
    gyro_att.clear();  // turned through the old mounting
    gyro_q = Eigen::Quaterniond::Identity();
    if (keep_gravity)
      gravity.rotate(dR);
    else
      gravity.reset();
    imu_trusted = true;
    for (const auto& m : imu_mount->recent()) use_imu(m, /*track_gravity=*/!keep_gravity);
  }

  /// Up direction in the base frame if the last 0.4 s of IMU data are quasi-static, else nullopt.
  std::optional<Eigen::Vector3d> quasi_static_up() const {
    if (imu_buf.size() < 40) return std::nullopt;
    std::vector<ImuSample> w(imu_buf.end() - 40, imu_buf.end());
    const auto g = estimate_gravity_from_imu(w, 0.03, 0.15);
    if (!g) return std::nullopt;
    return ext.T_base_imu.rotation().matrix() * g->up_body;  // IMU frame -> base frame
  }

  /// Aligns the odometry frame with gravity before its first frame: from the IMU at rest over its
  /// last half second, otherwise from the dominant ground plane of the frame's cloud. When neither
  /// is there yet and the IMU can level it, it waits for one (frames arrive before the IMU has half
  /// a second: a camera that starts with it, looking at the ceiling or held upright) and for a
  /// moment at rest (a hand-held sensor), skipping the frames meanwhile (false), up to `kWait` from
  /// the first frame. It does not wait when the frame shows the floor: a start that is already
  /// moving accelerates in a straight line without turning, which the IMU cannot tell from a tilt
  /// (S06 walks off at 0.8 m/s^2: 4.7 deg of pitch, a corridor climbed as a 2.3 m ramp). A ground
  /// robot's IMU is not used for it: until the accelerations have shown its heading, a TF that has
  /// it turned half around reads every tilt backwards (ADR-0058: the Summit's 0.46 deg became a
  /// start 0.9 deg off and a fourth floor in `prueba_03`), and a ground robot sees the ground plane
  /// it stands on.
  bool init_gravity(const std::vector<Eigen::Vector3f>& cloud, const gtsam::Pose3& T_base_sensor,
                    Timestamp stamp) {
    constexpr Timestamp kWindow = 500'000'000, kWait = 2'000'000'000;  // ns
    if (!first_frame) first_frame = stamp;
    const bool imu_levels = cfg.sensors.imu.enabled && cfg.robot.platform != Platform::Ground;
    std::optional<Eigen::Vector3d> up_base;
    if (imu_levels) {
      if (!imu_buf.empty() && imu_buf.back().stamp - imu_buf.front().stamp >= kWindow) {
        std::vector<ImuSample> w;
        for (const auto& m : imu_buf)
          if (m.stamp >= imu_buf.back().stamp - kWindow) w.push_back(m);
        if (const auto g = estimate_gravity_from_imu(w, 0.05, 0.5))
          up_base = ext.T_base_imu.rotation().matrix() * g->up_body;
      }
    }
    if (!up_base && !cloud.empty()) {
      const Eigen::Matrix3d R_bs = T_base_sensor.rotation().matrix();
      const auto up_sensor = estimate_up_from_ground_plane(
          cloud, R_bs.transpose() * Eigen::Vector3d::UnitZ(), cfg.runtime.seed);
      // Applied however small: the start has no attitude of its own to keep, and one cloud measures
      // the vertical within about 0.03 deg (a LiDAR sweep, about 0.15 deg: it averages the attitude
      // over 0.1 s of motion). Rounding a sub-degree tilt to level (a 0.5-degree threshold, as the
      // re-levelling has) left the simulator's 0.43-degree start tilt in every keyframe: S13 lost
      // 0.14 m of floor height on average and S07 3 seeds in 20.
      if (up_sensor) up_base = (R_bs * *up_sensor).normalized();
    }
    if (!up_base && imu_levels && stamp - first_frame < kWait) return false;
    gravity_done = true;
    if (!up_base) {
      log->log(
          LogLevel::Warn,
          imu_levels
              ? "no gravity alignment: the IMU did not rest for half a second at the start and "
                "no ground plane was seen; the map starts at the sensor's attitude"
              : "no gravity alignment: no ground plane was seen at the start; the map starts "
                "at the sensor's attitude");
      return true;
    }
    if (!odom_fe) return true;
    const gtsam::Pose3 T_init(rotation_from_up(*up_base), gtsam::Point3(0, 0, 0));
    odom_fe->reset(T_init);
    log->log(LogLevel::Info, "gravity alignment applied");
    return true;
  }

  /// Gravity-free vertical acceleration, averaged over the IMU samples since the last update: the
  /// specific force along up as the IMU's own gravity tracker gives it (base frame), so that it
  /// does not depend on the odometry's attitude (a start that could not be levelled leaves the
  /// odometry frame tilted: 20 deg read as -0.6 m/s^2, a lift that never was). None until that
  /// gravity can be trusted (at once after a start at rest, else after the tracker has converged).
  std::optional<double> take_vertical_accel() {
    if (!cfg.sensors.imu.floor_changes)
      imu_since_update.clear();  // the IMU gives the attitude only
    const auto up = gravity.up();
    if (imu_since_update.empty() || !up) {
      imu_since_update.clear();
      return std::nullopt;
    }
    const Eigen::Matrix3d R_bi = ext.T_base_imu.rotation().matrix();
    double sum = 0;
    for (const auto& m : imu_since_update) sum += (R_bi * m.accel).dot(*up) - kStandardGravity;
    const double az = sum / imu_since_update.size();
    if (debug_channel("IMU") && std::abs(az) > 0.5) {
      const auto& m0 = imu_since_update.front();
      const Eigen::Vector3d f = R_bi * m0.accel;
      log->log(LogLevel::Debug,
               strprintf("imu az=%.2f n=%zu first a=(%.2f %.2f %.2f) base=(%.2f %.2f %.2f) "
                         "up=(%.2f %.2f %.2f)",
                         az, imu_since_update.size(), m0.accel.x(), m0.accel.y(), m0.accel.z(),
                         f.x(), f.y(), f.z(), up->x(), up->y(), up->z()));
    }
    imu_since_update.clear();
    return az;
  }

  // ---------------- keyframes ----------------

  void insert_keyframe(const FrontendResult& r, std::vector<Eigen::Vector3f> cloud,
                       const gtsam::Pose3& T_base_sensor, Timestamp stamp,
                       std::optional<VisualKf> vis) {
    KeyframeInput in;
    in.id = next_kf++;
    in.stamp = stamp;
    in.session = session;
    in.floor = current_floor();
    // Restarted at this very frame (a floor event): its pose is the restart's, not the one the
    // frontend measured before it (raised by a ride's height since, Elevator-VIGS Mall-M: the
    // first keyframe of floor 1 was bridged 17.7 m under the next).
    const gtsam::Pose3 T_odom_base =
        restart_stamp == stamp && last_pose ? *last_pose : r.T_odom_base;
    in.T_odom_base = T_odom_base;
    in.tracking = r.quality.state;
    in.cloud = std::move(cloud);
    in.T_base_sensor = T_base_sensor;
    in.baro_height = baro_h;
    in.up_body = quasi_static_up();
    if (planar()) in.plane_up = plane_up();
    gtsam::Pose3 rel;
    std::optional<KeyframeId> odom_from;
    if (last_kf && last_kf_pose) {
      odom_from = last_kf;
      rel = last_kf_pose->between(T_odom_base);
    }
    // Every keyframe entering an existing graph must be connected: without an odometry predecessor
    // it is linked to the previous keyframe of that graph with the weak bridge factor.
    std::optional<KeyframeId> link = bridge_from;
    if (in.floor >= 0 && !odom_from && !link && last_kf_in_graph.count(key_floor(in.floor)))
      link = last_kf_in_graph[key_floor(in.floor)];
    auto added = backend.add_keyframe(in, odom_from, rel, link);
    if (!added) log->log(LogLevel::Error, added.error().message);
    kfs.push_back({in.id, stamp, session, in.floor, T_odom_base});
    last_kf = in.id;
    last_kf_pose = T_odom_base;
    if (in.floor >= 0) bridge_from.reset();
    ++since_loop;
    if (vis) visual_kf[in.id] = *vis;

    if (in.floor >= 0) {
      last_kf_on_floor[in.floor] = in.id;
      last_kf_in_graph[key_floor(in.floor)] = in.id;
      const bool arrived_continuously = open_tr.active && open_tr.continuous &&
                                        open_tr.to == in.floor && open_tr.from != open_tr.to;
      finish_open_transition(in);
      if (arrived_continuously) place_transit_keyframes(in);
      if (since_loop >= tuning.loop_cooldown_keyframes) {
        // `both`: the LiDAR first, then the camera if no LiDAR loop was accepted (until this fix
        // `both` behaved as `lidar`: the camera was never tried).
        const bool lidar_loops = modes.loop_closure != LoopSource::Camera && lidar_fe;
        const bool camera_loops = modes.loop_closure == LoopSource::Camera ||
                                  modes.loop_closure == LoopSource::Both || !lidar_fe;
        if (lidar_loops) try_lidar_loop(in);
        if (camera_loops && vis && since_loop >= tuning.loop_cooldown_keyframes)
          try_visual_loop(in, *vis);
      }
      register_place(in.id);
    } else if (pending.active) {
      pending.kfs.push_back(in.id);
      resolve_pending(in, vis);
    } else {
      transit_kfs.push_back(in.id);
    }
  }

  /// A transition that closed on the floor it started on, with continuous tracking, was not a floor
  /// change (a slope of the terrain, a curb): its keyframes belong to that floor and are chained to
  /// the last keyframe before it by the odometry they were tracked with.
  void absorb_transit_keyframes(FloorId floor) {
    if (transit_kfs.empty()) return;
    std::optional<KeyframeId> before;
    if (last_kf_on_floor.count(floor)) before = last_kf_on_floor[floor];
    const auto r =
        backend.promote_to_floor(transit_kfs, floor, before, /*bridge_is_odometry=*/true);
    if (!r) log->log(LogLevel::Error, r.error().message);
    for (KeyframeId id : transit_kfs) {
      for (auto& k : kfs)
        if (k.id == id) k.floor = floor;
      register_place(id);
      last_kf_on_floor[floor] = id;
      last_kf_in_graph[key_floor(floor)] = id;
    }
    transit_kfs.clear();
  }

  /// The keyframes tracked through a ramp or stairs stay in the map: those on the lower half of the
  /// climb (by altitude) go to the floor it left, chained forward from the last keyframe there; the
  /// rest go to the floor it arrived at (`first` is its first keyframe, already in its graph),
  /// chained backwards from it.
  void place_transit_keyframes(const KeyframeInput& first) {
    if (transit_kfs.empty() || tuning.backend.debug_disable_floor_gating)
      return;  // the control mode keeps them in its single graph
    const FloorId from = open_tr.from, to = first.floor;
    const double z_from = open_tr.kf_before ? backend.keyframe(*open_tr.kf_before).T_odom_base.z()
                                            : first.T_odom_base.z();
    const double z_mid = 0.5 * (z_from + first.T_odom_base.z());
    const bool rising = first.T_odom_base.z() >= z_from;
    std::size_t split = 0;  // transit_kfs[0, split) leave on `from`
    // A climb broken by a tracking loss splits at the restart: each part is chained by its own
    // odometry to the floor it touches (the motion during the loss is unknown).
    const auto session_change =
        open_tr.kf_before ? std::find_if(transit_kfs.begin(), transit_kfs.end(),
                                         [&](KeyframeId id) {
                                           return backend.keyframe(id).session !=
                                                  backend.keyframe(*open_tr.kf_before).session;
                                         })
                          : transit_kfs.end();
    if (session_change != transit_kfs.end()) {
      split = static_cast<std::size_t>(session_change - transit_kfs.begin());
    } else {
      while (split < transit_kfs.size()) {
        const double z = backend.keyframe(transit_kfs[split]).T_odom_base.z();
        if (rising ? z >= z_mid : z <= z_mid) break;
        ++split;
      }
    }
    const auto mid = transit_kfs.begin() + static_cast<std::ptrdiff_t>(split);
    const std::vector<KeyframeId> low(transit_kfs.begin(), mid), high(mid, transit_kfs.end());
    auto mark = [&](const std::vector<KeyframeId>& ids, FloorId floor) {
      for (KeyframeId id : ids) {
        for (auto& k : kfs)
          if (k.id == id)
            k.floor = floor;  // not a loop candidate: a scan taken mid-climb makes a poor place
      }
    };
    if (!low.empty() && open_tr.kf_before) {
      const auto r =
          backend.promote_to_floor(low, from, open_tr.kf_before, /*bridge_is_odometry=*/true);
      if (!r)
        log->log(LogLevel::Error, r.error().message);
      else
        mark(low, from);
    }
    const std::vector<KeyframeId>& rest = open_tr.kf_before ? high : transit_kfs;
    if (!rest.empty()) {
      const auto r = backend.promote_transit_before(rest, to, first.id);
      if (!r)
        log->log(LogLevel::Error, r.error().message);
      else
        mark(rest, to);
    }
    transit_kfs.clear();
  }

  void finish_open_transition(const KeyframeInput& first_on_new_floor) {
    if (!open_tr.active || first_on_new_floor.floor != open_tr.to) return;
    TransitionRecord t;
    t.type = open_tr.type;
    t.from = open_tr.from;
    t.to = open_tr.to;
    t.kf_before = open_tr.kf_before.value_or(first_on_new_floor.id);
    t.kf_after = first_on_new_floor.id;
    t.delta_z = open_tr.dz;
    // A gap in the tracking leaves the climb measured (the frontend restarted at the last pose) but
    // the horizontal motion during the loss unknown: the relative pose is kept, with a wide
    // covariance.
    t.continuous_tracking = open_tr.continuous && !open_tr.tracking_gap;
    t.covariance = transition_covariance(open_tr.type, t.continuous_tracking);
    if (open_tr.continuous && open_tr.kf_before) {
      // In the gravity-aligned (yaw only) frames, which is what Pose4 and the fusion assume:
      // expressed in the tilted body frame of a keyframe that already stands on the foot of a ramp,
      // the climb would leak into x and the height come out wrong.
      const auto yaw_frame = [](const gtsam::Pose3& T) {
        const auto R = T.rotation().matrix();
        return gtsam::Pose3(gtsam::Rot3::Yaw(std::atan2(R(1, 0), R(0, 0))), T.translation());
      };
      const gtsam::Pose3 rel = yaw_frame(backend.keyframe(*open_tr.kf_before).T_odom_base)
                                   .between(yaw_frame(first_on_new_floor.T_odom_base));
      const auto R = rel.rotation().matrix();
      t.relative = std::array<double, 4>{std::atan2(R(1, 0), R(0, 0)), rel.x(), rel.y(),
                                         open_tr.dz_from_sensor ? open_tr.dz : rel.z()};
    }
    transitions.push_back(t);
    open_tr.active = false;
  }

  static Eigen::Matrix4d transition_covariance(TransitionType type, bool continuous) {
    // (yaw, x, y, z) standard deviations per transition kind.
    double sy = 0.05, sx = 0.10, sz = 0.10;
    switch (type) {
      case TransitionType::Stairs:  // z: LiDAR drifts in stairwells
      case TransitionType::Ramp:
        sy = 0.03;
        sx = 0.10;
        sz = 0.15;
        break;
      case TransitionType::Elevator:
        sy = 0.03;
        sx = 0.05;
        sz = 0.20;
        break;  // z from the barometer / IMU
      case TransitionType::Aerial:
        sy = 0.05;
        sx = 0.15;
        sz = 0.15;
        break;
      case TransitionType::Unknown:
        sy = 3.14;
        sx = 50.0;
        sz = 1.0;
        break;
    }
    if (!continuous && type != TransitionType::Unknown) {
      sy *= 5;
      sx *= 5;
    }
    Eigen::Matrix4d C = Eigen::Matrix4d::Zero();
    C(0, 0) = sy * sy;
    C(1, 1) = sx * sx;
    C(2, 2) = sx * sx;
    C(3, 3) = sz * sz;
    return C;
  }

  /// Adds the keyframe to the place-recognition databases of its own floor only.
  void register_place(KeyframeId id) {
    const KeyframeInput& in = backend.keyframe(id);
    if (in.floor < 0) return;
    if (lidar_fe && !in.cloud.empty() && modes.loop_closure != LoopSource::Camera) {
      ScanContext sc = make_scan_context(in.cloud, tuning.scan_context);
      sc_for(in.floor).add(in.id, sc);
      sc_of[in.id] = std::move(sc);
    }
    const auto it = visual_kf.find(id);
    if (it != visual_kf.end() && tuning.vocabulary) {
      std::vector<Descriptor> d;
      for (const auto& k : it->second.keypoints) d.push_back(k.descriptor);
      bow_for(in.floor).add(in.id, in.session, in.stamp, tuning.vocabulary->transform(d));
    }
  }

  // ---------------- place recognition ----------------

  /// PnP verification of a visual candidate: relative base pose T_cand_cur, or nullopt.
  std::optional<gtsam::Pose3> verify_visual(const VisualKf& vis, KeyframeId cand_id) const {
    const auto it = visual_kf.find(cand_id);
    if (it == visual_kf.end()) return std::nullopt;
    const VisualKf& cand = it->second;
    std::vector<PnPCorrespondence> corr;
    for (size_t i = 0; i < vis.keypoints.size(); ++i) {
      int best = -1, best_d = 1 << 20, second = 1 << 20;
      for (size_t j = 0; j < cand.keypoints.size(); ++j) {
        if (!cand.p_cam[j]) continue;
        const int dist =
            hamming_distance(vis.keypoints[i].descriptor, cand.keypoints[j].descriptor);
        if (dist < best_d) {
          second = best_d;
          best_d = dist;
          best = static_cast<int>(j);
        } else if (dist < second)
          second = dist;
      }
      if (best < 0 || best_d > tuning.visual.tracker.max_hamming ||
          best_d > tuning.visual.tracker.ratio * second)
        continue;
      const std::optional<Eigen::Vector3d>& p_best = cand.p_cam[best];
      if (!p_best) continue;
      corr.push_back({*p_best, Eigen::Vector2d(vis.keypoints[i].u, vis.keypoints[i].v)});
    }
    const auto T_cam_obj =
        verify_loop_pnp(corr, vis.K, tuning.visual_loop_min_inliers,
                        tuning.visual.tracker.ransac_reprojection_px * pixel_scale(vis.K), 300);
    if (!T_cam_obj) return std::nullopt;
    // T_cam_obj maps candidate-camera points to the current camera: T_curcam_candcam.
    const gtsam::Pose3 T_candcam_curcam = T_cam_obj->inverse();
    return ext.T_base_camera * T_candcam_curcam * ext.T_base_camera.inverse();
  }

  BowVector bow_of(const VisualKf& vis) const {
    std::vector<Descriptor> d;
    d.reserve(vis.keypoints.size());
    for (const auto& k : vis.keypoints) d.push_back(k.descriptor);
    return tuning.vocabulary->transform(d);
  }

  /// Visual loop closure: BoW candidates (current floor only), PnP verification with >= 30 inliers.
  void try_visual_loop(const KeyframeInput& cur, const VisualKf& vis) {
    if (!tuning.vocabulary) return;
    const auto cands = bow_for(cur.floor).query(bow_of(vis), cur.session, cur.stamp,
                                                tuning.bow_exclude_recent_s, tuning.bow_min_score,
                                                static_cast<size_t>(tuning.bow_max_candidates));
    if (debug_channel("LOOPS") && !cands.empty())
      log->log(LogLevel::Debug,
               strprintf("visual loop kf %lu: %zu BoW candidates, best score %.3f",
                         static_cast<unsigned long>(cur.id), cands.size(), cands.front().score));
    for (const auto& c : cands) {
      if (!backend.has_keyframe(c.id)) continue;
      if (backend.floor_of(c.id) != cur.floor) {
        ++cross_floor_seen;
        if (!tuning.backend.debug_disable_floor_gating)
          continue;  // belt and braces (the control mode removes it)
      }
      auto rel = verify_visual(vis, c.id);
      if (debug_channel("LOOPS"))
        log->log(LogLevel::Debug,
                 strprintf("visual loop kf %lu -> %lu: PnP %s", static_cast<unsigned long>(cur.id),
                           static_cast<unsigned long>(c.id), rel ? "verified" : "rejected"));
      if (!rel) continue;
      if (modes.verify_loop_with_lidar && lidar_fe && !cur.cloud.empty()) {
        // Camera loop candidates are verified (and refined) with LiDAR registration.
        const FloorGraph* fg = cur.floor >= 0 && backend.floor_of(c.id) == cur.floor
                                   ? &backend.floor(cur.floor)
                                   : nullptr;
        const auto refined = verify_lidar(cur, c.id, 0.0, fg, &*rel);
        if (!refined) continue;
        rel = refined;
      }
      if (backend.add_loop(c.id, cur.id, *rel)) {
        since_loop = 0;
        log->log(LogLevel::Info, "visual loop closure accepted");
        break;
      }
    }
  }

  /// GICP verification of a LiDAR candidate against a submap around it.
  std::optional<gtsam::Pose3> verify_lidar(const KeyframeInput& cur, KeyframeId cand_id, double yaw,
                                           const FloorGraph* fg,
                                           const gtsam::Pose3* init_rel_base = nullptr) const {
    const gtsam::Pose3 T_lc = ext.T_base_lidar;
    std::vector<Eigen::Vector3f> submap;
    gtsam::Pose3 init_graph;
    bool have_graph_init = false;
    if (fg && fg->has(cand_id) && fg->has(cur.id)) {
      const gtsam::Pose3 Xc = fg->pose(cand_id);
      const auto ids = fg->keyframe_ids();
      const auto it = std::find(ids.begin(), ids.end(), cand_id);
      if (it == ids.end()) return std::nullopt;
      const long pos = it - ids.begin();
      for (long k = std::max<long>(0, pos - tuning.loop_submap_radius);
           k <= std::min<long>(static_cast<long>(ids.size()) - 1, pos + tuning.loop_submap_radius);
           ++k) {
        const auto* cl = backend.cloud(ids[k]);
        if (!cl || backend.floor_of(ids[k]) != backend.floor_of(cand_id)) continue;
        const gtsam::Pose3 T_cand_k_lidar = (Xc * T_lc).between(fg->pose(ids[k]) * T_lc);
        const auto w = transform_points(*cl, T_cand_k_lidar.matrix());
        submap.insert(submap.end(), w.begin(), w.end());
      }
      init_graph = (Xc * T_lc).between(fg->pose(cur.id) * T_lc);
      have_graph_init = true;
    } else {
      const auto* cl = backend.cloud(cand_id);
      if (!cl) return std::nullopt;
      submap = *cl;
    }
    const gtsam::Pose3 init_yaw(gtsam::Rot3::Yaw(yaw), gtsam::Point3(0, 0, 0));
    LidarFrontend::RegistrationOutput best;
    bool have = false;
    std::vector<gtsam::Pose3> inits{init_yaw};
    if (have_graph_init) inits.insert(inits.begin(), init_graph);
    if (init_rel_base)
      inits = {T_lc.inverse() * (*init_rel_base) * T_lc};  // the visual estimate as the only guess
    for (const gtsam::Pose3& init : inits) {
      auto out = LidarFrontend::register_clouds(submap, cur.cloud, init, tuning.lidar);
      if (!have || out.quality.fitness > best.quality.fitness) {
        best = out;
        have = true;
      }
    }
    // The residual of a right registration depends on the sensor and the scene (an HDL-64 in a
    // street leaves more than a LiDAR indoors): above the fixed bound, the loop is compared with
    // the residual of this scan against its own predecessors, a registration known to be right
    // (ADR-0057).
    const bool fits =
        have && best.quality.fitness >= tuning.loop_min_fitness &&
        best.quality.state != TrackingState::Degraded;  // degraded: ambiguous geometry
    double ref = -1.0;
    if (fits && best.mean_error > tuning.loop_max_mean_error) {
      if (const auto r = reference_error(cur, fg)) ref = *r;
    }
    if (debug_channel("LOOPS"))
      log->log(LogLevel::Debug,
               strprintf("loop verify kf %lu -> %lu: fitness=%.3f (min %.3f) mean_error=%.3f (max "
                         "%.3f, or %.1f x reference %.3f) degraded=%d",
                         static_cast<unsigned long>(cur.id), static_cast<unsigned long>(cand_id),
                         have ? best.quality.fitness : -1.0, tuning.loop_min_fitness,
                         have ? best.mean_error : -1.0, tuning.loop_max_mean_error,
                         tuning.loop_max_error_ratio, ref,
                         have && best.quality.state == TrackingState::Degraded));
    if (!fits) return std::nullopt;
    if (best.mean_error > std::max(tuning.loop_max_mean_error, tuning.loop_max_error_ratio * ref))
      return std::nullopt;
    return T_lc * best.T_target_source * T_lc.inverse();
  }

  /// Mean residual of `cur` registered against the submap of the keyframes before it, from their
  /// graph poses: what a right registration of this scan leaves in this place. Computed once per
  /// keyframe; none for the first keyframe of a floor.
  std::optional<double> reference_error(const KeyframeInput& cur, const FloorGraph* fg) const {
    if (ref_error_kf == cur.id) return ref_error;
    ref_error_kf = cur.id;
    ref_error.reset();
    if (!fg || !fg->has(cur.id)) return ref_error;
    const auto ids = fg->keyframe_ids();
    const auto it = std::find(ids.begin(), ids.end(), cur.id);
    if (it == ids.end() || it == ids.begin()) return ref_error;
    const gtsam::Pose3 T_lc = ext.T_base_lidar;
    const long prev = it - ids.begin() - 1;
    const gtsam::Pose3 Xp = fg->pose(ids[prev]) * T_lc;
    std::vector<Eigen::Vector3f> submap;
    // As many keyframes as the loop submap (2 * radius + 1), all on the side already mapped.
    for (long k = std::max<long>(0, prev - 2L * tuning.loop_submap_radius); k <= prev; ++k) {
      const auto* cl = backend.cloud(ids[k]);
      if (!cl) continue;
      const auto w = transform_points(*cl, Xp.between(fg->pose(ids[k]) * T_lc).matrix());
      submap.insert(submap.end(), w.begin(), w.end());
    }
    const auto out = LidarFrontend::register_clouds(
        submap, cur.cloud, Xp.between(fg->pose(cur.id) * T_lc), tuning.lidar);
    if (out.quality.state != TrackingState::Lost && out.mean_error > 0.0)
      ref_error = out.mean_error;
    return ref_error;
  }
  mutable KeyframeId ref_error_kf = std::numeric_limits<KeyframeId>::max();
  mutable std::optional<double> ref_error;

  void try_lidar_loop(const KeyframeInput& cur) {
    if (cur.cloud.empty()) return;
    const ScanContext sc = make_scan_context(cur.cloud, tuning.scan_context);
    const auto cands = sc_for(cur.floor).query(sc);
    const FloorGraph& fg = backend.floor(cur.floor);
    if (debug_channel("LOOPS") && !cands.empty())
      log->log(LogLevel::Debug, strprintf("loop kf %lu: %zu scan-context candidates",
                                          static_cast<unsigned long>(cur.id), cands.size()));
    for (const auto& c : cands) {
      if (!backend.has_keyframe(c.id)) continue;
      if (backend.floor_of(c.id) != cur.floor) {
        ++cross_floor_seen;
        if (!tuning.backend.debug_disable_floor_gating) continue;
      }
      const auto rel = verify_lidar(cur, c.id, c.match.yaw, &fg);
      if (!rel) continue;
      const bool accepted = backend.add_loop(c.id, cur.id, *rel).ok();
      if (debug_channel("LOOPS"))
        log->log(LogLevel::Debug,
                 strprintf("loop kf %lu -> %lu: backend %s", static_cast<unsigned long>(cur.id),
                           static_cast<unsigned long>(c.id), accepted ? "accepted" : "rejected"));
      if (accepted) {
        since_loop = 0;
        log->log(LogLevel::Info, "loop closure accepted");
        break;
      }
    }
  }

  // ---------------- sessions with an unknown floor ----------------

  void start_pending(FloorId from_floor) {
    pending = Pending{};
    pending.active = true;
    pending.from_floor = from_floor;
    pending.kf_before = last_kf_on_floor.count(from_floor)
                            ? std::optional<KeyframeId>(last_kf_on_floor[from_floor])
                            : std::nullopt;
    pending.in_transition = mgr.in_observed_transition();
  }

  /// Tries to place the pending session on an existing floor (relocalization); otherwise opens a
  /// new floor after enough unlocalized keyframes (transition recorded as `Unknown`).
  void resolve_pending(const KeyframeInput& cur, const std::optional<VisualKf>& vis) {
    // Lost in the middle of stairs or a ramp the odometry was measuring: the frontend restarted at
    // the last pose, so the height is still continuous and the climb goes on. The session is not an
    // unobservable floor change; the transition decides its floor when it ends (on_floor_event).
    // Relocalizing here would snap a half-way keyframe onto the floor it looks like. So would it in
    // a ride the IMU or the barometer is measuring, which an odometry-opened transition becomes
    // when the odometry is lost in the cabin: a glass cabin looks onto the floor it left
    // (Elevator-VIGS Mall-E: relocalized on floor 0 half-way up, the ride was dropped).
    if (mgr.state() == FloorState::InTransition) return;
    // With a barometer or an IMU a floor change cannot go unnoticed, so a tracking loss is not one:
    // the session stays on its floor; with `multifloor.enabled: false` there is no other floor to
    // go to. Otherwise the change is unobservable and every floor is a relocalization candidate
    // (and a failure to relocalize opens a new floor).
    const bool height_sensors = cfg.sensors.barometer.enabled ||
                                (cfg.sensors.imu.enabled && cfg.sensors.imu.floor_changes) ||
                                !cfg.multifloor.enabled;
    // Relocalization may legitimately look at every floor: it decides which floor this session is
    // on.
    if (vis && tuning.vocabulary) {
      const BowVector q = bow_of(*vis);
      for (auto& [floor_key, db] : bow_db) {
        const auto cands = db.query(q, cur.session, cur.stamp, 0.0, tuning.bow_min_score,
                                    static_cast<size_t>(tuning.bow_max_candidates));
        for (const auto& c : cands) {
          const auto rel = verify_visual(*vis, c.id);
          if (!rel) continue;
          const FloorId target = backend.floor_of(c.id);
          if (target < 0 || (height_sensors && target != pending.from_floor)) continue;
          place_pending_on(target, c.id, cur.id, *rel);
          return;
        }
      }
    } else if (lidar_fe && !cur.cloud.empty()) {
      const ScanContext sc = make_scan_context(cur.cloud, tuning.scan_context);
      for (auto& [floor_key, db] : sc_db) {
        for (const auto& c : db.query(sc)) {
          const auto rel = verify_lidar(cur, c.id, c.match.yaw, nullptr);
          if (!rel) continue;
          const FloorId target = backend.floor_of(c.id);
          if (target < 0 || (height_sensors && target != pending.from_floor)) continue;
          place_pending_on(target, c.id, cur.id, *rel);
          return;
        }
      }
    }
    if (static_cast<int>(pending.kfs.size()) >= tuning.pending_keyframes_for_new_floor)
      give_up_pending();
  }

  /// Relocalization gave up (enough keyframes without a match, or the end of the run): with height
  /// sensors the session stays on its floor, otherwise the change was unobservable and it gets a
  /// new floor.
  void give_up_pending() {
    const bool height_sensors = cfg.sensors.barometer.enabled ||
                                (cfg.sensors.imu.enabled && cfg.sensors.imu.floor_changes) ||
                                !cfg.multifloor.enabled;
    if (height_sensors) {
      // Not relocalized, but the height sensors saw no change: a new session of the same floor.
      const FloorId f = pending.from_floor;
      mgr.set_floor(f);
      commit_pending(f, last_kf_on_floor.count(f) ? std::optional<KeyframeId>(last_kf_on_floor[f])
                                                  : std::nullopt);
      log->log(LogLevel::Info,
               cfg.multifloor.enabled
                   ? "relocalization failed with height sensors active: session kept on its floor"
                   : "relocalization failed: session kept on the only level (multifloor disabled)");
      return;
    }
    {
      const FloorId f = mgr.open_new_floor();
      TransitionRecord t;
      t.type = TransitionType::Unknown;
      t.from = pending.from_floor;
      t.to = f;
      t.kf_before = pending.kf_before.value_or(pending.kfs.front());
      t.kf_after = pending.kfs.front();
      t.continuous_tracking = false;
      t.delta_z = cfg.multifloor.floor_height_prior;  // nullopt without a prior
      t.covariance = transition_covariance(TransitionType::Unknown, false);
      transitions.push_back(t);
      commit_pending(f, std::nullopt);
      log->log(LogLevel::Warn, "relocalization failed: opened a new floor (unobservable change)");
    }
  }

  /// End of the run: nothing is left undecided, so that every keyframe ends in the map (audit
  /// AUD-022: a transition still open when the run ended left 307 of 328 keyframes out). An open
  /// transition is decided as when its time limit runs out; a session still pending is given up on
  /// as when relocalization gives up.
  void resolve_at_end() {
    if (mgr.state() == FloorState::InTransition) {
      const FloorEvent ev = mgr.finish();
      if (ev.kind == FloorEvent::Kind::TransitionEnded) {
        log->log(LogLevel::Info, "end of the run with a floor transition open: decided now");
        on_floor_event(ev);
        // No keyframe of the arrival floor will come: the keyframes of the climb are its first
        // ones.
        if (!transit_kfs.empty() && ev.from != ev.to) {
          const KeyframeId first = transit_kfs.front();
          absorb_transit_keyframes(ev.to);
          finish_open_transition(backend.keyframe(first));
        }
      }
    }
    if (pending.active && !pending.kfs.empty()) {
      log->log(LogLevel::Info, "end of the run with a session not relocalized yet");
      give_up_pending();
    }
  }

  void place_pending_on(FloorId target, KeyframeId cand, KeyframeId cur, const gtsam::Pose3& rel) {
    mgr.set_floor(target);
    commit_pending(target,
                   last_kf_on_floor.count(target) ? std::optional<KeyframeId>(cand) : std::nullopt);
    if (backend.add_loop(cand, cur, rel)) {
      since_loop = 0;
      log->log(LogLevel::Info, "relocalized: session placed on its floor");
    }
  }

  void commit_pending(FloorId floor, std::optional<KeyframeId> bridge) {
    const auto r = backend.promote_to_floor(pending.kfs, floor, bridge);
    if (!r) log->log(LogLevel::Error, r.error().message);
    for (KeyframeId id : pending.kfs) {
      for (auto& k : kfs)
        if (k.id == id) k.floor = floor;
      register_place(id);
      last_kf_on_floor[floor] = id;
      last_kf_in_graph[key_floor(floor)] = id;
    }
    pending = Pending{};
  }

  // ---------------- tracking loss and floor events ----------------

  /// Restarts the active frontend at the corrected pose and tells the floor manager that the height
  /// jump of the restart is a correction, not motion.
  /// `pose` with its roll and pitch replaced by the measured gravity, yaw and position kept: from
  /// the IMU when it is at rest, else (if `ground_plane`) from the horizontal plane of the current
  /// frame. Unchanged when neither is available. A restarted frontend otherwise keeps the attitude
  /// error of the pose it restarts from, and every keyframe of the new floor inherits it as a tilt
  /// that the 4-DoF fusion cannot remove (ADR-0033).
  static constexpr double kMinGroundRelevelDeg =
      0.5;  ///< below this a ground-plane correction is noise

  gtsam::Pose3 relevel(const gtsam::Pose3& pose, bool ground_plane) const {
    std::optional<Eigen::Vector3d> up_base;
    const char* source = "the IMU";
    bool up_from_imu = false;
    if (cfg.sensors.imu.enabled && imu_buf.size() >= 2) {
      // The last second of IMU, if the robot was really at rest (strict: a robot that just stopped
      // walking or a cabin that starts moving still accelerates by a few tenths of m/s^2).
      std::vector<ImuSample> w;
      for (auto it = imu_buf.rbegin();
           it != imu_buf.rend() && (imu_buf.back().stamp - it->stamp) * 1e-9 <= 1.0; ++it)
        w.push_back(*it);
      if (w.size() >= 20 && (imu_buf.back().stamp - w.back().stamp) * 1e-9 >= 0.9)
        if (const auto g = estimate_gravity_from_imu(w, 0.02, 0.15)) {
          up_base = ext.T_base_imu.rotation().matrix() * g->up_body;
          up_from_imu = true;
        }
    }
    if (!up_base && ground_plane && (cur_rgbd || cur_scan)) {
      const gtsam::Pose3& T_bs = cur_rgbd ? ext.T_base_camera : ext.T_base_lidar;
      const std::vector<Eigen::Vector3f> cloud =
          cur_rgbd ? depth_to_cloud(cur_rgbd->depth, cur_rgbd->K, 4, tuning.visual.tracker.depth)
                   : cur_scan->points;
      const Eigen::Matrix3d R_bs = T_bs.rotation().matrix();
      // Up as currently estimated, in the sensor frame; the plane must be within 10 deg of it.
      const Eigen::Vector3d up_sensor_now =
          R_bs.transpose() * pose.rotation().matrix().transpose() * Eigen::Vector3d::UnitZ();
      if (const auto u =
              estimate_up_from_ground_plane(cloud, up_sensor_now, cfg.runtime.seed, 10.0))
        up_base = R_bs * *u;
      source = "the ground plane";
    }
    if (!up_base) return pose;
    const gtsam::Rot3 R = level_to(pose.rotation(), *up_base);
    const double correction_deg =
        gtsam::Rot3::Logmap(pose.rotation().between(R)).norm() * 180.0 / M_PI;
    // A ground plane can be the wrong surface (a slanted one taken for the floor): its correction
    // is bounded by a plausible attitude drift. The IMU, strictly at rest, is trusted further (a
    // LiDAR odometry inside an elevator cabin can drift several degrees).
    const double bound = up_from_imu ? tuning.max_relevel_imu_deg : tuning.max_relevel_deg;
    // One cloud gives the vertical within 0.1-0.3 deg: a smaller correction is the fit's own noise,
    // and applying it would replace a good attitude by a slightly worse one at every restart (S11
    // lost 3 of 20 seeds that way once the ground was found reliably below the sensor). Only a real
    // drift is corrected.
    if (!up_from_imu && correction_deg < kMinGroundRelevelDeg) return pose;
    if (correction_deg > bound) {
      log->log(
          LogLevel::Info,
          strprintf("frontend restart: gravity from %s %.1f deg away from the attitude, not used",
                    source, correction_deg));
      return pose;
    }
    log->log(LogLevel::Info, strprintf("frontend restart re-levelled by %.2f deg (gravity from %s)",
                                       correction_deg, source));
    return gtsam::Pose3(R, pose.translation());
  }

  /// Restarts the odometry at the corrected last pose, re-levelled. `climb_dz` is a vertical motion
  /// the robot is assumed to have made meanwhile: it raises the restart pose and, unlike the
  /// backend's correction, is not compensated in the manager, so the altitude moves by it.
  /// `rebase_dz` raises it too, as a height the manager has already counted (a ride the IMU or the
  /// barometer measured): the altitude stays, and the odometry's z is the height again.
  void restart_frontend(double climb_dz = 0.0, bool level = true, double rebase_dz = 0.0) {
    if (!odom_fe || !last_pose) return;
    gtsam::Pose3 start = corrected_last_pose();
    if (const auto turn = gyro_turn(last_pose_stamp, frame_stamp)) {
      if (debug_channel("TRACK"))
        log->log(LogLevel::Debug,
                 strprintf("restart: the gyroscope turned the body by %.1f deg in %.1f s blind",
                           gtsam::Rot3::Logmap(*turn).norm() * 180.0 / M_PI,
                           (frame_stamp - last_pose_stamp) * 1e-9));
      start = gtsam::Pose3(start.rotation() * *turn, start.translation());
    }
    // On stairs or a ramp the visible "ground" is the slope: only the IMU may re-level there.
    const gtsam::Pose3 corrected = level ? relevel(start, !mgr.in_observed_transition()) : start;
    mgr.odometry_rebased(corrected.z() - last_pose->z() + rebase_dz);
    const gtsam::Pose3 restart(corrected.rotation(),
                               corrected.translation() + gtsam::Point3(0, 0, climb_dz + rebase_dz));
    odom_fe->reset(restart);
    last_pose = restart;
    last_pose_stamp = frame_stamp;
    restart_stamp = frame_stamp;
  }

  /// The latest pose expressed in the optimized frame: the last keyframe's optimized pose plus the
  /// motion since it. Restarting a frontend from this (rather than from the raw odometry pose) does
  /// not carry the drift that the backend has already corrected into the new session or floor.
  gtsam::Pose3 corrected_last_pose() const {
    if (!last_pose) return gtsam::Pose3();
    if (!last_kf || !backend.has_keyframe(*last_kf)) return *last_pose;
    const KeyframeInput& k = backend.keyframe(*last_kf);
    if (k.floor < 0) return *last_pose;
    const gtsam::Pose3 c = backend.pose_odom(*last_kf) * k.T_odom_base.between(*last_pose);
    if (debug_channel("TRACK")) {
      const auto rpy = c.rotation().rpy();
      log->log(
          LogLevel::Debug,
          strprintf(
              "restart raw=(%.2f %.2f %.2f) corrected=(%.2f %.2f %.2f) rpy_deg=(%.2f %.2f %.2f)",
              last_pose->x(), last_pose->y(), last_pose->z(), c.x(), c.y(), c.z(), rpy.x() * 57.3,
              rpy.y() * 57.3, rpy.z() * 57.3));
    }
    return c;
  }

  void handle_lost(Timestamp stamp) {
    last_fe_stamp = 0;  // the next rotation for the IMU mounting starts at the next valid pose
    if (lost_since == 0) lost_since = stamp;
    const double lost_s = (stamp - lost_since) * 1e-9;
    if (mgr.in_unobserved_transition())
      return;  // the lift explains the loss; the frontend restarts when it ends
    if (cfg.multifloor.split_on_tracking_loss && lost_s > cfg.multifloor.tracking_loss_timeout &&
        last_pose && odom_fe) {
      // New session, restarting at the last known pose; its floor is decided by relocalization. A
      // further loss while a session is already pending restarts the frontend but keeps the list.
      if (!pending.active) start_pending(mgr.last_floor());
      // Restart from the last Good pose (the degraded frames before a loss drift, mostly in
      // attitude).
      Timestamp blind_since = lost_since;
      if (last_good &&
          (lost_since - last_good->first) * 1e-9 < cfg.multifloor.tracking_loss_timeout) {
        last_pose = last_good->second;
        last_pose_stamp = last_good->first;
        blind_since = last_good->first;
      }
      // Lost in the middle of stairs or a ramp: the climb did not stop while the camera was blind.
      restart_frontend(mgr.climb_rate() * (stamp - blind_since) * 1e-9);
      last_kf.reset();
      last_kf_pose.reset();
      ++session;
      lost_since = 0;
      log->log(LogLevel::Warn, "tracking lost: new session (floor pending)");
    }
  }

  void on_floor_event(const FloorEvent& ev) {
    if (ev.kind == FloorEvent::Kind::TransitionStarted)
      log->log(LogLevel::Info,
               std::string("floor transition started") +
                   (ev.vertical_unobserved ? " (height change not seen by the odometry)" : ""));
    if (ev.kind == FloorEvent::Kind::TransitionEnded)
      log->log(LogLevel::Info, std::string("floor transition ended: ") + to_string(ev.type) + " " +
                                   std::to_string(ev.from) + "->" + std::to_string(ev.to) +
                                   " dz=" + std::to_string(ev.delta_z));
    if (ev.kind == FloorEvent::Kind::TransitionStarted && ev.vertical_unobserved && last_pose &&
        odom_fe) {
      // The height change is invisible to the odometry (elevator): restart the frontend now, so
      // that it cannot match the old floor's map against an identical-looking arrival floor. Not
      // re-levelled here: the cabin is about to accelerate; it is at the arrival (TransitionEnded).
      restart_frontend(0.0, /*level=*/false);
      last_kf.reset();
      last_kf_pose.reset();
      if (pending.active && pending.kfs.empty()) pending = Pending{};
      return;
    }
    if (ev.kind != FloorEvent::Kind::TransitionEnded) return;
    if (ev.from == ev.to && ev.type == TransitionType::Unknown) {
      // The height came back to where the transition started: no floor change, nothing to record.
      // Its keyframes join the floor (a pending session keeps waiting for its relocalization, as
      // before). Ended by a ride the IMU measures from now on, a session lost before it was on
      // the floor the ride leaves: the ride's end would otherwise place it on the floor it reaches.
      if (ev.ride_follows && pending.active)
        commit_pending(pending.from_floor,
                       last_kf_on_floor.count(pending.from_floor)
                           ? std::optional<KeyframeId>(last_kf_on_floor[pending.from_floor])
                           : std::nullopt);
      if (!pending.active) absorb_transit_keyframes(ev.to);
      return;
    }
    if (ev.from == ev.to && ev.continuous_tracking && !ev.needs_frontend_reset && !pending.active) {
      log->log(LogLevel::Info,
               "height change was not a floor change: keeping its keyframes on floor " +
                   std::to_string(ev.to));
      absorb_transit_keyframes(ev.to);
      return;
    }
    // A ramp or stairs tracked continuously keeps its keyframes for the floor it arrives at. Not a
    // ride the odometry did not see, even tracked: the keyframes in the cabin have the height it
    // left (Elevator-VIGS Mall-M: one was placed on the floor reached, 17.7 m under the next).
    if (!(ev.continuous_tracking && ev.from != ev.to) || ev.needs_frontend_reset)
      transit_kfs.clear();
    if (odom_fe)
      odom_fe
          ->request_keyframe();  // the new floor needs its first keyframe even if the robot hovers
    open_tr = OpenTransition{};
    open_tr.active = true;
    open_tr.type = ev.type;
    open_tr.from = ev.from;
    open_tr.to = ev.to;
    open_tr.dz = ev.delta_z;
    open_tr.continuous = ev.continuous_tracking;
    open_tr.dz_from_sensor = ev.needs_frontend_reset;
    if (last_kf_on_floor.count(ev.from)) open_tr.kf_before = last_kf_on_floor[ev.from];
    // Returning to a floor that already has keyframes: link with a weak factor in its own graph.
    if (last_kf_on_floor.count(ev.to)) bridge_from = last_kf_on_floor[ev.to];
    if (ev.needs_frontend_reset && last_pose && odom_fe) {
      // The odometry saw no vertical motion (elevator, manual change): restart the frontend at the
      // same pose (clearing the old floor's map), raised by the height the manager measured and
      // told it is no motion (it is not counted twice). Left at the height before the ride, the
      // odometry's z linked a floor reached again by the stairs to its keyframes 15 m off
      // (Elevator-VIGS Office4: back on floor 0, its loop closures with the start were rejected).
      restart_frontend(0.0, true, ev.reset_dz);
      last_kf.reset();
      last_kf_pose.reset();
      ++session;
    }
    if (pending.active && pending.in_transition && ev.continuous_tracking && ev.from != ev.to) {
      // Lost and restarted in the middle of this climb: its keyframes are the rest of the climb, so
      // they are transit keyframes, placed by height when the first keyframe of the new floor
      // arrives (which also records the transition from the last keyframe before the climb).
      transit_kfs.insert(transit_kfs.end(), pending.kfs.begin(), pending.kfs.end());
      pending = Pending{};
      open_tr.tracking_gap = true;
    }
    if (pending.active) {
      // The manager knows the floor now: the pending keyframes belong to it.
      if (pending.kfs.empty()) {
        pending = Pending{};
      } else {
        const KeyframeId first = pending.kfs.front();
        commit_pending(ev.to, last_kf_on_floor.count(ev.to)
                                  ? std::optional<KeyframeId>(last_kf_on_floor[ev.to])
                                  : std::nullopt);
        finish_open_transition(backend.keyframe(first));
      }
    }
  }

  void feed_manager(const FrontendResult& r, Timestamp stamp) {
    // `multifloor.enabled: false`: one level, the floor manager never leaves floor 0 (an outdoor
    // drive has slopes, not floors). A tracking loss still splits sessions (handle_lost).
    if (!cfg.multifloor.enabled) return;
    FloorSample fs;
    fs.stamp = stamp;
    fs.tracking_ok = r.valid;
    if (r.valid) {
      fs.position = r.T_odom_base.translation();
      fs.up_axis = Eigen::Vector3d(r.T_odom_base.rotation().matrix().col(2));
    }
    fs.baro_height = baro_h;
    fs.imu_vertical_accel = take_vertical_accel();
    on_floor_event(mgr.update(fs));
  }
};

SlamSystem::SlamSystem(const Config& c, const Extrinsics& e, const SystemTuning& t,
                       std::shared_ptr<Logger> l)
    : impl_(std::make_unique<Impl>(c, e, t, std::move(l))) {}
SlamSystem::~SlamSystem() = default;

void SlamSystem::add_imu(const ImuSample& m) {
  auto& s = *impl_;
  if (!s.cfg.sensors.imu.enabled) return;
  s.imu_mount->add_imu(m);
  if (!s.imu_checked)
    s.check_imu_against_tf();
  else if (s.imu_trusted)
    s.use_imu(m);
}

void SlamSystem::add_baro(const BaroSample& b) {
  auto& s = *impl_;
  if (!s.baro_filter) return;
  s.baro_h = s.baro_filter->update(b.stamp, b.pressure_pa);
}
void SlamSystem::add_baro_reference(const BaroSample& b) {
  auto& s = *impl_;
  if (s.baro_filter) s.baro_filter->set_base_station(b.pressure_pa);
}

void SlamSystem::add_odom(const OdomMeasurement& m) {
  auto& s = *impl_;
  if (!s.cfg.sensors.external_odometry.enabled) return;
  if (!s.ext_odom.empty() && m.stamp <= s.ext_odom.back().stamp)
    return;  // out of order or repeated
  s.ext_odom.push_back(m);
  constexpr Timestamp kKeep = 10'000'000'000;  // ns
  while (s.ext_odom.front().stamp < m.stamp - kKeep) s.ext_odom.pop_front();
}

void SlamSystem::add_rgbd(const RgbdFrame& f) {
  auto& s = *impl_;
  if (s.visual_fe && s.modes.odometry != OdometrySource::Camera) {
    s.last_cam = f;  // keypoints for the next LiDAR keyframe (loop closure candidates, colors)
    return;
  }
  if (!s.visual_fe || s.modes.odometry != OdometrySource::Camera) return;
  if (!s.gravity_done &&
      !s.init_gravity(depth_to_cloud(f.depth, f.K, 4, s.tuning.visual.tracker.depth),
                      s.ext.T_base_camera, f.stamp))
    return;  // waiting for the IMU to level the start
  s.frame_stamp = f.stamp;
  s.cur_rgbd = &f;
  struct Clear {
    const RgbdFrame*& p;
    ~Clear() { p = nullptr; }
  } clear_rgbd{s.cur_rgbd};
  SensorFrame sf;
  sf.stamp = f.stamp;
  sf.rgbd = &f;
  const auto odom = s.ext_odom_at(f.stamp);
  if (odom) sf.odom = &*odom;
  if (s.planar())
    sf.up_base = s.plane_up();  // also bounds the tracking's own tilt
  else if (s.imu_trusted)
    sf.up_base = s.gravity.up();
  FrontendResult r = s.visual_fe->process(sf);
  s.to_plane(r);
  if (debug_channel("TRACK"))
    s.log->log(
        LogLevel::Debug,
        strprintf("track t=%.2f kps=%zu valid=%d inl=%d state=%d floor=%d pend=%d",
                  (f.stamp - 1000000000000LL) * 1e-9, f.keypoints.size(), r.valid,
                  r.quality.inliers, int(r.quality.state), s.current_floor(), s.pending.active));
  if (!r.valid) {
    s.handle_lost(f.stamp);
    s.feed_manager(r, f.stamp);
    return;
  }
  s.lost_since = 0;
  s.last_pose = r.T_odom_base;
  s.last_pose_stamp = f.stamp;
  if (r.quality.state == TrackingState::Good) s.last_good = {f.stamp, r.T_odom_base};
  s.learn_imu_mounting(r, f.stamp);
  s.feed_manager(r, f.stamp);
  if (r.is_keyframe) {
    const VisualKeyframeData& kd = s.visual_fe->last_keyframe();
    s.insert_keyframe(r, kd.cloud, s.ext.T_base_camera, f.stamp,
                      Impl::VisualKf{kd.keypoints, kd.p_cam, kd.K});
  }
}

void SlamSystem::add_lidar(const LidarScan& scan) {
  auto& s = *impl_;
  if (!s.lidar_fe || s.modes.odometry != OdometrySource::Lidar) return;
  if (!s.gravity_done && !s.init_gravity(scan.points, s.ext.T_base_lidar, scan.stamp))
    return;  // waiting for the IMU to level the start
  s.frame_stamp = scan.stamp;
  s.cur_scan = &scan;
  struct Clear {
    const LidarScan*& p;
    ~Clear() { p = nullptr; }
  } clear_scan{s.cur_scan};
  SensorFrame f;
  f.stamp = scan.stamp;
  f.lidar = &scan;
  // A ground robot's turns leave the IMU's heading on the base unobserved, and with it the
  // direction of every tilt the IMU reports: its gravity bounds the LiDAR's attitude once the
  // accelerations have shown that heading (ADR-0058). Before, a TF that has the IMU turned half
  // around reads every slope backwards.
  if (!s.planar() && s.imu_trusted && (s.cfg.robot.platform != Platform::Ground || s.imu_learned))
    f.up_base = s.gravity.up();
  FrontendResult r = s.lidar_fe->process(f);
  s.to_plane(r);
  if (!r.valid) {
    s.handle_lost(scan.stamp);
    s.feed_manager(r, scan.stamp);
    return;
  }
  s.lost_since = 0;
  s.last_pose = r.T_odom_base;
  s.last_pose_stamp = scan.stamp;
  if (r.quality.state == TrackingState::Good) s.last_good = {scan.stamp, r.T_odom_base};
  s.learn_imu_mounting(r, scan.stamp);
  s.feed_manager(r, scan.stamp);
  if (r.is_keyframe) {
    std::optional<Impl::VisualKf> vis;
    if (s.visual_fe && s.last_cam && std::llabs(s.last_cam->stamp - scan.stamp) < 60'000'000LL) {
      const VisualObservation obs = make_observation(*s.last_cam, s.tuning.visual.tracker.depth);
      vis = Impl::VisualKf{obs.keypoints, obs.p_cam, s.last_cam->K};
    }
    const KeyframeId id = s.next_kf;
    s.insert_keyframe(r, s.lidar_fe->last_keyframe_cloud(), s.ext.T_base_lidar, scan.stamp, vis);
    if (vis && s.cfg.fusion.colorize_cloud && s.last_cam && !s.last_cam->rgb.empty())
      s.keyframe_images[id] = std::make_shared<RgbdFrame>(*s.last_cam);
  }
}

SlamSystem::FloorStatus SlamSystem::floor_status() const {
  const auto& s = *impl_;
  FloorStatus st;
  st.state = s.pending.active ? FloorState::Lost : s.mgr.state();
  st.floor = s.current_floor();
  st.pending = s.pending.active;
  st.num_floors = s.mgr.num_floors();
  st.altitude = s.mgr.altitude();
  st.session = s.session;
  if (!s.transitions.empty()) st.last_transition = s.transitions.back().type;
  return st;
}

std::optional<gtsam::Pose3> SlamSystem::floor_from_odom() const {
  const auto& s = *impl_;
  const FloorId f = s.mgr.last_floor();
  const auto fl = s.backend.floors();
  if (std::find(fl.begin(), fl.end(), s.key_floor(f)) == fl.end()) return std::nullopt;
  return s.backend.floor(f).T_odom_floor().inverse();
}

std::vector<std::pair<Timestamp, gtsam::Pose3>> SlamSystem::floor_path(FloorId floor) const {
  const auto& s = *impl_;
  std::vector<std::pair<Timestamp, gtsam::Pose3>> out;
  const auto fl = s.backend.floors();
  if (std::find(fl.begin(), fl.end(), s.key_floor(floor)) == fl.end()) return out;
  const FloorGraph& g = s.backend.floor(floor);
  for (KeyframeId id : g.keyframe_ids()) out.push_back({g.node(id).stamp, g.pose(id)});
  return out;
}

std::vector<Eigen::Vector3f> SlamSystem::floor_cloud(FloorId floor, double voxel) const {
  const auto& s = *impl_;
  std::vector<Eigen::Vector3f> merged;
  const auto fl = s.backend.floors();
  if (std::find(fl.begin(), fl.end(), s.key_floor(floor)) == fl.end()) return merged;
  const FloorGraph& g = s.backend.floor(floor);
  for (KeyframeId id : g.keyframe_ids()) {
    const auto* cloud = s.backend.cloud(id);
    if (!cloud || cloud->empty()) continue;
    const auto w =
        transform_points(*cloud, (g.pose(id) * s.backend.keyframe(id).T_base_sensor).matrix());
    merged.insert(merged.end(), w.begin(), w.end());
  }
  return voxel_downsample(merged, voxel);
}

void SlamSystem::new_session() {
  auto& s = *impl_;
  if (!s.odom_fe || !s.last_pose) return;
  s.restart_frontend();
  const FloorId f = s.mgr.last_floor();
  s.bridge_from =
      s.last_kf_on_floor.count(f) ? std::optional<KeyframeId>(s.last_kf_on_floor[f]) : std::nullopt;
  s.last_kf.reset();
  s.last_kf_pose.reset();
  ++s.session;
  s.log->log(LogLevel::Info, "new session requested");
}

void SlamSystem::mark_floor_change(TransitionType type) {
  auto& s = *impl_;
  s.on_floor_event(s.mgr.mark_floor_change(type));
}

const std::vector<KeyframeInfo>& SlamSystem::keyframes() const { return impl_->kfs; }
const std::vector<AcceptedLoop>& SlamSystem::loops() const { return impl_->backend.loops(); }
const std::vector<TransitionRecord>& SlamSystem::transitions() const { return impl_->transitions; }
FloorId SlamSystem::current_floor() const { return impl_->current_floor(); }
SessionId SlamSystem::current_session() const { return impl_->session; }
TrackingQuality SlamSystem::tracking_quality() const {
  return impl_->odom_fe ? impl_->odom_fe->quality() : TrackingQuality();
}
std::optional<gtsam::Pose3> SlamSystem::current_pose_odom() const { return impl_->last_pose; }
std::optional<std::vector<KeypointUse>> SlamSystem::keypoint_use() const {
  const auto& s = *impl_;
  if (!s.visual_fe || s.modes.odometry != OdometrySource::Camera) return std::nullopt;
  return s.visual_fe->last_keypoint_use();
}
std::optional<double> SlamSystem::baro_height() const { return impl_->baro_h; }
std::optional<double> SlamSystem::floor_altitude(FloorId floor) const {
  const auto& s = *impl_;
  if (floor < 0 || static_cast<size_t>(floor) >= s.mgr.num_floors()) return std::nullopt;
  return s.mgr.level_altitude(floor);
}
const Backend& SlamSystem::backend() const { return impl_->backend; }
size_t SlamSystem::cross_floor_candidates_seen() const { return impl_->cross_floor_seen; }

MapData SlamSystem::finalize(const std::string& profile_yaml) {
  auto& s = *impl_;
  s.resolve_at_end();
  s.backend.optimize_all();
  MapData m;
  m.strataslam_version = kVersion;
  m.profile_yaml = profile_yaml;
  m.transitions = s.transitions;
  for (FloorId f : s.backend.floors()) {
    const FloorGraph& g = s.backend.floor(f);
    FloorData fd;
    fd.metadata.id = f;
    if (static_cast<size_t>(f) < s.mgr.num_floors())
      fd.metadata.altitude_m = s.mgr.level_altitude(f);
    // Every sensor the run used, the odometry's first (audit A6-05: only the odometry sensor was
    // listed).
    fd.metadata.sensors = {s.modes.odometry == OdometrySource::Lidar ? "lidar" : "camera"};
    const auto& sc = s.cfg.sensors;
    if (sc.camera.enabled && s.modes.odometry == OdometrySource::Lidar)
      fd.metadata.sensors.push_back("camera");
    if (sc.lidar.enabled && s.modes.odometry != OdometrySource::Lidar)
      fd.metadata.sensors.push_back("lidar");
    if (sc.imu.enabled) fd.metadata.sensors.push_back("imu");
    if (sc.barometer.enabled) fd.metadata.sensors.push_back("barometer");
    if (sc.external_odometry.enabled) fd.metadata.sensors.push_back("external_odometry");
    std::vector<Eigen::Vector3f> merged;
    std::vector<std::array<uint8_t, 3>> merged_colors;
    for (KeyframeId id : g.keyframe_ids()) {
      const FloorGraph::Node& node = g.node(id);
      const gtsam::Pose3 X = g.pose(id);
      KeyframeRecord kr;
      kr.id = id;
      kr.stamp_ns = node.stamp;
      kr.session = node.session;
      kr.t = X.translation();
      kr.q = Eigen::Quaterniond(X.rotation().matrix());
      fd.keyframes.push_back(kr);
      if (std::find(fd.metadata.sessions.begin(), fd.metadata.sessions.end(), node.session) ==
          fd.metadata.sessions.end()) {
        fd.metadata.sessions.push_back(node.session);
      }
      const auto* cloud = s.backend.cloud(id);
      if (cloud && !cloud->empty()) {
        PointCloud pc;
        pc.points = *cloud;
        fd.clouds[id] = std::move(pc);
        const auto w =
            transform_points(*cloud, (X * s.backend.keyframe(id).T_base_sensor).matrix());
        merged.insert(merged.end(), w.begin(), w.end());
        // Color the keyframe's LiDAR points with the camera image taken at the same time (if any).
        const auto img = s.keyframe_images.find(id);
        if (img != s.keyframe_images.end()) {
          const gtsam::Pose3 T_cam_lidar = s.ext.T_base_camera.inverse() * s.ext.T_base_lidar;
          const auto c = colorize_points(*cloud, T_cam_lidar, img->second->K, img->second->rgb,
                                         &img->second->depth);
          merged_colors.insert(merged_colors.end(), c.begin(), c.end());
        } else {
          merged_colors.insert(merged_colors.end(), cloud->size(),
                               std::array<uint8_t, 3>{128, 128, 128});
        }
        const auto vk = s.visual_kf.find(id);
        if (vk != s.visual_kf.end()) {
          VisualPayload vp;
          vp.descriptor_cols = static_cast<uint32_t>(kDescriptorBytes);
          for (size_t i = 0; i < vk->second.keypoints.size(); ++i) {
            const auto& k = vk->second.keypoints[i];
            const auto& p_cam = vk->second.p_cam[i];
            vp.keypoints.push_back(
                {k.u, k.v, k.response, p_cam ? static_cast<float>(p_cam->z()) : 0.f});
            vp.descriptors.insert(vp.descriptors.end(), k.descriptor.begin(), k.descriptor.end());
          }
          fd.visual[id] = std::move(vp);
        }
        const auto it = s.sc_of.find(id);
        if (it != s.sc_of.end()) {
          fd.lidar[id] =
              ScanContextDatabase::to_payload(it->second, static_cast<uint32_t>(cloud->size()),
                                              static_cast<float>(s.tuning.scan_context.max_range));
        }
      }
    }
    fd.edges = g.edge_records();
    if (!merged.empty()) {
      if (!s.keyframe_images.empty() && merged_colors.size() == merged.size()) {
        fd.floor_cloud = voxel_downsample_colored(merged, merged_colors, s.cfg.output.voxel_size);
      } else {
        PointCloud fc;
        fc.points = voxel_downsample(merged, s.cfg.output.voxel_size);
        fd.floor_cloud = std::move(fc);
      }
    }
    m.floors.push_back(std::move(fd));
  }
  return m;
}

}  // namespace strataslam
