#include "strataslam/pipeline/slam_system.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
  // Height sensors still sending: a sample since the last frame, and the frame time (the frontend's
  // clock, not the sensor's) when one last came. A silent IMU or barometer sees no floor change.
  bool imu_since_frame = false, baro_since_frame = false;
  bool imu_silent = false, baro_silent = false;
  double imu_sign = 1.0, imu_scale = 1.0;
  Timestamp imu_frame = 0, baro_frame = 0, watch_start = 0;
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
  // A planar scanner with a camera: the frames of the last half second, for the depth of each
  // keyframe from the image nearest to it in time (ADR-0068). The newest frame is not it: a
  // camera's stamps may run ahead of the scanner's (100 ms on the Tiago), and frames come every
  // 30-70 ms.
  std::deque<std::shared_ptr<const RgbdFrame>> recent_cams;
  std::map<KeyframeId, std::shared_ptr<const RgbdFrame>> keyframe_images;  // only when colorizing
  // The colours of each keyframe's cloud from its own image, and which points it saw (only when
  // colorizing): a camera's from the pixels its points came from (ADR-0068), a LiDAR's from the
  // image taken with the scan. The map's colours start from these (ADR-0069).
  struct KfColors {
    std::vector<std::array<uint8_t, 3>> rgb;
    std::vector<bool> seen;
  };
  std::map<KeyframeId, KfColors> keyframe_colors;
  // A planar scanner with a camera: each keyframe's depth cloud in its base frame, the 3D of the
  // map that the scan cannot see (ADR-0068). The scan stays the keyframe's cloud (loops, grid).
  std::map<KeyframeId, PointCloud> depth_clouds;
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
  int since_loop = 1000;
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
  // Unaligned: a 16-byte aligned member here pads the struct (clang-analyzer Padding).
  Eigen::Quaternion<double, Eigen::DontAlign> gyro_q =
      Eigen::Quaternion<double, Eigen::DontAlign>::Identity();
  Timestamp gyro_last = 0;
  size_t cross_floor_seen = 0;
  // The odometry's attitude over the last minute of tracking out of confined spaces, in the
  // frontend's current frame (cleared at its restarts): the gyroscope's heading drift against it.
  std::deque<std::pair<Timestamp, gtsam::Rot3>> heading_obs;
  Timestamp last_confined = 0;  // of the last scan that saw a confined space
  // The IMU's gravity against the floors the LiDAR sees (check_accel_bias): per sample, the
  // horizontal components of their difference in the base frame, and the base's heading.
  struct TiltSample {
    double dx, dy, yaw;
  };
  std::vector<TiltSample> tilt_samples;
  Timestamp last_tilt_check = 0;
  bool tilt_reported = false;
  // m/s^2, IMU frame: subtracted from every accelerometer sample. The profile's calibration, or
  // the offset measured against the floors with `accel_bias: auto` (ADR-0072, ADR-0074).
  std::optional<Eigen::Vector3d> accel_bias;
  /// A hand-held sensor's recent heights and the floor the LiDAR saw under it: picking it up or
  /// putting it down (a table, the floor) is told from a floor change by the floor staying.
  struct LiftSample {
    Timestamp stamp = 0;
    Eigen::Vector3d sensor = Eigen::Vector3d::Zero();  ///< the LiDAR's position (odometry frame)
    std::optional<double> ground;                      ///< height of the floor under it
  };
  std::deque<LiftSample> lift_hist;
  std::optional<Timestamp> lift_trigger;  ///< a lift seen starting, until it settles
  int lift_sign = 0;                      ///< +1 picked up, -1 put down

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
    bool height_cues = false;    ///< a height sensor was sending all along (height_cues_live)
  } pending;

  static FloorManagerConfig floor_config(const Config& c, FloorManagerConfig base) {
    base.floor_height_prior = c.multifloor.floor_height_prior;
    base.min_vertical_change = c.multifloor.min_vertical_change;
    base.tracking_loss_timeout = c.multifloor.tracking_loss_timeout;
    base.aerial_platform = c.robot.platform == Platform::Aerial;
    base.ground_platform = c.robot.platform == Platform::Ground;
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
    if (const auto& b = cfg.sensors.imu.accel_bias)
      accel_bias = Eigen::Vector3d((*b)[0], (*b)[1], (*b)[2]);
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
    if (cfg.sensors.lidar.type == LidarType::Planar) {
      // A planar scanner (ADR-0066): a few hundred points a sweep, a few metres to the walls, all
      // of them in one plane. Its sampling is that of an indoor map (5 cm, the scanner's own
      // spacing at a few metres) and its local map the reach of its range; the plane is registered
      // alone.
      lc.planar_scan = true;
      lc.scan_voxel = 0.05;
      lc.map_voxel = 0.05;
      lc.keyframe_cloud_voxel = 0.05;
      lc.max_correspondence = 0.5;
      lc.min_points = 30;  // 1.5 m of wall: what a sweep keeps at arm's length of an obstacle
      lc.map_radius = std::min(lc.map_radius, 2.0 * cfg.sensors.lidar.max_range);
    }
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

  /// What the body turned between two frames by the gyroscope, interpolated between its records,
  /// when they cover both stamps (the IMU has reached `to`).
  std::optional<gtsam::Rot3> gyro_delta(Timestamp from, Timestamp to) const {
    constexpr Timestamp kSlack = 20'000'000;  // ns: one record step
    if (!gyro_turn(from, to) || gyro_last + kSlack < to) return std::nullopt;
    const auto at = [&](Timestamp t) {
      auto it = std::upper_bound(gyro_att.begin(), gyro_att.end(), t,
                                 [](Timestamp v, const auto& e) { return v < e.first; });
      // Past the last record (kept every 20 ms): towards the attitude of the last sample, or a turn
      // of up to 20 ms is lost in every scan, and they add up (120 deg/s: 2.4 deg per scan).
      if (it == gyro_att.end()) {
        const auto& [tb, qb] = gyro_att.back();
        if (gyro_last <= tb) return Eigen::Quaterniond(qb);
        const double f =
            std::min(1.0, static_cast<double>(t - tb) / static_cast<double>(gyro_last - tb));
        return Eigen::Quaterniond(qb).slerp(f, Eigen::Quaterniond(gyro_q));
      }
      const Eigen::Quaterniond q1 = it->second, q0 = std::prev(it)->second;
      return q0.slerp(static_cast<double>(t - std::prev(it)->first) /
                          static_cast<double>(it->first - std::prev(it)->first),
                      q1);
    };
    return gtsam::Rot3((at(from).conjugate() * at(to)).toRotationMatrix());
  }

  /// rad/s: how much faster the gyroscope turns about the vertical than the LiDAR odometry, from
  /// the ends of the last minute out of confined spaces (0 with less than 20 s of it, or beyond
  /// 1 deg/s: not a drift).
  double heading_drift() const {
    if (heading_obs.size() < 2) return 0.0;
    const auto& [t0, R0] = heading_obs.front();
    const auto& [t1, R1] = heading_obs.back();
    const double span = (t1 - t0) * 1e-9;
    if (span < 20.0) return 0.0;
    const auto turn = gyro_turn(t0, t1);
    if (!turn) return 0.0;
    // About the world's vertical, whatever the body's axes (a hand-held rig may point one up).
    const double d = gtsam::Rot3::Logmap((R0 * *turn) * R1.inverse()).z() / span;
    return std::abs(d) < M_PI / 180.0 ? d : 0.0;
  }

  /// The accelerometer's offset, measured against the floors (ADR-0072): once a second, up by the
  /// IMU against the normal of the floor below the LiDAR, both in the base frame (not in a cabin
  /// the IMU measures a ride in: its accelerations are the ride's). An
  /// offset of the accelerometer is fixed in the body and shows at every heading; a sloping floor
  /// is fixed in the world and turns with the heading, so over headings all round its median is
  /// the offset. Said once in the log, with the `imu.accel_bias` it asks for, and with `accel_bias:
  /// auto` applied from then on (ADR-0074).
  void check_accel_bias(const LidarScan& scan, const FrontendResult& r) {
    if (tilt_reported || !cfg.sensors.imu.enabled || !imu_trusted || mgr.in_unobserved_transition())
      return;
    if (cfg.robot.platform == Platform::Ground && !imu_learned) return;  // its tilt's direction
    if (!gravity.up() || imu_buf.empty() || scan.stamp - last_tilt_check < 1'000'000'000) return;
    last_tilt_check = scan.stamp;
    // The accelerometer itself, its mean over the last 2 s (the gait's accelerations average out
    // there): the gravity tracker's up turns the offset round with the gyroscope at each turn and
    // the accelerometer brings it back only over its time constant.
    const auto up = mean_up(2'000'000'000);
    if (!up) return;
    const Eigen::Matrix3d R_bl = ext.T_base_lidar.rotation().matrix();
    const auto n =
        estimate_up_from_ground_plane(scan.points, R_bl.transpose() * *up, cfg.runtime.seed, 10.0);
    if (!n) return;
    const Eigen::Vector3d d = up->normalized() - R_bl * *n;
    tilt_samples.push_back({d.x(), d.y(), r.T_odom_base.rotation().yaw()});
    constexpr size_t kMinSamples = 60;  // a minute on floors
    if (tilt_samples.size() < kMinSamples) return;
    // Headings all round: the mean of their unit vectors is short.
    double c = 0, sn = 0;
    for (const auto& t : tilt_samples) {
      c += std::cos(t.yaw);
      sn += std::sin(t.yaw);
    }
    if (std::hypot(c, sn) / static_cast<double>(tilt_samples.size()) > 0.5) return;
    auto median = [&](double TiltSample::*m) {
      std::vector<double> v;
      v.reserve(tilt_samples.size());
      for (const auto& t : tilt_samples) v.push_back(t.*m);
      const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
      std::nth_element(v.begin(), mid, v.end());
      return *mid;
    };
    const Eigen::Vector3d d_med(median(&TiltSample::dx), median(&TiltSample::dy), 0.0);
    const double deg = std::asin(std::min(1.0, d_med.norm())) * 180.0 / M_PI;
    tilt_reported = true;
    // The specific force that makes that tilt, back in the IMU's own frame and units.
    const Eigen::Vector3d add = imu_sign / imu_scale *
                                ext.T_base_imu.rotation().matrix().transpose() *
                                (kStandardGravity * d_med);
    const Eigen::Vector3d b = accel_bias.value_or(Eigen::Vector3d::Zero()) + add;
    constexpr double kNoticeDeg = 0.5;  // a floor's own unevenness and the fits' noise
    if (deg < kNoticeDeg) {
      log->log(LogLevel::Info,
               strprintf("IMU: its gravity agrees with the floors the LiDAR sees (%.2f deg, %zu "
                         "scans at all headings)",
                         deg, tilt_samples.size()));
      return;
    }
    // `accel_bias: auto`: applied from now on. A sloping floor cannot read like this (over headings
    // all round it cancels); a LiDAR mounted off its TF does, and then levelling the IMU on the
    // LiDAR's floors is still what keeps them level in the map. Beyond 3 deg it is no
    // accelerometer's offset (a BMI270's is 1.7): a TF to fix, only said (ADR-0074).
    constexpr double kMaxAutoDeg = 3.0;
    if (cfg.sensors.imu.accel_bias_auto && !cfg.sensors.imu.accel_bias && deg <= kMaxAutoDeg) {
      accel_bias = b;
      // What is already held moves with it: the samples of the last seconds (the mean up of a
      // keyframe) and the tracker's up, which would otherwise take three time constants.
      const Eigen::Vector3d step = imu_sign * imu_scale * add;
      for (auto& m : imu_buf) m.accel -= step;
      gravity.shift_up(-d_med);
      log->log(LogLevel::Warn,
               strprintf("IMU: its gravity is %.1f deg from the floors the LiDAR sees (median of "
                         "%zu scans at all headings): its accelerometer's offset, applied from now "
                         "on (accel_bias: auto). To have it from the start, set imu.accel_bias: "
                         "[%.3f, %.3f, %.3f] (ADR-0074)",
                         deg, tilt_samples.size(), b.x(), b.y(), b.z()));
      return;
    }
    log->log(LogLevel::Warn,
             strprintf("IMU: its gravity is %.1f deg from the floors the LiDAR sees (median of %zu "
                       "scans at all headings): every metre travelled climbs %.0f mm. An "
                       "accelerometer offset does that; if the floors are level and the TF right, "
                       "set imu.accel_bias: [%.3f, %.3f, %.3f] (ADR-0072)",
                       deg, tilt_samples.size(), 1000.0 * std::tan(deg * M_PI / 180.0), b.x(),
                       b.y(), b.z()));
  }

  void observe_heading(Timestamp stamp, const gtsam::Rot3& R) {
    constexpr Timestamp kKeep = 60'000'000'000;  // ns
    heading_obs.emplace_back(stamp, R);
    while (heading_obs.front().first < stamp - kKeep) heading_obs.pop_front();
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
  /// Up in the base frame from the accelerometer's mean over the last `window` ns (corrected
  /// samples); none with fewer than 20 samples.
  std::optional<Eigen::Vector3d> mean_up(Timestamp window) const {
    if (imu_buf.empty()) return std::nullopt;
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
    size_t n = 0;
    for (auto it = imu_buf.rbegin();
         it != imu_buf.rend() && imu_buf.back().stamp - it->stamp <= window; ++it, ++n)
      a += it->accel;
    if (n < 20 || a.norm() < 1e-6) return std::nullopt;
    return (ext.T_base_imu.rotation().matrix() * a).normalized();
  }

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
    // In motion, with a calibrated accelerometer, its mean over the last 2 s, looser (ADR-0073).
    // Uncalibrated, its offset (0.5-1.5 deg on Elevator-VIGS) would tilt every floor by that much,
    // where the frontend's 1 deg bound leaves the LiDAR room to find the level (Campus1, Station1:
    // 0.3-0.7 m worse with it). Without it a walk's
    // keyframes had no attitude in the graph, only relative motions, which bend at almost no cost:
    // a loop that did not close in height was closed by tilting the floor (the portable rig's
    // floor 0 sank 2.3 m over 40 m to absorb its two staircases' 0.7 m). Not the gravity tracker's
    // up: with an accelerometer offset it turns the offset round with the gyroscope at each turn
    // and brings it back over its time constant, a heading-dependent tilt that bent Elevator-VIGS
    // floors (Station1 0.3 m, Factory-M 0.2 m). The mean has the offset the start was levelled
    // with. Not in a ride the IMU measures (the cabin accelerates), nor on a ground robot whose IMU
    // heading is not learned yet.
    if (!in.up_body && imu_trusted && accel_bias && !mgr.in_unobserved_transition() &&
        (cfg.robot.platform != Platform::Ground || imu_learned))
      if (const auto up = mean_up(2'000'000'000)) {
        in.up_body = *up;
        in.up_sigma = tuning.moving_gravity_sigma;
      }
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

  /// The colours of keyframe `id`'s cloud in the saved map: its own image's, and for the points it
  /// did not see, the images of up to 8 other keyframes of its floor within 6 m, nearest first,
  /// where the point lies on the surface the image measured (ADR-0069). Empty without any.
  std::vector<std::array<uint8_t, 3>> map_colors(const FloorGraph& g, KeyframeId id,
                                                 const std::vector<Eigen::Vector3f>& cloud) const {
    const size_t kViews = static_cast<size_t>(std::max(0, cfg.fusion.colorize_views));
    constexpr double kRadius = 6.0;  // m
    KfColors k;
    const auto own = keyframe_colors.find(id);
    if (own != keyframe_colors.end() && own->second.rgb.size() == cloud.size() &&
        own->second.seen.size() == cloud.size()) {
      k = own->second;
    } else if (keyframe_images.empty()) {
      return {};
    } else {
      k.rgb.assign(cloud.size(), {128, 128, 128});
      k.seen.assign(cloud.size(), false);
    }
    if (!keyframe_images.empty() &&
        std::find(k.seen.begin(), k.seen.end(), false) != k.seen.end()) {
      const gtsam::Pose3 Xi = g.pose(id) * backend.keyframe(id).T_base_sensor;
      std::vector<std::pair<double, KeyframeId>> near;
      for (KeyframeId j : g.keyframe_ids()) {
        if (j == id || !keyframe_images.count(j)) continue;
        const double d = (g.pose(j).translation() - g.pose(id).translation()).norm();
        if (d <= kRadius) near.push_back({d, j});
      }
      std::sort(near.begin(), near.end());
      if (near.size() > kViews) near.resize(kViews);
      std::vector<ColorView> views;
      views.reserve(near.size());
      for (const auto& [d, j] : near)
        views.push_back(
            {(g.pose(j) * ext.T_base_camera).inverse() * Xi, keyframe_images.at(j).get()});
      colorize_from_views(cloud, views, cfg.sensors.camera.depth_max, 0.3, &k.rgb, &k.seen);
    }
    if (std::find(k.seen.begin(), k.seen.end(), true) == k.seen.end()) return {};
    return std::move(k.rgb);
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
    // A planar scanner registers x, y and heading and holds the rest at the guess: two sweeps of
    // one level are in one plane, so the guess is put in it (a camera's estimate has a few
    // centimetres and tenths of a degree out of it, which the registration would keep and the graph
    // refuse).
    if (tuning.lidar.planar_scan)
      for (gtsam::Pose3& init : inits)
        init = gtsam::Pose3(gtsam::Rot3::Yaw(init.rotation().yaw()),
                            gtsam::Point3(init.x(), init.y(), 0.0));
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
    pending.height_cues = height_cues_live();
  }

  /// A height sensor of the profile (barometer, IMU measuring floor changes) is still sending: a
  /// floor change could not go unnoticed. One that fell silent sees none (the portable rig's camera
  /// and IMU stopped 20 s before an elevator ride three floors down: the loss in the cabin was
  /// taken for no floor change, as if the IMU had seen the cabin stand still).
  bool height_cues_live() const {
    return (cfg.sensors.barometer.enabled && !baro_silent && baro_frame != 0) ||
           (cfg.sensors.imu.enabled && cfg.sensors.imu.floor_changes && !imu_silent &&
            imu_frame != 0);
  }

  /// Per frame: which height sensors have sent since the last one, said in the log when one falls
  /// silent and when it comes back.
  void watch_height_sensors(Timestamp stamp) {
    constexpr Timestamp kSilence = 2'000'000'000;
    if (watch_start == 0) watch_start = stamp;
    const auto watch = [&](bool enabled, bool& since, Timestamp& last, bool& silent,
                           const char* name) {
      if (!enabled) return;
      if (since) {
        if (silent)
          log->log(LogLevel::Info,
                   strprintf("%s: data again after %.1f s", name, (stamp - last) * 1e-9));
        silent = false;
        last = stamp;
      } else if (!silent && stamp - (last ? last : watch_start) > kSilence) {
        silent = true;
        log->log(LogLevel::Warn,
                 strprintf("%s: no data for %.1f s: floor changes it would measure go unnoticed "
                           "until it comes back, and a tracking loss may be one",
                           name, (stamp - (last ? last : watch_start)) * 1e-9));
      }
      since = false;
    };
    watch(cfg.sensors.imu.enabled && cfg.sensors.imu.floor_changes, imu_since_frame, imu_frame,
          imu_silent, "IMU");
    watch(cfg.sensors.barometer.enabled, baro_since_frame, baro_frame, baro_silent, "barometer");
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
    // With a barometer or an IMU sending a floor change cannot go unnoticed, so a tracking loss is
    // not one: the session stays on its floor; with `multifloor.enabled: false` there is no other
    // floor to go to. Otherwise (none, or silent since the loss) the change is unobservable and
    // every floor is a relocalization candidate (and a failure to relocalize opens a new floor).
    pending.height_cues = pending.height_cues && height_cues_live();
    const bool height_sensors = pending.height_cues || !cfg.multifloor.enabled;
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
    const bool height_sensors =
        (pending.height_cues && height_cues_live()) || !cfg.multifloor.enabled;
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
        // Its arrival as decided: a level reached again replaces the new one (ADR-0073).
        const FloorEvent done = on_floor_event(ev);
        // No keyframe of the arrival floor will come: the keyframes of the climb are its first
        // ones.
        if (!transit_kfs.empty() && done.from != done.to) {
          const KeyframeId first = transit_kfs.front();
          absorb_transit_keyframes(done.to);
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
    heading_obs.clear();  // another frame
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
    if (mgr.in_unobserved_transition()) {
      // The lift explains the loss; the frontend restarts when it ends. Not once out of the cabin
      // (the scans stopped being confined a moment ago): its local map is the cabin, it would stay
      // lost, and the ride, not seeing the walk out of it, would go on integrating the IMU. It
      // restarts there, turned by the gyroscope, the session going on.
      constexpr Timestamp kOutOfCabin = 30'000'000'000;  // ns
      if (last_confined > 0 && stamp - last_confined < kOutOfCabin && stamp != last_confined &&
          lost_s > cfg.multifloor.tracking_loss_timeout && last_pose && odom_fe) {
        restart_frontend(0.0, /*level=*/false);
        last_kf.reset();
        last_kf_pose.reset();
        lost_since = 0;
      }
      return;
    }
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

  /// The end of the transition just closed (its last transit keyframes) seen again on `level`: a
  /// keyframe of that level the place recognition finds and the registration verifies, the
  /// registration's own height between them within the level tolerance. That height is measured
  /// against the structure itself, not through the odometry's stairs: a mezzanine looking down
  /// onto the floor below registers 1-2 m above it. Returns the height difference (ADR-0073).
  std::optional<double> arrival_on_level(FloorId level) {
    if (sc_db.count(key_floor(level)) == 0 && bow_db.count(key_floor(level)) == 0)
      return std::nullopt;
    constexpr size_t kTried = 6;  // the last keyframes, on the level reached
    for (size_t k = 0; k < std::min(kTried, transit_kfs.size()); ++k) {
      const KeyframeInput& cur = backend.keyframe(transit_kfs[transit_kfs.size() - 1 - k]);
      std::vector<std::pair<KeyframeId, gtsam::Pose3>> found;
      if (lidar_fe && !cur.cloud.empty() && sc_db.count(key_floor(level))) {
        const ScanContext sc = make_scan_context(cur.cloud, tuning.scan_context);
        for (const auto& c : sc_for(level).query(sc))
          if (const auto rel = verify_lidar(cur, c.id, c.match.yaw, nullptr))
            found.emplace_back(c.id, *rel);
      }
      const auto vis = visual_kf.find(cur.id);
      if (found.empty() && vis != visual_kf.end() && tuning.vocabulary &&
          bow_db.count(key_floor(level))) {
        const auto cands = bow_for(level).query(bow_of(vis->second), cur.session, cur.stamp, 0.0,
                                                tuning.bow_min_score,
                                                static_cast<size_t>(tuning.bow_max_candidates));
        for (const auto& c : cands)
          if (const auto rel = verify_visual(vis->second, c.id)) found.emplace_back(c.id, *rel);
      }
      for (const auto& [cand, rel] : found) {
        // Up in the candidate's frame: the level graph is gravity-aligned.
        const FloorGraph& g = backend.floor(level);
        const gtsam::Rot3 R = g.has(cand) ? g.pose(cand).rotation() : gtsam::Rot3();
        const double dz = (R.matrix() * rel.translation()).z();
        if (debug_channel("FLOORS"))
          log->log(LogLevel::Debug,
                   strprintf("arrival kf %lu seen from kf %lu of level %d: %.2f m above it",
                             static_cast<unsigned long>(cur.id), static_cast<unsigned long>(cand),
                             level, dz));
        if (std::abs(dz) <= mgr.level_tolerance()) return dz;
      }
    }
    return std::nullopt;
  }

  /// Returns the event as it was decided: its arrival may be revised to a level seen again.
  FloorEvent on_floor_event(FloorEvent ev) {
    // A new level near an existing one: the place decides (ADR-0073).
    if (ev.kind == FloorEvent::Kind::TransitionEnded && ev.near_level >= 0 && ev.from != ev.to) {
      const double dz_level = mgr.level_altitude(ev.to) - mgr.level_altitude(ev.near_level);
      if (const auto dz = arrival_on_level(ev.near_level);
          dz && mgr.revise_arrival(ev.near_level)) {
        log->log(LogLevel::Info,
                 strprintf("floor %d reached again: its place is seen %.2f m from it (the stairs "
                           "measured %.2f m)",
                           ev.near_level, *dz, dz_level));
        ev.to = ev.near_level;
      } else {
        log->log(LogLevel::Info,
                 strprintf("a new level %.2f m from floor %d: not seen as the same place", dz_level,
                           ev.near_level));
      }
    }
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
      return ev;
    }
    if (ev.kind != FloorEvent::Kind::TransitionEnded) return ev;
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
      return ev;
    }
    if (ev.from == ev.to && ev.continuous_tracking && !ev.needs_frontend_reset && !pending.active) {
      log->log(LogLevel::Info,
               "height change was not a floor change: keeping its keyframes on floor " +
                   std::to_string(ev.to));
      absorb_transit_keyframes(ev.to);
      return ev;
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
    return ev;
  }

  /// Height of the floor under the LiDAR (odometry frame): the lowest dense horizontal layer of the
  /// returns within 2.5 m around it and at least 0.2 m below it, past the profile's min_range (the
  /// person and the rig). A table top or a step is higher than the floor; a layer of a few points
  /// is not a floor.
  /// k of the every-k-th thinning that keeps a floor's keyframe points within `max_points`.
  size_t thinning(const FloorGraph& g, size_t max_points) const;

  std::optional<double> ground_under(const LidarScan& scan,
                                     const gtsam::Pose3& T_odom_lidar) const {
    constexpr double kRadius = 2.5, kBelow = 0.2, kBin = 0.05;
    const Eigen::Matrix3d R = T_odom_lidar.rotation().matrix();
    const Eigen::Vector3d& t = T_odom_lidar.translation();
    const double min_range = cfg.sensors.lidar.min_range;
    std::vector<double> z;
    for (size_t i = 0; i < scan.points.size(); i += 2) {
      const Eigen::Vector3d p = scan.points[i].cast<double>();
      if (p.norm() < min_range) continue;
      const Eigen::Vector3d w = R * p + t;
      if (w.z() < t.z() - kBelow && std::hypot(w.x() - t.x(), w.y() - t.y()) < kRadius)
        z.push_back(w.z());
    }
    if (z.size() < 30) return std::nullopt;
    const double lo = *std::min_element(z.begin(), z.end());
    const auto n = static_cast<size_t>((t.z() - kBelow - lo) / kBin) + 1;
    std::vector<int> hist(n, 0);
    for (double v : z) ++hist[std::min(n - 1, static_cast<size_t>((v - lo) / kBin))];
    const int peak = *std::max_element(hist.begin(), hist.end());
    if (peak < 20) return std::nullopt;
    for (size_t k = 0; k < n; ++k)
      if (hist[k] >= std::max(20, peak / 5)) return lo + (static_cast<double>(k) + 0.5) * kBin;
    return std::nullopt;
  }

  /// A hand-held sensor picked up or put down (a table, the floor), told from a floor change by the
  /// floor the LiDAR sees under it staying. Triggered strictly: within 0.8 s the sensor moved 0.3 m
  /// or more up or down, at least twice as much as across, the floor under it within 0.10 m. A lift
  /// goes at 1-1.5 m/s straight up (Elevator-VIGS Office4: 0.94 m in 0.6 s off the ground); stairs
  /// and ramps take the floor with the sensor, go at 0.2-0.3 m/s and mostly across, and an elevator
  /// takes its cabin. Then the whole lift is measured, from the rest before it (the lowest or
  /// highest height in the 2 s before the trigger) to where the sensor settles (0.4 s within 6 cm),
  /// less what the floor under it moved between the two (up to 0.4 m: a sensor put on the floor
  /// sees it at a grazing angle, and the floor around may slope; Elevator-VIGS Residential3, 0.34
  /// m, which left a pick-up without its put-down and a false level): the floor manager holds its
  /// altitude by that, and a transition the lift opened ends on its floor. A hand-held rig's walk
  /// is then on its floor, in the map and its loop closures, instead of in a transition that only
  /// the next stairs close (the portable rig, 120 s; Office4, picked up and put down six times).
  void detect_lift(const LidarScan& scan, const FrontendResult& r) {
    if (cfg.robot.platform != Platform::Handheld || !cfg.multifloor.enabled) return;
    constexpr Timestamp kKeep = 4'000'000'000, kTrigger = 800'000'000, kBefore = 2'000'000'000,
                        kSettle = 400'000'000, kGiveUp = 5'000'000'000;
    constexpr double kMinLift = 0.3, kTriggerFloor = 0.10, kFloorKept = 0.4, kStill = 0.06;
    const gtsam::Pose3 T = r.T_odom_base * ext.T_base_lidar;
    lift_hist.push_back({scan.stamp, T.translation(), ground_under(scan, T)});
    while (lift_hist.front().stamp < scan.stamp - kKeep) lift_hist.pop_front();
    const LiftSample& now = lift_hist.back();
    if (!lift_trigger) {
      if (!now.ground) return;
      const double g_now = now.ground.value_or(0.0);
      for (const LiftSample& then : lift_hist) {
        if (then.stamp < now.stamp - kTrigger || !then.ground || &then == &now) continue;
        const Eigen::Vector3d d = now.sensor - then.sensor;
        if (std::abs(d.z()) >= kMinLift &&
            std::abs(g_now - then.ground.value_or(0.0)) <= kTriggerFloor &&
            std::abs(d.z()) >= 2.0 * d.head<2>().norm()) {
          lift_trigger = now.stamp;
          lift_sign = d.z() > 0 ? 1 : -1;
          break;
        }
      }
      return;
    }
    if (now.stamp - *lift_trigger > kGiveUp) {  // it never settled: not a lift
      lift_trigger.reset();
      return;
    }
    // Settled: the last 0.4 s within 6 cm; the lift ends at their first sample.
    if (now.stamp - lift_hist.front().stamp < kSettle) return;
    double lo = now.sensor.z(), hi = lo;
    size_t end = lift_hist.size() - 1;
    for (size_t k = lift_hist.size(); k > 0; --k) {
      const LiftSample& h = lift_hist[k - 1];
      if (h.stamp < now.stamp - kSettle) break;
      lo = std::min(lo, h.sensor.z());
      hi = std::max(hi, h.sensor.z());
      end = k - 1;
    }
    if (hi - lo > kStill) return;
    size_t start = end;
    for (size_t k = 0; k < lift_hist.size(); ++k)
      if (lift_hist[k].stamp >= *lift_trigger - kBefore && lift_hist[k].stamp <= *lift_trigger &&
          lift_sign * lift_hist[k].sensor.z() < lift_sign * lift_hist[start].sensor.z())
        start = k;
    std::optional<double> g0, g1;  // the floor seen nearest the start (back) and the end (on)
    for (size_t k = start + 1; k > 0 && !g0; --k) g0 = lift_hist[k - 1].ground;
    for (size_t k = end; k < lift_hist.size() && !g1; ++k) g1 = lift_hist[k].ground;
    lift_trigger.reset();
    if (!g0 || !g1) return;
    const double floor_moved = g1.value_or(0.0) - g0.value_or(0.0);
    const double dz = lift_hist[end].sensor.z() - lift_hist[start].sensor.z();
    if (std::abs(floor_moved) > kFloorKept || lift_sign * dz < kMinLift) return;
    const double lift = dz - floor_moved;
    // A transition the lift opened began after its start, or up to 3 s before by the IMU: its
    // vertical-acceleration window opens on the lift's push (Elevator-VIGS Residential3).
    const FloorEvent ev = mgr.device_lifted(lift, lift_hist[start].stamp - 3'000'000'000);
    lift_hist.clear();
    log->log(LogLevel::Info, strprintf("the sensor was %s %.2f m over the same floor: not a floor "
                                       "change",
                                       lift > 0 ? "picked up" : "put down", std::abs(lift)));
    if (ev.kind != FloorEvent::Kind::None) on_floor_event(ev);
  }

  void feed_manager(const FrontendResult& r, Timestamp stamp) {
    // `multifloor.enabled: false`: one level, the floor manager never leaves floor 0 (an outdoor
    // drive has slopes, not floors). A tracking loss still splits sessions (handle_lost).
    if (!cfg.multifloor.enabled) return;
    watch_height_sensors(stamp);
    FloorSample fs;
    fs.stamp = stamp;
    fs.tracking_ok = r.valid;
    fs.vertical_observed = !r.confined;
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

void SlamSystem::add_imu(const ImuSample& raw) {
  auto& s = *impl_;
  if (!s.cfg.sensors.imu.enabled) return;
  // The accelerometer's calibration, in its own frame, before anything reads it (the mounting,
  // the gravity, the vertical acceleration): ADR-0072.
  s.imu_since_frame = true;
  ImuSample m = raw;
  if (s.accel_bias) m.accel -= *s.accel_bias;
  s.imu_mount->add_imu(m);
  if (!s.imu_checked)
    s.check_imu_against_tf();
  else if (s.imu_trusted)
    s.use_imu(m);
}

void SlamSystem::add_baro(const BaroSample& b) {
  auto& s = *impl_;
  if (!s.baro_filter) return;
  s.baro_since_frame = true;
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
    if (s.tuning.lidar.planar_scan) {
      constexpr Timestamp kKeepCams = 500'000'000;  // ns
      s.recent_cams.push_back(std::make_shared<const RgbdFrame>(f));
      while (s.recent_cams.front()->stamp < f.stamp - kKeepCams) s.recent_cams.pop_front();
    }
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
  const auto odom = s.modes.odometry_prior ? s.ext_odom_at(f.stamp) : std::nullopt;
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
    // Each point of the keyframe's cloud takes the colour of the pixel it came from (ADR-0068:
    // before, `colorize_cloud` coloured LiDAR points only, and a camera's map had no colour).
    if (s.cfg.fusion.colorize_cloud && !f.rgb.empty()) {
      Impl::KfColors& kc = s.keyframe_colors[s.next_kf];
      kc.rgb = colorize_points(kd.cloud, gtsam::Pose3(), f.K, f.rgb, nullptr, 0.3, 20.0,
                               {128, 128, 128}, &kc.seen);
    }
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
  // The robot's own odometry, when the profile makes it the LiDAR's prior (ADR-0071).
  const auto odom = s.modes.odometry_prior ? s.ext_odom_at(scan.stamp) : std::nullopt;
  if (odom) f.odom = &*odom;
  // A ground robot's turns leave the IMU's heading on the base unobserved, and with it the
  // direction of every tilt the IMU reports: its gravity bounds the LiDAR's attitude once the
  // accelerations have shown that heading (ADR-0058). Before, a TF that has the IMU turned half
  // around reads every slope backwards, but not steeper: the size of the tilt bounds it until then
  // (out of an elevator cabin the Summit XL's odometry took 7 deg of pitch on the level, a 2.7 m
  // ramp in 55 s, on a bag too short to show the heading).
  if (!s.planar() && s.imu_trusted) {
    f.up_base = s.gravity.up();
    f.up_tilt_only = s.cfg.robot.platform == Platform::Ground && !s.imu_learned;
  }
  // The gyroscope's turn since the last pose, its heading drift (measured against the odometry out
  // of confined spaces) taken out: the rotation of a scan in an elevator cabin, where the
  // registration's goes wrong by tens of degrees as the person turns round in it.
  std::optional<gtsam::Rot3> raw_turn, R_prev;
  if (s.last_pose) {
    R_prev = s.last_pose->rotation();
    raw_turn = s.gyro_delta(s.last_pose_stamp, scan.stamp);
  }
  const double turn_s = (scan.stamp - s.last_pose_stamp) * 1e-9;
  if (raw_turn && R_prev) {
    const gtsam::Rot3& R = *R_prev;
    // Projected on the rotations: a product of matrices is not one to the last digit, and in a
    // confined space nothing else re-estimates the rotation it gives.
    f.gyro_delta = gtsam::Rot3::ClosestTo(
        (R.inverse() * gtsam::Rot3::Yaw(-s.heading_drift() * turn_s) * R * *raw_turn).matrix());
  }
  FrontendResult r = s.lidar_fe->process(f);
  if (r.confined) s.last_confined = scan.stamp;
  if (r.valid && !r.confined) {
    s.observe_heading(scan.stamp, r.T_odom_base.rotation());
    s.check_accel_bias(scan, r);
  }
  if (debug_channel("GYRO") && r.valid && f.gyro_delta && R_prev)
    s.log->log(LogLevel::Debug,
               strprintf("gyro t=%.3f confined=%d drift=%.3f deg/s registration off the "
                         "gyroscope by %.2f deg",
                         scan.stamp * 1e-9, int(r.confined), s.heading_drift() * 180.0 / M_PI,
                         gtsam::Rot3::Logmap(
                             f.gyro_delta->between(R_prev->between(r.T_odom_base.rotation())))
                                 .norm() *
                             180.0 / M_PI));
  s.to_plane(r);
  if (debug_channel("TRACK"))
    s.log->log(LogLevel::Debug,
               strprintf("scan t=%.2f pts=%zu valid=%d inl=%d fit=%.3f eig=%.4f state=%d "
                         "yaw=%.2f",
                         scan.stamp * 1e-9, scan.points.size(), r.valid, r.quality.inliers,
                         r.quality.fitness, r.quality.min_eigenvalue, int(r.quality.state),
                         r.valid ? r.T_odom_base.rotation().yaw() * 180.0 / M_PI : 0.0));
  if (!r.valid) {
    s.handle_lost(scan.stamp);
    s.feed_manager(r, scan.stamp);
    return;
  }
  s.lost_since = 0;
  s.last_pose = r.T_odom_base;
  const Timestamp prev_pose_stamp = s.last_pose_stamp;
  s.last_pose_stamp = scan.stamp;
  if (r.quality.state == TrackingState::Good) s.last_good = {scan.stamp, r.T_odom_base};
  s.learn_imu_mounting(r, scan.stamp);
  s.feed_manager(r, scan.stamp);
  // Also in a confined scan: a sensor put on the floor sees only what is near (Elevator-VIGS
  // Residential3: 17 s unseen, the put-down closed as a ramp and a level). In a cabin the odometry
  // sees no vertical motion, so it cannot look like a lift.
  s.detect_lift(scan, r);
  if (r.is_keyframe) {
    std::optional<Impl::VisualKf> vis;
    if (s.visual_fe && s.last_cam && std::llabs(s.last_cam->stamp - scan.stamp) < 60'000'000LL) {
      const VisualObservation obs = make_observation(*s.last_cam, s.tuning.visual.tracker.depth);
      vis = Impl::VisualKf{obs.keypoints, obs.p_cam, s.last_cam->K};
    }
    const KeyframeId id = s.next_kf;
    s.insert_keyframe(r, s.lidar_fe->last_keyframe_cloud(), s.ext.T_base_lidar, scan.stamp, vis);
    if (vis && s.cfg.fusion.colorize_cloud && s.last_cam && !s.last_cam->rgb.empty()) {
      const RgbdFrame& img = *s.last_cam;
      // The scan's colours from this image, as the map had them at save time (now also live).
      if (const auto* cloud = s.backend.cloud(id)) {
        Impl::KfColors& kc = s.keyframe_colors[id];
        kc.rgb = colorize_points(*cloud, s.ext.T_base_camera.inverse() * s.ext.T_base_lidar, img.K,
                                 img.rgb, &img.depth, 0.3, 20.0, {128, 128, 128}, &kc.seen);
      }
      // Kept for the other keyframes' points at save time, at the profile's scale (ADR-0069).
      if (s.cfg.fusion.colorize_views > 0)
        s.keyframe_images[id] = std::make_shared<RgbdFrame>(decimate_frame(
            img, static_cast<int>(std::lround(1.0 / s.cfg.fusion.colorize_image_scale))));
    }
    // A planar scanner sees one line of the room: the 3D of the map is the camera's depth, placed
    // by the scan's pose (ADR-0068), from the image nearest to the sweep in time (within 150 ms).
    // The base is moved to the image's time along the motion of the last scan interval: at
    // 0.5 rad/s, 50 ms is 1.4 deg, 12 cm on a wall 5 m away.
    std::shared_ptr<const RgbdFrame> near;
    for (const auto& c : s.recent_cams)
      if (!near || std::llabs(c->stamp - scan.stamp) < std::llabs(near->stamp - scan.stamp))
        near = c;
    constexpr Timestamp kDepthWindow = 150'000'000;  // ns
    if (near && std::llabs(near->stamp - scan.stamp) <= kDepthWindow &&
        !near->depth.meters.empty()) {
      gtsam::Pose3 T_kf_cam_base;
      if (prev_pose_stamp > 0 && scan.stamp > prev_pose_stamp) {
        const double frac = static_cast<double>(near->stamp - scan.stamp) /
                            static_cast<double>(scan.stamp - prev_pose_stamp);
        T_kf_cam_base = gtsam::Pose3::Expmap(gtsam::Pose3::Logmap(r.delta) * frac);
      }
      s.depth_clouds[id] =
          depth_map_cloud(*near, s.tuning.visual.cloud_stride, s.tuning.visual.tracker.depth,
                          T_kf_cam_base * s.ext.T_base_camera);
    }
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

std::optional<gtsam::Pose3> SlamSystem::floor_anchor_from_odom() const {
  const auto& s = *impl_;
  const FloorId f = s.mgr.last_floor();
  const auto fl = s.backend.floors();
  if (std::find(fl.begin(), fl.end(), s.key_floor(f)) == fl.end()) return std::nullopt;
  return s.backend.floor(f).T_odom_floor().inverse();
}

std::optional<gtsam::Pose3> SlamSystem::floor_from_odom() const {
  const auto& s = *impl_;
  const FloorId f = s.mgr.last_floor();
  const auto fl = s.backend.floors();
  if (std::find(fl.begin(), fl.end(), s.key_floor(f)) == fl.end()) return std::nullopt;
  const FloorGraph& g = s.backend.floor(f);
  const auto ids = g.keyframe_ids();
  if (ids.empty()) return g.T_odom_floor().inverse();
  // The optimized pose of the latest keyframe against the odometry's when it was taken: the
  // drift the graph has corrected so far (zero until a loop closes). Heading and position only:
  // both frames are levelled by gravity, and the keyframe's own roll and pitch (a robot on a
  // ramp, 3.5 deg on the Summit's prueba_03) would tilt the whole odometry frame about the
  // keyframe and lift points tens of metres away by metres.
  const gtsam::Pose3 X = g.pose(ids.back());
  const gtsam::Pose3& O = s.backend.keyframe(ids.back()).T_odom_base;
  const auto yaw = [](const gtsam::Rot3& R) {
    const Eigen::Matrix3d m = R.matrix();
    return std::atan2(m(1, 0), m(0, 0));
  };
  const gtsam::Rot3 Rz = gtsam::Rot3::Yaw(yaw(X.rotation()) - yaw(O.rotation()));
  return gtsam::Pose3(Rz, X.translation() - Rz.rotate(O.translation()));
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

namespace {
/// Every k-th element of `v` (k > 1; callers keep `v` itself for k <= 1, without a copy).
template <class T>
std::vector<T> every_kth(const std::vector<T>& v, size_t k) {
  std::vector<T> out;
  out.reserve(v.size() / k + 1);
  for (size_t i = 0; i < v.size(); i += k) out.push_back(v[i]);
  return out;
}
}  // namespace

size_t SlamSystem::Impl::thinning(const FloorGraph& g, size_t max_points) const {
  if (max_points == 0) return 1;
  size_t total = 0;
  for (KeyframeId id : g.keyframe_ids()) {
    if (const auto* cloud = backend.cloud(id)) total += cloud->size();
    const auto d = depth_clouds.find(id);
    if (d != depth_clouds.end()) total += d->second.points.size();
  }
  return std::max<size_t>(1, (total + max_points - 1) / max_points);
}

std::vector<Eigen::Vector3f> SlamSystem::floor_cloud(FloorId floor, double voxel,
                                                     size_t max_points) const {
  const auto& s = *impl_;
  std::vector<Eigen::Vector3f> merged;
  const auto fl = s.backend.floors();
  if (std::find(fl.begin(), fl.end(), s.key_floor(floor)) == fl.end()) return merged;
  const FloorGraph& g = s.backend.floor(floor);
  const size_t k = s.thinning(g, max_points);
  for (KeyframeId id : g.keyframe_ids()) {
    const auto* cloud = s.backend.cloud(id);
    if (!cloud || cloud->empty()) continue;
    const Eigen::Matrix4d T = (g.pose(id) * s.backend.keyframe(id).T_base_sensor).matrix();
    const auto w = k > 1 ? transform_points(every_kth(*cloud, k), T) : transform_points(*cloud, T);
    merged.insert(merged.end(), w.begin(), w.end());
    const auto d = s.depth_clouds.find(id);
    if (d != s.depth_clouds.end()) {
      const Eigen::Matrix4d Td = g.pose(id).matrix();
      const auto wd = k > 1 ? transform_points(every_kth(d->second.points, k), Td)
                            : transform_points(d->second.points, Td);
      merged.insert(merged.end(), wd.begin(), wd.end());
    }
  }
  return voxel_downsample(merged, voxel);
}

PointCloud SlamSystem::floor_cloud_colored(FloorId floor, double voxel, size_t max_points) const {
  const auto& s = *impl_;
  PointCloud out;
  const auto fl = s.backend.floors();
  if (std::find(fl.begin(), fl.end(), s.key_floor(floor)) == fl.end()) return out;
  const FloorGraph& g = s.backend.floor(floor);
  const size_t k = s.thinning(g, max_points);
  std::vector<Eigen::Vector3f> merged;
  std::vector<std::array<uint8_t, 3>> colors;
  bool any = false;
  const std::array<uint8_t, 3> grey{128, 128, 128};
  for (KeyframeId id : g.keyframe_ids()) {
    const auto* cloud = s.backend.cloud(id);
    if (cloud && !cloud->empty()) {
      const auto pts = k > 1 ? every_kth(*cloud, k) : *cloud;
      const auto w =
          transform_points(pts, (g.pose(id) * s.backend.keyframe(id).T_base_sensor).matrix());
      merged.insert(merged.end(), w.begin(), w.end());
      const auto kc = s.keyframe_colors.find(id);
      if (kc != s.keyframe_colors.end() && kc->second.rgb.size() == cloud->size()) {
        any = true;
        const auto rgb = k > 1 ? every_kth(kc->second.rgb, k) : kc->second.rgb;
        colors.insert(colors.end(), rgb.begin(), rgb.end());
      } else {
        colors.insert(colors.end(), pts.size(), grey);
      }
    }
    const auto d = s.depth_clouds.find(id);
    if (d != s.depth_clouds.end() && !d->second.points.empty()) {
      const auto pts = k > 1 ? every_kth(d->second.points, k) : d->second.points;
      const auto wd = transform_points(pts, g.pose(id).matrix());
      merged.insert(merged.end(), wd.begin(), wd.end());
      if (d->second.colors.size() == d->second.points.size()) {
        any = true;
        const auto rgb = k > 1 ? every_kth(d->second.colors, k) : d->second.colors;
        colors.insert(colors.end(), rgb.begin(), rgb.end());
      } else {
        colors.insert(colors.end(), pts.size(), grey);
      }
    }
  }
  if (!any) {
    out.points = voxel_downsample(merged, voxel);
    return out;
  }
  return voxel_downsample_colored(merged, colors, voxel);
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
  // Every floor where the odometry left it, in the frame of the lowest one: the building as the
  // live view draws it (ADR-0064).
  const auto floors = s.backend.floors();
  Pose4 root_origin;  // read only inside the loop below, which has a floor
  if (!floors.empty())
    root_origin = s.backend.floor(*std::min_element(floors.begin(), floors.end())).origin();
  for (FloorId f : floors) {
    const FloorGraph& g = s.backend.floor(f);
    FloorData fd;
    fd.metadata.id = f;
    if (static_cast<size_t>(f) < s.mgr.num_floors())
      fd.metadata.altitude_m = s.mgr.level_altitude(f);
    {
      const Pose4 X = root_origin.inverse() * g.origin();
      fd.metadata.placement =
          Eigen::Vector4d(X.theta(), X.translation().x(), X.translation().y(), X.translation().z());
    }
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
    // The origins of the clouds' rays in the base frame, for the free space of a grid (ADR-0068).
    fd.metadata.cloud_origin =
        (s.modes.odometry == OdometrySource::Lidar ? s.ext.T_base_lidar : s.ext.T_base_camera)
            .translation();
    std::vector<Eigen::Vector3f> merged;
    std::vector<std::array<uint8_t, 3>> merged_colors;
    bool colored = false;  // some keyframe has colours: the floor cloud is coloured, grey elsewhere
    const std::array<uint8_t, 3> grey{128, 128, 128};
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
        // Saved in the keyframe's base frame (`cloud_frame: base`): placed by its pose alone.
        PointCloud pc;
        pc.points = transform_points(*cloud, s.backend.keyframe(id).T_base_sensor.matrix());
        const auto w =
            transform_points(*cloud, (X * s.backend.keyframe(id).T_base_sensor).matrix());
        merged.insert(merged.end(), w.begin(), w.end());
        // Its colours: a LiDAR's points from the camera image taken at the same time and from
        // the images around it, a camera's from their own pixels (ADR-0068, ADR-0069); kept with
        // the keyframe's cloud as well.
        std::vector<std::array<uint8_t, 3>> c = s.map_colors(g, id, *cloud);
        if (c.empty()) {
          merged_colors.insert(merged_colors.end(), cloud->size(), grey);
        } else {
          colored = true;
          merged_colors.insert(merged_colors.end(), c.begin(), c.end());
          pc.colors = std::move(c);
        }
        fd.clouds[id] = std::move(pc);
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
      // A planar scanner's keyframe: the camera's depth, the 3D of the floor cloud (ADR-0068).
      const auto dc = s.depth_clouds.find(id);
      if (dc != s.depth_clouds.end() && !dc->second.points.empty()) {
        const auto w = transform_points(dc->second.points, X.matrix());
        merged.insert(merged.end(), w.begin(), w.end());
        if (dc->second.colors.size() == dc->second.points.size()) {
          colored = true;
          merged_colors.insert(merged_colors.end(), dc->second.colors.begin(),
                               dc->second.colors.end());
        } else {
          merged_colors.insert(merged_colors.end(), dc->second.points.size(), grey);
        }
        fd.depth_clouds[id] = dc->second;
        fd.metadata.depth_origin = s.ext.T_base_camera.translation();
      }
    }
    fd.edges = g.edge_records();
    if (!merged.empty()) {
      if (colored && merged_colors.size() == merged.size()) {
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
