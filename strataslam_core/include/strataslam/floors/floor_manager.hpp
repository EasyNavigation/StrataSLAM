#pragma once

#include <Eigen/Core>
#include <deque>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "strataslam/common/logging.hpp"
#include "strataslam/common/types.hpp"
#include "strataslam/config/config.hpp"
#include "strataslam/floors/transition.hpp"

namespace strataslam {

enum class FloorState { OnFloor, InTransition, Lost };

/// Thresholds of the floor manager. All have documented defaults and are overridable.
struct FloorManagerConfig {
  std::optional<double> floor_height_prior = 3.0;  ///< m
  double min_vertical_change = 1.5;                ///< m, used when there is no prior
  double level_tolerance_factor = 0.4;  ///< a level matches within this x prior (or x min change)
  double depart_dz = 0.30;              ///< m of odometry height change that opens a transition
  double baro_depart = 0.50;            ///< m of barometric height change that opens a transition
  double settle_s = 2.0;     ///< s of calm before a transition may close (x2.5 without prior)
  double calm_vz = 0.06;     ///< m/s
  double calm_slope = 0.04;  ///< climb per metre travelled (2.3 deg) above which the robot is not
                             ///< calm: a gentle ramp is slower than calm_vz
  double slope_long_window_s = 8.0;  ///< s of the longer slope window that keeps a transition open
                                     ///< through the flicker of a gentle ramp
  double calm_slope_long = 0.025;    ///< climb per metre over that window (>= 3 m travelled)
  /// A level is also recognized by its plateau, whatever its height: calm for plateau_s while the
  /// robot travels plateau_travel_m, after a change of at least plateau_min_change. A stair landing
  /// lasts a few seconds; a floor is lived on. It is what accepts a mezzanine that is not a
  /// multiple of the prior.
  double plateau_s = 20.0;
  double plateau_travel_m = 10.0;
  double plateau_min_change = 1.0;
  double calm_slope_window_s =
      2.0;  ///< s over which that slope is measured (only while moving >= 1 m)
  double elevator_baro_min = 1.0;  ///< m, barometric change that identifies an elevator
  double elevator_static_ratio =
      0.5;                           ///< odometry change must be below this x the barometric change
  double ramp_max_slope_deg = 15.0;  ///< stairs are steeper, ramps gentler
  double slope_window_m = 1.0;       ///< horizontal run over which slopes are measured
  double aerial_max_horizontal =
      0.5;  ///< horizontal run per metre climbed below which motion is "vertical"
  double imu_depart_accel =
      0.25;                   ///< m/s^2, 1 s mean of vertical acceleration that opens a transition
  double imu_depart_s = 0.5;  ///< s above the threshold before the transition opens
  double imu_calm_accel = 0.10;  ///< m/s^2
  double elevator_imu_min =
      1.0;  ///< m, vertical change measured by the IMU that identifies an elevator
  /// s of IMU calm, once the cabin's integrated speed is back near zero, that end a ride the IMU
  /// measures: its height is integrated up to there, without waiting for the odometry to calm down
  /// (a person walks out of the cabin at once).
  double ride_stop_s = 1.0;
  double max_transition_s = 120.0;     ///< forced close after this long
  double tracking_loss_timeout = 2.0;  ///< s
  bool aerial_platform = false;        ///< robot.platform == aerial
};

struct FloorSample {
  Timestamp stamp = 0;
  bool tracking_ok = false;                            ///< odometry valid for this sample
  Eigen::Vector3d position = Eigen::Vector3d::Zero();  ///< odometry position (valid if tracking_ok)
  /// Base z axis in the odometry frame (unit): how the robot is tilted. A real ramp tilts it by the
  /// slope; a drift of the odometry in z does not.
  std::optional<Eigen::Vector3d> up_axis;
  std::optional<double> baro_height;         ///< m, filtered, relative to boot
  std::optional<double> imu_vertical_accel;  ///< m/s^2, gravity-free, world frame
};

struct FloorEvent {
  enum class Kind { None, TransitionStarted, TransitionEnded, TrackingLost } kind = Kind::None;
  TransitionType type = TransitionType::Unknown;  ///< for TransitionEnded
  FloorId from = 0, to = 0;
  double delta_z = 0;  ///< m (barometric for elevators, odometric otherwise)
  bool continuous_tracking = false;
  /// Elevator or manual change: the odometry saw no vertical motion. The manager's altitude already
  /// includes `reset_dz`; the frontend should only restart (empty map), without shifting its z.
  bool needs_frontend_reset = false;
  double reset_dz = 0;
  /// TransitionStarted: the height change was seen by the barometer/IMU but not by the odometry, so
  /// the frontend should restart now to avoid matching the old floor's (possibly identical) map.
  bool vertical_unobserved = false;
  /// TransitionEnded with `from == to`: the transition the odometry opened ends at the onset of a
  /// ride the IMU now measures, which stays open (its own TransitionEnded follows). What was mapped
  /// until then belongs to the floor it left.
  bool ride_follows = false;
};

/// State machine OnFloor / InTransition(type) / Lost. Pure logic over scalar signals, so it is
/// tested with synthetic profiles and independent of any sensor frontend.
class FloorManager {
 public:
  explicit FloorManager(const FloorManagerConfig& config);
  /// Destination of the developer traces (debug_channel "ALT" and "FLOORS").
  void set_logger(std::shared_ptr<Logger> logger) { log_ = std::move(logger); }

  FloorEvent update(const FloorSample& s);
  /// End of the run with a transition still open: decides it now, with the last sample, as when
  /// max_transition_s runs out (a whole number of floors closes on a level, otherwise it stays on
  /// its floor). Returns the TransitionEnded event, or nothing when no transition is open.
  FloorEvent finish();

  /// Operator request (`~/mark_floor_change`): closes the current floor and opens the next one.
  FloorEvent mark_floor_change(TransitionType type, FloorId target = -1);

  FloorState state() const { return state_; }
  /// True while a transition opened by the barometer/IMU (not by the odometry) is in progress.
  bool in_unobserved_transition() const {
    return state_ == FloorState::InTransition && acc_.unobserved;
  }
  /// True while a transition the odometry is measuring (stairs, ramp) is in progress.
  bool in_observed_transition() const {
    return state_ == FloorState::InTransition && !acc_.unobserved;
  }
  /// Vertical speed (m/s) over the last second of tracked odometry while a transition the odometry
  /// is measuring is in progress; 0 otherwise.
  double climb_rate() const;
  /// Floor id for keyframes created now; -1 while a transition is in progress.
  FloorId floor() const { return state_ == FloorState::OnFloor ? floor_ : kUnassignedFloor; }
  FloorId last_floor() const { return floor_; }
  double altitude() const { return altitude_; }
  size_t num_floors() const { return levels_.size(); }
  double level_altitude(FloorId f) const { return levels_.at(static_cast<size_t>(f)); }
  /// The odometry was restarted with its height shifted by `dz` (a correction, not motion): keeps
  /// the altitude unchanged and forgets the vertical-speed history.
  void odometry_rebased(double dz);
  /// Creates a new level at the current altitude (used after an unobservable change).
  FloorId open_new_floor();
  /// Re-assigns the current floor (after a relocalization).
  void set_floor(FloorId f) {
    floor_ = f;
    state_ = FloorState::OnFloor;
  }

 private:
  Timestamp last_ride_dbg_ = 0;  ///< FLOORS debug channel: last trace of a ride's speed
  struct Acc {
    Timestamp t0 = 0;
    double alt0 = 0, baro0 = 0;
    bool had_baro = false;
    double s_h = 0;
    Eigen::Vector3d last_pos = Eigen::Vector3d::Zero();
    std::vector<std::pair<double, double>> profile;  ///< (horizontal run, altitude)
    Timestamp calm_since = 0;
    Timestamp plateau_since = 0;    ///< start of the current uninterrupted calm
    double plateau_h0 = 0;          ///< s_h at that moment
    bool unobserved = false;        ///< opened by baro/IMU while the odometry stayed put
    Timestamp imu_onset = 0;        ///< start of the vertical-acceleration window that triggered it
    Timestamp az_over_since = 0;    ///< opened by the odometry: since when the IMU accelerates
    int ride_sign = 0;              ///< an IMU-measured ride: +1 up, -1 down, 0 not known yet
    double ride_push = 0;           ///< m/s, its acceleration along that sign (the cabin starting)
    double ride_brake = 0;          ///< m/s, its acceleration against it (the cabin braking)
    Timestamp last_ride_stamp = 0;  ///< its last sample
    Timestamp last_pulse = 0;       ///< its last pulse (a cabin starting or braking)
    double ride_s_h0 = 0;           ///< m, the horizontal run when it started
    Timestamp ride_calm_since = 0;  ///< since when it has looked stopped
    Timestamp imu_end = 0;          ///< where the IMU's integration ends (0: now)
    double ride_ref = 0;            ///< m/s^2, the vertical acceleration before its onset
    double ride_v = 0;              ///< m/s, its speed from the 1 s means (from `ride_ref`)
    double ride_v_peak = 0;         ///< m/s, the largest |ride_v|
    Timestamp pulse1_end = 0;       ///< where its first pulse ended
    double pulse1_v = 0;            ///< m/s, ride_v there
    Timestamp gait_since = 0;       ///< since when the IMU has jolted as steps do
  };
  struct Motion {
    double vz = 0;            ///< m/s, vertical speed of the odometry altitude (0 without odometry)
    double dh = 0;            ///< m, horizontal displacement since the previous sample
    bool lost_event = false;  ///< the odometry has been lost for longer than tracking_loss_timeout
  };
  void ingest_height_sensors(const FloorSample& s);
  Motion update_odometry(const FloorSample& s);
  FloorEvent update_on_floor(const FloorSample& s, const Motion& m);
  FloorEvent update_in_transition(const FloorSample& s, const Motion& m);
  double tolerance() const;
  bool valid_level_change(double dz) const;
  FloorEvent close_transition(const FloorSample& s, bool plateau = false);
  /// The decision of a settled (or forced) transition; nothing when it must keep waiting.
  FloorEvent decide_transition(const FloorSample& s, bool forced);
  FloorSample last_sample_;
  double median_slope() const;
  FloorId assign_level(double altitude, double tol_override = -1.0, double sigma = 0.0);

  FloorManagerConfig cfg_;
  FloorState state_ = FloorState::OnFloor;
  FloorId floor_ = 0;
  std::vector<double> levels_;
  std::vector<double> level_sigma_;  ///< m, how well each level's altitude is known
  double altitude_ = 0;
  double altitude_offset_ = 0;
  Timestamp last_tracked_ = 0;  ///< stamp of the last sample with valid odometry
  bool started_ = false;
  // OnFloor reference
  double ref_alt_ = 0;
  std::optional<double> baro_ref_;
  // vertical speed estimation
  std::deque<std::pair<Timestamp, double>> alt_hist_;
  std::deque<std::pair<Timestamp, double>> run_win_;  ///< horizontal run in a transition, last 3 s
  std::deque<std::pair<Timestamp, double>> odom_vz_win_;  ///< tracked vertical speed, last 6 s
  std::deque<std::tuple<Timestamp, double, double>>
      slope_win_;  ///< (stamp, cumulative horizontal run, altitude)
  double hrun_ = 0;
  Timestamp settle_until_ = 0;  ///< after an elevator the restarted odometry keeps settling: the
                                ///< reference follows it until then
  Timestamp last_ref_stamp_ =
      0;  ///< previous OnFloor sample, for the time-based reference tracking
  Eigen::Vector3d last_pos_dbg_ = Eigen::Vector3d::Zero();
  int alt_dbg_ = 0;  // debug print decimation
  std::shared_ptr<Logger> log_ = std::make_shared<NullLogger>();
  bool climbing_long_ = false;
  bool climbing_ =
      false;  ///< sustained climb per metre travelled (a gentle ramp is slower than calm_vz)
  std::deque<std::pair<Timestamp, double>> baro_hist_;
  std::deque<std::pair<Timestamp, double>>
      az_hist_;  ///< recent vertical accelerations (bias removed)
  double az_bias_ = 0;
  Timestamp az_over_since_ = 0;
  std::vector<std::pair<double, double>> filtered_az(Timestamp onset, Timestamp end) const;
  double imu_speed_gain(Timestamp onset, Timestamp end) const;
  std::optional<double> odom_max_vz_since(Timestamp t) const;
  double imu_delta_z(Timestamp onset, Timestamp end) const;
  double az_mean(Timestamp now, double window_s) const;
  double az_mean_between(Timestamp from, Timestamp to) const;
  double az_spread(Timestamp from, Timestamp to) const;
  std::optional<Timestamp> push_before(Timestamp onset, double v_brake) const;
  Eigen::Vector3d last_pos_ = Eigen::Vector3d::Zero();
  bool have_last_pos_ = false;
  Timestamp lost_since_ = 0;
  Acc acc_;
  TransitionType pending_type_ = TransitionType::Unknown;
};

}  // namespace strataslam
