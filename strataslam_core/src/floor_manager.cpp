#include "strataslam/floors/floor_manager.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strataslam {

namespace {
/// ns of vertical acceleration kept on a floor: a ride whose push stayed under the departure
/// threshold is found back from its brake (Elevator-VIGS Residential3: 17 s before it).
constexpr Timestamp kAzHistory = 60'000'000'000;
/// m/s^2, spread of the vertical acceleration over 1 s that a cabin stays under (0.1-0.4 on
/// Elevator-VIGS; the steps of a walk or a stair jolt it by 0.5-4.4).
constexpr double kCabinSpread = 0.45;
double rate(const std::deque<std::pair<Timestamp, double>>& h) {
  if (h.size() < 2) return 0.0;
  const double dt = (h.back().first - h.front().first) * 1e-9;
  return dt > 1e-6 ? (h.back().second - h.front().second) / dt : 0.0;
}
void push(std::deque<std::pair<Timestamp, double>>& h, Timestamp t, double v, double window_s) {
  h.push_back({t, v});
  while (h.size() > 2 && (t - h.front().first) * 1e-9 > window_s) h.pop_front();
}
}  // namespace

FloorManager::FloorManager(const FloorManagerConfig& config) : cfg_(config) {
  levels_.push_back(0.0);
  level_sigma_.push_back(0.0);
}

/// Trimmed mean (without the 20 % highest and the 20 % lowest) of the vertical acceleration over
/// the last `window_s`: an accelerometer on a ground robot gives impulses (a wheel hitting an edge
/// reads tens of m/s^2 for one sample) that a plain mean would take for an elevator, while a
/// sustained acceleration, which is what an elevator is, and a symmetric oscillation (walking) are
/// not distorted.
double FloorManager::az_mean(Timestamp now, double window_s) const {
  std::vector<double> v;
  for (auto it = az_hist_.rbegin(); it != az_hist_.rend() && (now - it->first) * 1e-9 <= window_s;
       ++it)
    v.push_back(it->second);
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t cut = v.size() / 5;
  double sum = 0;
  for (size_t k = cut; k < v.size() - cut; ++k) sum += v[k];
  return sum / static_cast<double>(v.size() - 2 * cut);
}

/// The spread of the vertical acceleration between `from` and `to`: half its 10-90 % range, the
/// jolts of a gait against the smooth push of a cabin.
double FloorManager::az_spread(Timestamp from, Timestamp to) const {
  std::vector<double> v;
  for (const auto& [t, az] : az_hist_)
    if (t >= from && t <= to) v.push_back(az);
  if (v.size() < 5) return 0.0;
  std::sort(v.begin(), v.end());
  return 0.5 * (v[v.size() * 9 / 10] - v[v.size() / 10]);
}

/// The push of a ride whose brake, from `onset`, gained `v_brake`: back through the history, an
/// acceleration the other way that gained most of that speed, before a cabin's calm. Its start, or
/// nullopt when there is none (a walk jolts the calm, and its 1 s means do not add up to a push).
std::optional<Timestamp> FloorManager::push_before(Timestamp onset, double v_brake) const {
  constexpr double kMatch = 0.6;  // of the brake's speed, gained by the push
  constexpr double kEdge = 0.05;  // m/s^2 along the push: where it starts and ends
  constexpr Timestamp kSecond = 1'000'000'000;
  const double along = v_brake > 0 ? -1.0 : 1.0;  // the push accelerated the other way
  const double ref = acc_.ride_ref;
  double v = 0;
  Timestamp prev = onset, push_end = 0, matched = 0;
  for (auto it = az_hist_.rbegin(); it != az_hist_.rend(); ++it) {
    const Timestamp t = it->first;
    if (t > onset) continue;
    const double a = along * (az_mean_between(t - kSecond, t) - ref);
    v += a * (prev - t) * 1e-9;
    prev = t;
    if (push_end == 0 && a > kEdge) push_end = t;
    if (matched == 0 && v >= kMatch * std::abs(v_brake)) matched = t;
    if (matched != 0 && a < kEdge) {
      if (v > 2.0 * std::abs(v_brake)) return std::nullopt;  // not the same ride's speed
      // The cabin between the push and the brake: calm.
      if (push_end - kSecond > t && az_spread(t, push_end - kSecond) >= kCabinSpread)
        return std::nullopt;
      if (onset - kSecond > push_end && az_spread(push_end, onset - kSecond) >= kCabinSpread)
        return std::nullopt;
      return t - kSecond;
    }
  }
  return std::nullopt;
}

/// The trimmed mean of the vertical acceleration between `from` and `to` (0 without samples).
double FloorManager::az_mean_between(Timestamp from, Timestamp to) const {
  std::vector<double> v;
  for (const auto& [t, az] : az_hist_)
    if (t >= from && t <= to) v.push_back(az);
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t cut = v.size() / 5;
  double sum = 0;
  for (size_t k = cut; k < v.size() - cut; ++k) sum += v[k];
  return sum / static_cast<double>(v.size() - 2 * cut);
}

/// The vertical acceleration between `onset` and `end`, (t seconds, az), through a 5-point median:
/// the impulses of the sensor would otherwise be integrated.
std::vector<std::pair<double, double>> FloorManager::filtered_az(Timestamp onset,
                                                                 Timestamp end) const {
  std::vector<std::pair<double, double>> a;
  for (const auto& h : az_hist_) {
    if (h.first >= onset && h.first <= end) a.push_back({h.first * 1e-9, h.second});
  }
  std::vector<double> f(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    std::vector<double> w;
    for (size_t j = (i >= 2 ? i - 2 : 0); j < std::min(a.size(), i + 3); ++j)
      w.push_back(a[j].second);
    std::sort(w.begin(), w.end());
    f[i] = w[w.size() / 2];
  }
  for (size_t i = 0; i < a.size(); ++i) a[i].second = f[i];
  return a;
}

/// Double integration of the vertical acceleration between `onset` and `end`, with a zero-velocity
/// update at the end (the cabin is at rest again): the residual velocity is spread linearly, which
/// removes most of the effect of an accelerometer bias.
double FloorManager::imu_delta_z(Timestamp onset, Timestamp end) const {
  const auto a = filtered_az(onset, end);
  if (a.size() < 3) return 0.0;
  std::vector<double> v(a.size(), 0.0);
  for (size_t i = 1; i < a.size(); ++i)
    v[i] = v[i - 1] + 0.5 * (a[i].second + a[i - 1].second) * (a[i].first - a[i - 1].first);
  const double T = a.back().first - a.front().first;
  const double v_end = v.back();
  double dz = 0;
  for (size_t i = 1; i < a.size(); ++i) {
    const double vi = v[i] - v_end * (a[i].first - a.front().first) / T;
    const double vp = v[i - 1] - v_end * (a[i - 1].first - a.front().first) / T;
    dz += 0.5 * (vi + vp) * (a[i].first - a[i - 1].first);
  }
  return dz;
}

/// Vertical speed gained from `onset` to `end`: the acceleration integrated once.
double FloorManager::imu_speed_gain(Timestamp onset, Timestamp end) const {
  const auto a = filtered_az(onset, end);
  double v = 0;
  for (size_t i = 1; i < a.size(); ++i)
    v += 0.5 * (a[i].second + a[i - 1].second) * (a[i].first - a[i - 1].first);
  return v;
}

/// The fastest vertical speed the odometry measured from `t` on (its 1 s rate while it tracked),
/// or nullopt when it never tracked then.
std::optional<double> FloorManager::odom_max_vz_since(Timestamp t) const {
  std::optional<double> m;
  for (const auto& h : odom_vz_win_)
    if (h.first >= t) m = std::max(m.value_or(0.0), std::abs(h.second));
  return m;
}

double FloorManager::tolerance() const {
  return cfg_.level_tolerance_factor *
         (cfg_.floor_height_prior ? *cfg_.floor_height_prior : cfg_.min_vertical_change);
}

/// With a prior, a change is a real floor change only if it is close to a whole number of floors:
/// the half-way landing of a U-shaped stair or ramp is not a level.
bool FloorManager::valid_level_change(double dz) const {
  const double a = std::abs(dz);
  if (cfg_.floor_height_prior) {
    const double n = a / *cfg_.floor_height_prior;
    return std::round(n) >= 1.0 && std::abs(n - std::round(n)) <= cfg_.level_tolerance_factor;
  }
  return a >= cfg_.min_vertical_change;
}

/// The level at `altitude`, known within `sigma` (m, one standard deviation), or a new one. Two
/// levels match within the tolerance, or within twice their combined uncertainty when that is
/// larger: a level reached by an IMU-measured ride is only known to a few percent of the ride.
FloorId FloorManager::assign_level(double altitude, double tol_override, double sigma) {
  const double tol = tol_override > 0 ? tol_override : tolerance();
  FloorId best = -1;
  double best_d = 1e9;
  for (size_t i = 0; i < levels_.size(); ++i) {
    const double d = std::abs(levels_[i] - altitude);
    if (d < std::max(tol, 2.0 * std::hypot(sigma, level_sigma_[i])) && d < best_d) {
      best_d = d;
      best = static_cast<FloorId>(i);
    }
  }
  if (best >= 0) return best;
  levels_.push_back(altitude);
  level_sigma_.push_back(sigma);
  return static_cast<FloorId>(levels_.size() - 1);
}

void FloorManager::odometry_rebased(double dz) {
  altitude_offset_ -= dz;
  alt_hist_.clear();
  slope_win_.clear();
  climbing_long_ = false;
  climbing_ = false;
  have_last_pos_ = false;
}

double FloorManager::climb_rate() const {
  if (state_ != FloorState::InTransition || acc_.unobserved) return 0.0;
  return rate(alt_hist_);  // last second of tracked altitude
}

FloorId FloorManager::open_new_floor() {
  // An unobservable change: assume one floor up (the direction is unknowable; the operator or the
  // anchors can correct it). The altitude continues from that level.
  const double step = cfg_.floor_height_prior ? *cfg_.floor_height_prior : cfg_.min_vertical_change;
  const double new_alt = levels_.at(static_cast<size_t>(floor_)) + step;
  altitude_offset_ += new_alt - altitude_;
  altitude_ = new_alt;
  levels_.push_back(altitude_);
  level_sigma_.push_back(level_sigma_.at(static_cast<size_t>(floor_)));
  floor_ = static_cast<FloorId>(levels_.size() - 1);
  state_ = FloorState::OnFloor;
  ref_alt_ = altitude_;
  slope_win_.clear();
  climbing_long_ = false;
  climbing_ = false;
  return floor_;
}

double FloorManager::median_slope() const {
  // Slope over windows of `slope_window_m` of horizontal run; median of windows that change height.
  const auto& p = acc_.profile;
  std::vector<double> slopes;
  size_t j = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    while (j < p.size() && p[j].first - p[i].first < cfg_.slope_window_m) ++j;
    if (j >= p.size()) break;
    const double dz = p[j].second - p[i].second, dh = p[j].first - p[i].first;
    if (std::abs(dz) > 0.05 && dh > 0.1) slopes.push_back(std::abs(dz) / dh);
  }
  if (slopes.empty()) return 0.0;
  std::sort(slopes.begin(), slopes.end());
  return slopes[slopes.size() / 2];
}

FloorEvent FloorManager::close_transition(const FloorSample& s, bool plateau) {
  FloorEvent ev;
  const double dz_odom = altitude_ - acc_.alt0;
  const bool have_baro = acc_.had_baro && s.baro_height.has_value() && baro_ref_.has_value();
  const double dz_baro = have_baro ? (*s.baro_height - acc_.baro0) : 0.0;
  const bool elevator_baro = have_baro && std::abs(dz_baro) >= cfg_.elevator_baro_min &&
                             std::abs(dz_odom) < cfg_.elevator_static_ratio * std::abs(dz_baro);
  // Without a barometer: the IMU's double-integrated vertical motion, with the odometry static.
  const double dz_imu = acc_.unobserved && !have_baro
                            ? imu_delta_z(acc_.imu_onset, acc_.imu_end ? acc_.imu_end : s.stamp)
                            : 0.0;
  const bool elevator_imu = acc_.unobserved && !have_baro &&
                            std::abs(dz_imu) >= cfg_.elevator_imu_min &&
                            std::abs(dz_odom) < cfg_.elevator_static_ratio * std::abs(dz_imu);
  const bool elevator = elevator_baro || elevator_imu;
  if (debug_channel("FLOORS"))
    log_->log(LogLevel::Debug, strprintf("close: dz_odom=%.3f dz_baro=%.3f dz_imu=%.3f "
                                         "unobserved=%d alt0=%.3f alt=%.3f onset_to_now=%.1fs",
                                         dz_odom, dz_baro, dz_imu, acc_.unobserved, acc_.alt0,
                                         altitude_, (s.stamp - acc_.imu_onset) * 1e-9));
  const FloorId from = floor_;
  double dz = dz_odom;
  TransitionType type = TransitionType::Unknown;
  if (elevator) {
    type = TransitionType::Elevator;
    dz = elevator_baro ? dz_baro : dz_imu;
    // Odometry did not follow: shift the altitude so that it continues from the measured height.
    altitude_offset_ += dz - dz_odom;
    altitude_ += dz - dz_odom;
    ev.needs_frontend_reset = true;
    ev.reset_dz = dz - dz_odom;
    ev.continuous_tracking = true;  // x, y and yaw stay continuous inside the cabin
    settle_until_ = s.stamp + static_cast<Timestamp>(30.0e9);
  } else if (cfg_.aerial_platform ||
             acc_.s_h < cfg_.aerial_max_horizontal * std::max(std::abs(dz_odom), 0.1)) {
    type = TransitionType::Aerial;
    ev.continuous_tracking = true;
  } else {
    const double slope = median_slope();
    type = slope < std::tan(cfg_.ramp_max_slope_deg * M_PI / 180.0) ? TransitionType::Ramp
                                                                    : TransitionType::Stairs;
    ev.continuous_tracking = true;
  }
  // A forced close (the odometry never calmed down) whose height is not a whole number of floors is
  // terrain, not a floor: stay on this floor instead of opening a level at an arbitrary altitude.
  // A plateau is a level whatever its height; two levels closer than the plateau change are one.
  // The level is placed by what the transition measured, from the level it left: not at the
  // absolute altitude, which carries the odometry's vertical drift accumulated while walking on the
  // floor before (S03, seed 7014: 0.34 m of it made the second floor 3.33 m high for a 2.99 m
  // climb; audit AUD-007).
  const double target = levels_.at(static_cast<size_t>(from)) + dz;
  // A ride the IMU measured double-integrates the accelerometer: within about 5 % of its height on
  // Elevator-VIGS (Office1-H 0.5 %, Residential2 0.7 %, Mall-E 8 %). Office4 rode 14.75 m up and
  // walked 15.43 m down the stairs to where it had started: 0.68 m off, over the 0.6 m tolerance,
  // it opened a third level instead of coming back to the first.
  constexpr double kImuRideRelError = 0.05;
  double sigma = level_sigma_.at(static_cast<size_t>(from));
  if (elevator_imu) sigma = std::hypot(sigma, kImuRideRelError * dz);
  const FloorId to =
      (elevator || valid_level_change(dz) || plateau)
          ? assign_level(target, plateau ? 0.5 * cfg_.plateau_min_change : -1.0, sigma)
          : from;
  ev.kind = FloorEvent::Kind::TransitionEnded;
  ev.type = type;
  ev.from = from;
  ev.to = to;
  ev.delta_z = dz;
  floor_ = to;
  state_ = FloorState::OnFloor;
  ref_alt_ = altitude_;
  slope_win_.clear();
  climbing_long_ = false;  // the altitude may have been re-based: the old climb says nothing now
  climbing_ = false;
  if (s.baro_height) baro_ref_ = s.baro_height;
  return ev;
}

FloorEvent FloorManager::update(const FloorSample& s) {
  last_sample_ = s;
  ingest_height_sensors(s);
  const Motion m = update_odometry(s);
  if (state_ == FloorState::OnFloor || state_ == FloorState::Lost) return update_on_floor(s, m);
  return update_in_transition(s, m);
}

/// First sample, IMU vertical acceleration and barometer histories (they work without odometry).
void FloorManager::ingest_height_sensors(const FloorSample& s) {
  if (!started_) {
    started_ = true;
    altitude_offset_ = s.tracking_ok ? -s.position.z() : 0.0;  // floor 0 is altitude 0
    ref_alt_ = 0.0;
    levels_[0] = 0.0;
  }
  if (s.imu_vertical_accel) {
    az_hist_.push_back({s.stamp, *s.imu_vertical_accel - az_bias_});
    // 12 s, or from the onset of a ride the IMU is measuring: a ride is decided once the height has
    // been calm for a while after it, often more than 12 s after it started (Elevator-VIGS Mall-E:
    // 13 s).
    const Timestamp keep_from = state_ == FloorState::InTransition && acc_.unobserved
                                    ? std::min(acc_.imu_onset, s.stamp - kAzHistory)
                                    : s.stamp - kAzHistory;
    while (az_hist_.size() > 2 && az_hist_.front().first < keep_from) az_hist_.pop_front();
  }
  if (s.baro_height) {
    if (!baro_ref_) baro_ref_ = s.baro_height;
    push(baro_hist_, s.stamp, *s.baro_height, 1.5);
  }
}

/// Odometry-dependent quantities: altitude, vertical speed, horizontal run and the climbing flags.
FloorManager::Motion FloorManager::update_odometry(const FloorSample& s) {
  // Only refreshed while the odometry is valid; the barometer and the IMU keep working when it is
  // not (an elevator door closing is where visual tracking breaks).
  Motion m;
  if (!s.tracking_ok) {
    if (lost_since_ == 0) lost_since_ = s.stamp;
    if (state_ != FloorState::Lost && (s.stamp - lost_since_) * 1e-9 > cfg_.tracking_loss_timeout)
      m.lost_event = true;
  } else {
    lost_since_ = 0;
    last_tracked_ = s.stamp;
    altitude_ = s.position.z() + altitude_offset_;
    push(alt_hist_, s.stamp, altitude_, 1.0);
    m.vz = rate(alt_hist_);
    push(odom_vz_win_, s.stamp, m.vz, 6.0);
    m.dh = have_last_pos_ ? (Eigen::Vector2d(s.position.x(), s.position.y()) -
                             Eigen::Vector2d(last_pos_.x(), last_pos_.y()))
                                .norm()
                          : 0.0;
    last_pos_ = s.position;
    have_last_pos_ = true;
    // A gentle ramp climbs slower than calm_vz: while moving, judge the climb per metre travelled
    // too.
    hrun_ += m.dh;
    slope_win_.emplace_back(s.stamp, hrun_, altitude_);
    while (slope_win_.size() > 2 &&
           (s.stamp - std::get<0>(slope_win_.front())) * 1e-9 > cfg_.slope_long_window_s)
      slope_win_.pop_front();
    // Short window: is it climbing now. Long window: is it still in the same climb (the odometry z
    // is noisy, a gentle ramp flickers the short one).
    auto slope_over = [&](double window_s, double min_run, double slope) {
      auto it = slope_win_.begin();
      while (it + 1 != slope_win_.end() && (s.stamp - std::get<0>(*it)) * 1e-9 > window_s) ++it;
      const double run = hrun_ - std::get<1>(*it);
      return run >= min_run && std::abs(altitude_ - std::get<2>(*it)) > slope * run;
    };
    climbing_ = slope_over(cfg_.calm_slope_window_s, 1.0, cfg_.calm_slope);
    climbing_long_ = slope_over(cfg_.slope_long_window_s, 3.0, cfg_.calm_slope_long) &&
                     slope_over(cfg_.calm_slope_window_s, 1.0, cfg_.calm_slope_long * 0.5);
    const double run = hrun_ - std::get<1>(slope_win_.front());
    if (debug_channel("ALT") && ++alt_dbg_ % debug_channel("ALT") == 0) {
      // slope measured over the window against the one the tilt of the robot implies along its
      // heading
      double meas = 0, expect = 0;
      if (run > 1.0 && s.up_axis && !slope_win_.empty()) {
        meas = (altitude_ - std::get<2>(slope_win_.front())) / run;
        const Eigen::Vector2d d = (s.position - last_pos_dbg_).head<2>();
        if (d.norm() > 1e-6 && std::abs((*s.up_axis).z()) > 0.5)
          expect = -((*s.up_axis).head<2>().dot(d / d.norm())) / (*s.up_axis).z();
      }
      log_->log(
          LogLevel::Debug,
          strprintf("alt t=%.1f z=%.2f vz=%.3f run=%.2f climbing=%d state=%d meas=%.3f tilt=%.3f",
                    (s.stamp - 1000000000000LL) * 1e-9, altitude_, m.vz, run,
                    climbing_ | (climbing_long_ << 1), int(state_), meas, expect));
    }
    last_pos_dbg_ = s.position;
  }
  return m;
}

/// On a floor: detects a departure (odometry, barometer or IMU) and tracks the slow drift of the
/// floor reference otherwise.
FloorEvent FloorManager::update_on_floor(const FloorSample& s, const Motion& m) {
  FloorEvent ev;
  state_ = FloorState::OnFloor;
  const bool odom_departure = s.tracking_ok && std::abs(altitude_ - ref_alt_) > cfg_.depart_dz;
  const bool baro_departure =
      baro_ref_ && s.baro_height && std::abs(*s.baro_height - *baro_ref_) > cfg_.baro_depart;
  // A sustained 1 s mean of vertical acceleration (walking bobs average out) while the odometry
  // sees no vertical motion: an elevator accelerating. Also while the odometry is lost, however
  // long: the cabin door closing is where tracking breaks, and the ride often starts after
  // tracking_loss_timeout (an earlier `!lost_event` here switched the IMU off from then on: S11
  // lost its third floor that way). The odometry's altitude only counts while it tracks: lost, it
  // stays where it was, and closing doors often make it jump first (Elevator-VIGS Office2-E: 0.2 m,
  // under depart_dz, kept the IMU from opening the 3.3 m ride that followed).
  bool imu_departure = false;
  Timestamp imu_onset = 0;
  if (s.imu_vertical_accel &&
      (!s.tracking_ok ||
       (std::abs(m.vz) < 0.1 && std::abs(altitude_ - ref_alt_) < 0.5 * cfg_.depart_dz))) {
    if (std::abs(az_mean(s.stamp, 1.0)) > cfg_.imu_depart_accel) {
      if (az_over_since_ == 0) az_over_since_ = s.stamp;
      if ((s.stamp - az_over_since_) * 1e-9 >= cfg_.imu_depart_s) {
        imu_departure = true;
        imu_onset = az_over_since_ - static_cast<Timestamp>(1.0e9);
      }
    } else {
      az_over_since_ = 0;
    }
  } else {
    az_over_since_ = 0;
  }
  if (odom_departure || baro_departure || imu_departure) {
    if (debug_channel("FLOORS"))
      log_->log(LogLevel::Debug,
                strprintf("depart: t=%.1f odom=%d baro=%d imu=%d alt=%.2f ref=%.2f vz=%.3f "
                          "climbing=%d az1s=%.3f since_settle_start=%.1fs",
                          (s.stamp - 1000000000000LL) * 1e-9, odom_departure, baro_departure,
                          imu_departure, altitude_, ref_alt_, m.vz, climbing_,
                          s.imu_vertical_accel ? az_mean(s.stamp, 1.0) : 0.0,
                          settle_until_ ? (s.stamp - settle_until_) * 1e-9 + 30.0 : -1.0));
    az_over_since_ = 0;
    state_ = FloorState::InTransition;
    acc_ = Acc();
    run_win_.clear();
    acc_.t0 = s.stamp;
    acc_.alt0 = ref_alt_;
    acc_.had_baro = baro_ref_.has_value() && s.baro_height.has_value();
    acc_.baro0 = baro_ref_.value_or(0.0);
    acc_.last_pos = s.position;
    acc_.profile.push_back({0.0, altitude_});
    acc_.unobserved = !odom_departure;
    acc_.imu_onset = imu_onset ? imu_onset : s.stamp - static_cast<Timestamp>(1.0e9);
    ev.kind = FloorEvent::Kind::TransitionStarted;
    ev.vertical_unobserved = acc_.unobserved;
    return ev;
  }
  // Slow tracking of the floor reference (drift) while standing on a floor.
  // Time constant of 5 s (0.02 per sample at 10 Hz), so a slow odometry drift is followed at any
  // sample rate.
  if (s.tracking_ok && ((!climbing_ && !climbing_long_) || s.stamp < settle_until_)) {
    const double dt =
        last_ref_stamp_ ? std::clamp((s.stamp - last_ref_stamp_) * 1e-9, 0.0, 5.0) : 0.1;
    ref_alt_ += (1.0 - std::exp(-dt / 5.0)) * (altitude_ - ref_alt_);
  }
  last_ref_stamp_ = s.stamp;
  if (s.baro_height && baro_ref_) *baro_ref_ += 0.02 * (*s.baro_height - *baro_ref_);
  // Accelerometer bias along the vertical: slow mean while nothing is accelerating.
  if (s.imu_vertical_accel && std::abs(az_mean(s.stamp, 1.0)) < 0.1)
    az_bias_ += 0.002 * (*s.imu_vertical_accel - az_bias_);
  if (ev.kind == FloorEvent::Kind::None && m.lost_event) ev.kind = FloorEvent::Kind::TrackingLost;
  return ev;
}

/// In a transition: waits until the height settles, then decides whether it was a floor change.
FloorEvent FloorManager::update_in_transition(const FloorSample& s, const Motion& m) {
  FloorEvent ev;
  const double vb = s.baro_height ? rate(baro_hist_) : 0.0;
  if (s.tracking_ok) {
    acc_.s_h += m.dh;
    acc_.profile.push_back({acc_.s_h, altitude_});
  }
  push(run_win_, s.stamp, acc_.s_h, 3.0);
  bool calm = std::abs(m.vz) < cfg_.calm_vz && std::abs(vb) < 0.1 &&
              (!s.imu_vertical_accel || std::abs(az_mean(s.stamp, 1.0)) < cfg_.imu_calm_accel);
  if (climbing_ || climbing_long_) calm = false;
  // Opened by the odometry, but the IMU accelerates vertically as it does to open a ride while the
  // odometry does not follow: an elevator whose closing doors made the odometry jump before it was
  // lost in the cabin (Elevator-VIGS Mall-E: 0.33 m at the doors, then the 4.5 m ride unseen and
  // the floor closed as a 1.5 m ramp), or one it tracks, seeing the cabin still (Office1: 14.3 m
  // left out). The IMU measures it from then on. While the odometry tracks, it must have stayed
  // still while the IMU gained the speed of a cabin: a hand-held rig lowered and raised again, a
  // stair or the bob of the gait move both, with 1 s means of up to 1.5 m/s^2 for up to 2.5 s
  // (Campus1, Office2), as long as a cabin's push (Station1: 1.1 s).
  if (!acc_.unobserved && !acc_.had_baro && s.imu_vertical_accel) {
    if (std::abs(az_mean(s.stamp, 1.0)) > cfg_.imu_depart_accel) {
      if (acc_.az_over_since == 0) acc_.az_over_since = s.stamp;
      if ((s.stamp - acc_.az_over_since) * 1e-9 >= cfg_.imu_depart_s) {
        constexpr double kCabinSpeed = 0.3;  // m/s gained: a cabin starting
        // The odometry followed less than half of it (or 0.1 m/s): a vertical move it tracks is
        // followed in full, while a cabin is still to it, or barely followed when degenerate inside
        // (Elevator-VIGS Factory-H: 0.22 m/s of the cabin's 0.7).
        constexpr double kStillVz = 0.1;
        constexpr double kSeenShare = 0.5;
        // And it accelerates smoothly, as a cabin does (kCabinSpread): the steps of a stair the
        // odometry tracks half-way jolt it, and its 1 s mean then looks like a cabin's push
        // (Office4: a flight of stairs down read as a 2.4 m ride to a level of its own). Checked
        // again each sample: a ride that starts as the last steps into the cabin are taken is
        // taken once they stop.
        const Timestamp onset = acc_.az_over_since - static_cast<Timestamp>(1.0e9);
        // Lost, for the whole half second: a short loss while the rig swings is not a cabin.
        const bool lost_through =
            !s.tracking_ok && lost_since_ != 0 &&
            (s.stamp - std::max(acc_.az_over_since, lost_since_)) * 1e-9 >= cfg_.imu_depart_s;
        const double gain = imu_speed_gain(onset, s.stamp);
        const double odom_vz = s.tracking_ok ? odom_max_vz_since(onset).value_or(0.0) : 0.0;
        const bool unseen =
            std::abs(gain) >= kCabinSpeed &&
            (lost_through ||
             (s.tracking_ok && odom_vz < std::max(kStillVz, kSeenShare * std::abs(gain)) &&
              az_spread(s.stamp - 1'000'000'000, s.stamp) < kCabinSpread));
        if (unseen) {
          acc_.unobserved = true;
          acc_.imu_onset = onset;
          // The ride starts now: the odometry may have opened the transition long before (its
          // drift while walking), and max_transition_s from then closed rides half-way
          // (Elevator-VIGS Campus6: 15 s into a 27 s ride).
          acc_.t0 = onset;
          if (debug_channel("FLOORS"))
            log_->log(LogLevel::Debug,
                      strprintf("the IMU takes the transition at t=%.1f (onset %.1f): az1s=%.3f, "
                                "odometry %s (imu %.2f "
                                "m/s gained, odometry at most %.2f m/s), az spread %.2f m/s^2, "
                                "run %.2f m in 3 s",
                                s.stamp * 1e-9, onset * 1e-9, az_mean(s.stamp, 1.0),
                                s.tracking_ok ? "tracking" : "lost", gain, odom_vz,
                                az_spread(s.stamp - 1'000'000'000, s.stamp),
                                run_win_.empty() ? 0.0 : acc_.s_h - run_win_.front().second));
          // What the odometry's transition held until the onset happened on the floor it left:
          // it ends there, no floor change, and the ride goes on (Elevator-VIGS Office4: opened
          // 50 s before the ride, the walk on floor 0 was mapped on the floor the ride reached).
          ev.kind = FloorEvent::Kind::TransitionEnded;
          ev.type = TransitionType::Unknown;
          ev.from = ev.to = floor_;
          ev.ride_follows = true;
          return ev;
        }
      }
    } else {
      acc_.az_over_since = 0;
    }
  } else {
    acc_.az_over_since = 0;
  }
  // A ride the IMU measures: a push (the cabin starting) and a brake of the opposite sign, each the
  // integral of the 1 s mean of the vertical acceleration where it stands clear of the noise. While
  // the cabin moves the transition is not calm, even when it cruises without accelerating: a cruise
  // longer than the settling time closed the ride half-way (a 14 m ride read 2.7 m). The speed
  // integrated from the onset cannot say it: a bias and the tilt of the gravity estimate make it
  // drift (Elevator-VIGS Office1-H: 0.03 m/s^2, 0.94 m/s left after a 33 m ride), the pulses can.
  const bool imu_ride = acc_.unobserved && !acc_.had_baro && s.imu_vertical_accel.has_value();
  constexpr double kRideMinSpeed = 0.3;  // m/s pushed: slower is not a cabin moving
  // m/s^2: the 1 s mean of a cabin starting or braking (0.39-0.9 on Elevator-VIGS), over the
  // vibration of a cruising cabin (Factory-H: +-0.16, which counted as push and brake alike kept
  // the 50 m ride from ever stopping).
  constexpr double kPulseAccel = 0.2;
  constexpr double kBrakeShare = 0.6;  // of the push, braked: the cabin has stopped
  // Its accelerations are measured from the one before the onset, what the bias estimate left: it
  // drifts while the person walks (Elevator-VIGS Mall-M: to -0.13 m/s^2 in 70 s, the cabin then
  // read +0.12 at rest, never calm, and a -18 m ride was integrated through the walk out, -4.1 m).
  // Bounded by the pulses: more than one is no bias.
  constexpr Timestamp kRideRefWindow = 5'000'000'000;
  bool braked = false;
  if (imu_ride) {
    const auto pulse = [&](Timestamp t, double a1) {
      const double dt = acc_.last_ride_stamp ? (t - acc_.last_ride_stamp) * 1e-9 : 0.0;
      acc_.last_ride_stamp = t;
      if (std::abs(a1) > kPulseAccel) acc_.last_pulse = t;
      acc_.ride_v += a1 * dt;
      acc_.ride_v_peak = std::max(acc_.ride_v_peak, std::abs(acc_.ride_v));
      if (acc_.pulse1_end == 0 && acc_.last_pulse != 0 && std::abs(a1) <= kPulseAccel) {
        acc_.pulse1_end = t;
        acc_.pulse1_v = acc_.ride_v;
      }
      if (acc_.ride_sign == 0 && std::abs(a1) > cfg_.imu_depart_accel)
        acc_.ride_sign = a1 > 0 ? 1 : -1;
      if (acc_.ride_sign != 0) {
        const double along = acc_.ride_sign * a1;
        if (along > kPulseAccel) acc_.ride_push += along * dt;
        if (along < -kPulseAccel) acc_.ride_brake -= along * dt;
      }
    };
    if (acc_.last_ride_stamp == 0) {
      acc_.ride_s_h0 = acc_.s_h;
      acc_.ride_ref = std::clamp(az_mean_between(acc_.imu_onset - kRideRefWindow, acc_.imu_onset),
                                 -kPulseAccel, kPulseAccel);
      // Counted from the onset: a ride taken over from the odometry is 1.5 s into its push, which
      // then fell short of a cabin's and the brake that followed never counted (Elevator-VIGS
      // Station2: 0.20 m/s of a 0.9 m/s push, the ride ended by chance as the person walked on).
      for (const auto& [t, az] : az_hist_)
        if (t >= acc_.imu_onset && t < s.stamp)
          pulse(t, az_mean_between(t - 1'000'000'000, t) - acc_.ride_ref);
    }
    const double a1 = az_mean(s.stamp, 1.0) - acc_.ride_ref;
    pulse(s.stamp, a1);
    braked = acc_.ride_push >= kRideMinSpeed && acc_.ride_brake >= kBrakeShare * acc_.ride_push;
    if (debug_channel("FLOORS") && s.stamp - last_ride_dbg_ >= 1'000'000'000) {
      last_ride_dbg_ = s.stamp;
      log_->log(LogLevel::Debug,
                strprintf("ride: %.1fs from the onset, push=%.2f brake=%.2f m/s az1s=%.3f "
                          "(ref %.3f) tracking=%d run=%.1f m",
                          (s.stamp - acc_.imu_onset) * 1e-9, acc_.ride_push, acc_.ride_brake, a1,
                          acc_.ride_ref, s.tracking_ok, acc_.s_h - acc_.ride_s_h0));
    }
    if (acc_.ride_push >= kRideMinSpeed && !braked) calm = false;
  }
  if (!calm) {
    acc_.calm_since = 0;
    acc_.plateau_since = 0;
  } else {
    if (acc_.calm_since == 0) acc_.calm_since = s.stamp;
    if (acc_.plateau_since == 0) {
      acc_.plateau_since = s.stamp;
      acc_.plateau_h0 = acc_.s_h;
    }
  }
  // A ride the IMU measures ends when the cabin stops: braked and the acceleration calm. Closed
  // there, with the height integrated up to there (the zero-velocity update at that end removes a
  // constant bias exactly): the odometry's calm comes much later on a person who walks out of the
  // cabin (Elevator-VIGS Mall-E: never in the 22 s left, so the 4.5 m ride was integrated over the
  // walk too and read 7.1 m).
  // A ride has also ended when the odometry has walked out of the cabin, which no cabin is deep
  // enough to hold: the brake can be lost in what the cabin's bumps added to the push
  // (Elevator-VIGS Factory-H, 50 m) or a ride opened late keeps adding the walk's pulses
  // (Residential3: -116 m). The cabin had stopped by the time the walk began (the vertical speed is
  // zero there) and by the end of its last pulse (a gentle brake may stay under kPulseAccel).
  if (imu_ride) {
    constexpr double kWalkedOutM = 3.0;
    const bool stopped =
        braked && std::abs(az_mean(s.stamp, 1.0) - acc_.ride_ref) < cfg_.imu_calm_accel;
    if (!stopped)
      acc_.ride_calm_since = 0;
    else if (acc_.ride_calm_since == 0)
      acc_.ride_calm_since = s.stamp;
    // Walking: 3 m, at a walking pace over the last 3 s (half a metre a second). An odometry
    // degenerate in a closed cabin drifts too, slowly (Factory-H: 3 m in 38 s of a 65 s ride).
    constexpr double kWalkingRunM = 1.5;
    const bool walking = !run_win_.empty() && (s.stamp - run_win_.front().first) * 1e-9 >= 2.5 &&
                         acc_.s_h - run_win_.front().second >= kWalkingRunM;
    // Not out of a cabin that still moves: its speed has come down to half its top, as it does
    // with a brake too gentle to count (Elevator-VIGS Office3: walked in and the cabin left at
    // once, the boarding walk ended a 7 m ride 3.5 s into its push). Unless it is no cabin: a
    // pulse taken for a push, a jolt or the brake of a ride unseen, never comes back to rest
    // (Station3: the walk on, 80 s, read -40 m). The IMU tells: steps jolt it (kCabinSpread) for
    // 3 s while the odometry walks, which a cabin does not even when the odometry drifts in it
    // (Office2: 10 m in 14 s); or, failing that, a walk longer than any cabin (30 m). Not sooner:
    // a jolt held as a ride for 30 m hid the ride that came 43 s after it (Office3).
    constexpr double kSlowedShare = 0.5;
    constexpr double kNoCabinM = 30.0;
    constexpr Timestamp kGaitS = 3'000'000'000;
    if (az_spread(s.stamp - 1'000'000'000, s.stamp) >= kCabinSpread) {
      if (acc_.gait_since == 0) acc_.gait_since = s.stamp;
    } else {
      acc_.gait_since = 0;
    }
    const bool gait = acc_.gait_since != 0 && s.stamp - acc_.gait_since >= kGaitS;
    const bool slowed = std::abs(acc_.ride_v) < kSlowedShare * acc_.ride_v_peak;
    const double walked = acc_.s_h - acc_.ride_s_h0;
    // Nor out of one still braking: its last pulse is over (Campus6: the odometry moving in the
    // cabin as it braked ended a 23.4 m ride at 17.9).
    constexpr Timestamp kPulseOver = 1'000'000'000;
    const bool stopped_moving = slowed && s.stamp - acc_.last_pulse >= kPulseOver;
    const bool walked_out = acc_.ride_push >= kRideMinSpeed && acc_.last_pulse != 0 && walking &&
                            walked >= kWalkedOutM &&
                            (stopped_moving || gait || walked >= kNoCabinM);
    const bool calm_long_enough =
        acc_.ride_calm_since != 0 && (s.stamp - acc_.ride_calm_since) * 1e-9 >= cfg_.ride_stop_s;
    if (calm_long_enough || walked_out) {
      acc_.imu_end =
          calm_long_enough
              ? acc_.ride_calm_since
              : std::min(s.stamp, std::max(acc_.last_pulse + static_cast<Timestamp>(1.0e9),
                                           run_win_.front().first));
      // Walked out with the speed it gained: one pulse alone, no ride (its height would be the
      // integral of a speed that never came back to rest).
      const bool at_rest = calm_long_enough || stopped_moving;
      if (debug_channel("FLOORS"))
        log_->log(LogLevel::Debug,
                  strprintf("ride ended: %s, %.1fs after the onset",
                            calm_long_enough ? "braked"
                            : at_rest        ? "walked out of the cabin"
                                             : "walked on without coming back to rest: no ride",
                            (acc_.imu_end - acc_.imu_onset) * 1e-9));
      if (at_rest) {
        ev = decide_transition(s, false);
        if (ev.kind != FloorEvent::Kind::None) return ev;
      } else if (acc_.pulse1_end != 0) {
        // The pulse was the brake of a ride whose push stayed under the departure threshold
        // (Elevator-VIGS Station3, Residential3: 0.23 m/s^2 at most): found back in the history,
        // the ride is measured from it to the end of the brake.
        if (const auto start = push_before(acc_.imu_onset, acc_.pulse1_v)) {
          acc_.imu_onset = *start;
          acc_.imu_end = acc_.pulse1_end + static_cast<Timestamp>(1.0e9);
          if (debug_channel("FLOORS"))
            log_->log(LogLevel::Debug,
                      strprintf("ride ended: a brake whose push was found %.1fs before it",
                                (acc_.pulse1_end - acc_.imu_onset) * 1e-9));
          ev = decide_transition(s, false);
          if (ev.kind != FloorEvent::Kind::None) return ev;
        }
      }
      // Not a floor's height: no ride after all. The transition goes back to the odometry, so that
      // a ride that follows is taken from its own onset (Elevator-VIGS Factory-H: a jolt while
      // boarding, then the 50 m ride integrated from it, read -27 m).
      acc_.unobserved = false;
      acc_.az_over_since = 0;
      acc_.imu_end = 0;
      acc_.ride_push = acc_.ride_brake = 0;
      acc_.ride_sign = 0;
      acc_.ride_calm_since = 0;
      acc_.last_ride_stamp = acc_.last_pulse = 0;
      acc_.ride_ref = 0;
      acc_.ride_v = acc_.ride_v_peak = 0;
      acc_.pulse1_end = 0;
      acc_.pulse1_v = 0;
      acc_.gait_since = 0;
    }
  }
  const double settle = cfg_.settle_s * (cfg_.floor_height_prior ? 1.0 : 2.5);
  const bool settled = acc_.calm_since != 0 && (s.stamp - acc_.calm_since) * 1e-9 >= settle;
  const double elapsed = (s.stamp - acc_.t0) * 1e-9;
  if (settled || elapsed > cfg_.max_transition_s) {
    ev = decide_transition(s, elapsed > cfg_.max_transition_s);
    if (ev.kind == FloorEvent::Kind::None)
      acc_.calm_since = 0;  // a landing in the middle of a stair or ramp: keep waiting
  }
  return ev;
}

FloorEvent FloorManager::decide_transition(const FloorSample& s, bool forced) {
  FloorEvent ev;
  const double dz_odom = altitude_ - acc_.alt0;
  const double dz_baro = (acc_.had_baro && s.baro_height) ? (*s.baro_height - acc_.baro0) : 0.0;
  const double dz_imu = (acc_.unobserved && !acc_.had_baro)
                            ? imu_delta_z(acc_.imu_onset, acc_.imu_end ? acc_.imu_end : s.stamp)
                            : 0.0;
  double dz = std::abs(dz_baro) > std::abs(dz_odom) ? dz_baro : dz_odom;
  if (std::abs(dz_imu) > std::abs(dz)) dz = dz_imu;
  if (debug_channel("FLOORS"))
    log_->log(LogLevel::Debug,
              strprintf("decide: dz_odom=%.3f dz_baro=%.3f dz_imu=%.3f unobserved=%d forced=%d "
                        "onset_to_now=%.1fs az_samples=%zu",
                        dz_odom, dz_baro, dz_imu, acc_.unobserved, forced,
                        (s.stamp - acc_.imu_onset) * 1e-9, az_hist_.size()));
  if (std::abs(dz) < cfg_.depart_dz) {
    // Went back to where it started: no floor change. Said as a transition that ended on its own
    // floor, so that the keyframes taken meanwhile join that floor (before 2026-11-04 nothing was
    // said and they were left out).
    state_ = FloorState::OnFloor;
    ref_alt_ = altitude_;
    if (s.baro_height) baro_ref_ = s.baro_height;
    ev.kind = FloorEvent::Kind::TransitionEnded;
    ev.type = TransitionType::Unknown;
    ev.from = ev.to = floor_;
    ev.delta_z = dz;
    ev.continuous_tracking = true;
    return ev;
  }
  if (valid_level_change(dz) || forced) return close_transition(s);
  if (std::abs(dz) >= cfg_.plateau_min_change && acc_.plateau_since != 0 &&
      (s.stamp - acc_.plateau_since) * 1e-9 >= cfg_.plateau_s &&
      acc_.s_h - acc_.plateau_h0 >= cfg_.plateau_travel_m)
    return close_transition(s, true);
  return ev;
}

FloorEvent FloorManager::finish() {
  if (state_ != FloorState::InTransition || !started_) return {};
  return decide_transition(last_sample_, /*forced=*/true);
}

FloorEvent FloorManager::mark_floor_change(TransitionType type, FloorId target) {
  FloorEvent ev;
  const FloorId from = floor_;
  FloorId to = target;
  const double step = cfg_.floor_height_prior ? *cfg_.floor_height_prior : cfg_.min_vertical_change;
  if (to < 0 || static_cast<size_t>(to) >= levels_.size()) {
    levels_.push_back(levels_.at(static_cast<size_t>(from)) + step);
    level_sigma_.push_back(level_sigma_.at(static_cast<size_t>(from)));
    to = static_cast<FloorId>(levels_.size() - 1);
  }
  // The operator says the robot is now on `to`: continue the altitude from that level.
  const double new_alt = levels_.at(static_cast<size_t>(to));
  ev.kind = FloorEvent::Kind::TransitionEnded;
  ev.type = type;
  ev.from = from;
  ev.to = to;
  ev.delta_z = new_alt - levels_.at(static_cast<size_t>(from));
  ev.continuous_tracking = false;
  ev.needs_frontend_reset = true;
  ev.reset_dz = new_alt - altitude_;
  altitude_offset_ += new_alt - altitude_;
  altitude_ = new_alt;
  floor_ = to;
  state_ = FloorState::OnFloor;
  ref_alt_ = altitude_;
  alt_hist_.clear();
  slope_win_.clear();
  climbing_long_ = false;
  climbing_ = false;
  return ev;
}

}  // namespace strataslam
