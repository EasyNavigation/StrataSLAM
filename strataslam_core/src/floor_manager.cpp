#include "strataslam/floors/floor_manager.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strataslam {

namespace {
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

/// Double integration of the vertical acceleration between `onset` and `end`, with a zero-velocity
/// update at the end (the cabin is at rest again): the residual velocity is spread linearly, which
/// removes most of the effect of an accelerometer bias.
double FloorManager::imu_delta_z(Timestamp onset, Timestamp end) const {
  std::vector<std::pair<double, double>> a;  // (t seconds, az)
  for (const auto& h : az_hist_) {
    if (h.first >= onset && h.first <= end) a.push_back({h.first * 1e-9, h.second});
  }
  if (a.size() < 3) return 0.0;
  {  // 5-point median: the impulses of the sensor would otherwise be integrated twice
    std::vector<double> f(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      std::vector<double> w;
      for (size_t j = (i >= 2 ? i - 2 : 0); j < std::min(a.size(), i + 3); ++j)
        w.push_back(a[j].second);
      std::sort(w.begin(), w.end());
      f[i] = w[w.size() / 2];
    }
    for (size_t i = 0; i < a.size(); ++i) a[i].second = f[i];
  }
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

FloorId FloorManager::assign_level(double altitude, double tol_override) {
  const double tol = tol_override > 0 ? tol_override : tolerance();
  FloorId best = -1;
  double best_d = 1e9;
  for (size_t i = 0; i < levels_.size(); ++i) {
    const double d = std::abs(levels_[i] - altitude);
    if (d < tol && d < best_d) {
      best_d = d;
      best = static_cast<FloorId>(i);
    }
  }
  if (best >= 0) return best;
  levels_.push_back(altitude);
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
  const double dz_imu = acc_.unobserved && !have_baro ? imu_delta_z(acc_.imu_onset, s.stamp) : 0.0;
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
  const FloorId to = (elevator || valid_level_change(dz) || plateau)
                         ? assign_level(target, plateau ? 0.5 * cfg_.plateau_min_change : -1.0)
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
    while (az_hist_.size() > 2 && (s.stamp - az_hist_.front().first) * 1e-9 > 12.0)
      az_hist_.pop_front();
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
  // lost its third floor that way).
  bool imu_departure = false;
  Timestamp imu_onset = 0;
  if (s.imu_vertical_accel && std::abs(m.vz) < 0.1 &&
      std::abs(altitude_ - ref_alt_) < 0.5 * cfg_.depart_dz) {
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
  bool calm = std::abs(m.vz) < cfg_.calm_vz && std::abs(vb) < 0.1 &&
              (!s.imu_vertical_accel || std::abs(az_mean(s.stamp, 1.0)) < cfg_.imu_calm_accel);
  if (climbing_ || climbing_long_) calm = false;
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
  const double dz_imu =
      (acc_.unobserved && !acc_.had_baro) ? imu_delta_z(acc_.imu_onset, s.stamp) : 0.0;
  double dz = std::abs(dz_baro) > std::abs(dz_odom) ? dz_baro : dz_odom;
  if (std::abs(dz_imu) > std::abs(dz)) dz = dz_imu;
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
