#include "strataslam_sim/trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

#include "strataslam_sim/rng.hpp"

namespace strataslam_sim {

using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;

namespace {

/// Planar path of straight lines and circular fillets, parametrized by arc length.
class PlanarPath {
 public:
  PlanarPath(const std::vector<Vector2d>& pts, double fillet) {
    std::vector<Vector2d> p;
    for (const auto& q : pts) {
      if (p.empty() || (q - p.back()).norm() > 1e-9) p.push_back(q);
    }
    Vector2d cur = p.front();
    double s = 0;
    for (size_t i = 1; i < p.size(); ++i) {
      const Vector2d d_in = (p[i] - p[i - 1]).normalized();
      double trim_end = 0;
      Vector2d d_out = d_in;
      double alpha = 0, radius = 0;
      if (i + 1 < p.size()) {
        d_out = (p[i + 1] - p[i]).normalized();
        alpha = std::atan2(d_in.x() * d_out.y() - d_in.y() * d_out.x(), d_in.dot(d_out));
        if (std::abs(alpha) > 1e-6) {
          double l = fillet * std::tan(std::abs(alpha) / 2);
          const double lim = 0.5 * std::min((p[i] - cur).norm() + 1e-12, (p[i + 1] - p[i]).norm());
          l = std::min(l, lim);
          radius = l / std::tan(std::abs(alpha) / 2);
          trim_end = l;
        }
      }
      const Vector2d line_end = p[i] - d_in * trim_end;
      const double len = (line_end - cur).norm();
      if (len > 1e-9) {
        segs_.push_back({s, len, cur, d_in, 0.0});
        s += len;
      }
      if (trim_end > 0) {
        const double arc = radius * std::abs(alpha);
        segs_.push_back({s, arc, line_end, d_in, (alpha > 0 ? 1.0 : -1.0) / radius});
        s += arc;
        cur = p[i] + d_out * trim_end;
      } else {
        cur = line_end;
      }
    }
    length_ = s;
  }

  double length() const { return length_; }

  void eval(double s, Vector2d* pos, double* yaw) const {
    s = std::clamp(s, 0.0, length_);
    const Seg* seg = &segs_.back();
    for (const auto& g : segs_) {
      if (s <= g.s0 + g.len) {
        seg = &g;
        break;
      }
    }
    const double u = std::min(s - seg->s0, seg->len);
    const double h0 = std::atan2(seg->dir.y(), seg->dir.x());
    if (seg->curv == 0.0) {
      *pos = seg->p0 + seg->dir * u;
      *yaw = h0;
    } else {
      const Vector2d nl(-seg->dir.y(), seg->dir.x());
      const double k = seg->curv;
      *pos = seg->p0 + (std::sin(k * u) * seg->dir + (1 - std::cos(k * u)) * nl) / k;
      *yaw = h0 + k * u;
    }
  }

 private:
  struct Seg {
    double s0, len;
    Vector2d p0, dir;
    double curv;
  };
  std::vector<Seg> segs_;
  double length_ = 0;
};

/// Distance travelled after time `t` for a trapezoidal speed profile starting/ending at rest.
double trapezoid(double t, double total_len, double vmax, double a, double* duration) {
  const double d_acc = vmax * vmax / (2 * a);
  double t_acc, t_cruise;
  if (total_len >= 2 * d_acc) {
    t_acc = vmax / a;
    t_cruise = (total_len - 2 * d_acc) / vmax;
  } else {
    t_acc = std::sqrt(total_len / a);
    t_cruise = 0;
    vmax = a * t_acc;
  }
  *duration = 2 * t_acc + t_cruise;
  if (t <= 0) return 0;
  if (t < t_acc) return 0.5 * a * t * t;
  if (t < t_acc + t_cruise) return 0.5 * a * t_acc * t_acc + vmax * (t - t_acc);
  const double td = *duration - t;
  if (td <= 0) return total_len;
  return total_len - 0.5 * a * td * td;
}

double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

}  // namespace

// ---------------- Trajectory ----------------

gtsam::Pose3 Trajectory::pose_at(double t) const {
  t = std::clamp(t, 0.0, duration());
  const double f = t / dt_;
  const size_t i = std::min(static_cast<size_t>(f), samples_.size() - 2);
  const double a = std::clamp(f - i, 0.0, 1.0);
  const auto& s0 = samples_[i];
  const auto& s1 = samples_[i + 1];
  const Eigen::Quaterniond q = Eigen::Quaterniond(s0.R).slerp(a, Eigen::Quaterniond(s1.R));
  return gtsam::Pose3(gtsam::Rot3(q.normalized().toRotationMatrix()),
                      Vector3d((1 - a) * s0.p + a * s1.p));
}

TrajectorySample Trajectory::state_at(double t) const {
  t = std::clamp(t, 0.0, duration());
  const double f = t / dt_;
  const size_t i = std::min(static_cast<size_t>(f), samples_.size() - 2);
  const double a = std::clamp(f - i, 0.0, 1.0);
  TrajectorySample s = samples_[a < 0.5 ? i : i + 1];
  s.cabin_z = (1 - a) * samples_[i].cabin_z + a * samples_[i + 1].cabin_z;
  s.t = t;
  return s;
}

std::string Trajectory::to_tum(double t0_s) const {
  std::ostringstream os;
  os << std::setprecision(12);
  for (const auto& s : samples_) {
    const Eigen::Quaterniond q(s.R);
    char stamp[48];
    std::snprintf(stamp, sizeof stamp, "%.9f", s.t + t0_s);
    os << stamp << ' ' << s.p.x() << ' ' << s.p.y() << ' ' << s.p.z() << ' ' << q.x() << ' '
       << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
  }
  return os.str();
}

// ---------------- Builder ----------------

TrajectoryBuilder::TrajectoryBuilder(const Building& b, const TrajectoryConfig& cfg,
                                     int start_level, const Vector2d& start_xy, double start_yaw)
    : building_(b),
      cfg_(cfg),
      level_(start_level),
      x_(start_xy.x()),
      y_(start_xy.y()),
      yaw_(start_yaw) {
  surface_z_ = b.level_height(level_);
  cabin_level_ = 0;
  cabin_z_ = 0;
}

TrajectoryBuilder& TrajectoryBuilder::set_cabin_level(int level) {
  cabin_level_ = level;
  cabin_z_ = building_.level_height(level);
  return *this;
}

Matrix3d TrajectoryBuilder::attitude(double yaw, double t, double wobble_scale) const {
  const double w = cfg_.wobble_deg * M_PI / 180.0 * wobble_scale;
  const double roll =
      w * std::sin(2 * M_PI * 0.8 * t + 0.3) + 0.3 * w * std::sin(2 * M_PI * 1.9 * t);
  const double pitch =
      w * std::sin(2 * M_PI * 1.1 * t + 1.1) + 0.3 * w * std::sin(2 * M_PI * 2.3 * t + 0.5);
  return (Eigen::AngleAxisd(yaw, Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll, Vector3d::UnitX()))
      .toRotationMatrix();
}

void TrajectoryBuilder::emit(const Vector3d& p, double yaw, int label, double cabin_z,
                             bool door_closed) {
  TrajectorySample s;
  s.t = t_;
  s.p = p;
  s.yaw = yaw;
  s.R = attitude(yaw, t_, aerial_ ? 0.5 : 1.0);
  s.true_floor = label;
  s.cabin_z = cabin_z;
  s.door_closed = door_closed;
  samples_.push_back(s);
  t_ += cfg_.dt;
}

TrajectoryBuilder& TrajectoryBuilder::hold(double seconds) {
  const int n = static_cast<int>(std::round(seconds / cfg_.dt));
  const double h = aerial_ ? agl_ : cfg_.sensor_height;
  const int label = std::abs(surface_z_ - building_.level_height(level_)) < 0.05 ? level_ : -1;
  for (int i = 0; i < n; ++i) emit({x_, y_, surface_z_ + h}, yaw_, label, cabin_z_, false);
  return *this;
}

TrajectoryBuilder& TrajectoryBuilder::turn_in_place(double target) {
  const double delta = wrap(target - yaw_);
  if (std::abs(delta) < 1e-6) return *this;
  double dur;
  const double a_turn = 1.0;  // rad/s^2
  // Trapezoid on angle with peak rate cfg_.turn_rate.
  trapezoid(0, std::abs(delta), cfg_.turn_rate, a_turn, &dur);
  const int n = static_cast<int>(std::ceil(dur / cfg_.dt));
  const double h = aerial_ ? agl_ : cfg_.sensor_height;
  const int label = std::abs(surface_z_ - building_.level_height(level_)) < 0.05 ? level_ : -1;
  const double y0 = yaw_;
  for (int i = 0; i <= n; ++i) {
    double d2;
    const double ang = trapezoid(i * cfg_.dt, std::abs(delta), cfg_.turn_rate, a_turn, &d2);
    emit({x_, y_, surface_z_ + h}, y0 + std::copysign(ang, delta), label, cabin_z_, false);
  }
  yaw_ = y0 + delta;
  return *this;
}

void TrajectoryBuilder::walk_path(const std::vector<Vector2d>& points,
                                  const std::function<double(double, double)>& surface,
                                  const std::function<int(double, double, double)>& label) {
  std::vector<Vector2d> pts{{x_, y_}};
  pts.insert(pts.end(), points.begin(), points.end());
  const Vector2d d0 = pts[1] - pts[0];
  if (d0.norm() > 1e-6) turn_in_place(std::atan2(d0.y(), d0.x()));
  PlanarPath path(pts, cfg_.fillet);
  double dur;
  trapezoid(0, path.length(), cfg_.walk_speed, cfg_.walk_accel, &dur);
  const int n = static_cast<int>(std::ceil(dur / cfg_.dt));
  const double h = aerial_ ? agl_ : cfg_.sensor_height;
  // Walking surface is smoothed over 0.25 s to mimic a body moving over steps.
  const int win = std::max(1, static_cast<int>(0.25 / cfg_.dt));
  std::vector<double> raw(n + 1), lab_z(n + 1);
  std::vector<Vector2d> xy(n + 1);
  std::vector<double> yaws(n + 1);
  for (int i = 0; i <= n; ++i) {
    double d2;
    const double s = trapezoid(i * cfg_.dt, path.length(), cfg_.walk_speed, cfg_.walk_accel, &d2);
    path.eval(s, &xy[i], &yaws[i]);
    raw[i] = surface(xy[i].x(), xy[i].y());
  }
  const double base_yaw = yaw_;
  double prev = yaws[0];
  double offset = 2 * M_PI * std::round((base_yaw - prev) / (2 * M_PI));  // unwrap vs current yaw
  for (int i = 0; i <= n; ++i) {
    double sum = 0;
    int cnt = 0;
    for (int j = std::max(0, i - win / 2); j <= std::min(n, i + win / 2); ++j, ++cnt) sum += raw[j];
    const double z = sum / cnt;
    double y = yaws[i] + offset;
    y = prev + wrap(y - prev);
    prev = y;
    yaws[i] = y;
    emit({xy[i].x(), xy[i].y(), z + h}, y, label(xy[i].x(), xy[i].y(), raw[i]), cabin_z_, false);
    surface_z_ = z;
  }
  x_ = xy[n].x();
  y_ = xy[n].y();
  yaw_ = yaws[n];
  surface_z_ = raw[n];
}

TrajectoryBuilder& TrajectoryBuilder::walk(const std::vector<Vector2d>& points) {
  const double z = surface_z_;
  const int level = level_;
  const double lz = building_.level_height(level_);
  walk_path(
      points, [z](double, double) { return z; },
      [level, lz, z](double, double, double s) {
        return std::abs(s - lz) < 0.05 && std::abs(z - lz) < 0.05 ? level : -1;
      });
  return *this;
}

TrajectoryBuilder& TrajectoryBuilder::walk_facing(const std::vector<Vector2d>& points, double yaw) {
  turn_in_place(yaw);
  std::vector<Vector2d> pts{{x_, y_}};
  pts.insert(pts.end(), points.begin(), points.end());
  PlanarPath path(pts, cfg_.fillet);
  double dur;
  trapezoid(0, path.length(), cfg_.walk_speed, cfg_.walk_accel, &dur);
  const int n = static_cast<int>(std::ceil(dur / cfg_.dt));
  const double z = surface_z_, h = cfg_.sensor_height;
  const int level = level_;
  const int label = std::abs(z - building_.level_height(level)) < 0.05 ? level : -1;
  for (int i = 0; i <= n; ++i) {
    double d2, hd;
    Vector2d p;
    const double s = trapezoid(i * cfg_.dt, path.length(), cfg_.walk_speed, cfg_.walk_accel, &d2);
    path.eval(s, &p, &hd);
    emit({p.x(), p.y(), z + h}, yaw_, label, cabin_z_, false);
    x_ = p.x();
    y_ = p.y();
  }
  return *this;
}

TrajectoryBuilder& TrajectoryBuilder::stairs(int level, bool up) {
  const Building& b = building_;
  const double xa = 0.5 * (10.2 + 11.8), xb = 0.5 * (12.4 + 14.0);
  std::vector<Vector2d> pts;
  if (up)
    pts = {{xa, 0.8}, {xa, -3.4}, {xb, -3.4}, {xb, 0.8}};
  else
    pts = {{xb, 0.8}, {xb, -3.4}, {xa, -3.4}, {xa, 0.8}};
  // The robot must be at the start point, facing the annex, on level (up) or level+1 (down).
  const double H = b.config().floor_height;
  const auto surface = [&b, level, H](double x, double y) {
    if (y >= 0.0) return b.level_height(level) + (x > 12.2 ? H : 0.0);
    return b.stairs_surface(level, x, y).value_or(b.level_height(level));
  };
  const auto label = [&b, level](double, double, double s) {
    for (int l : {level, level + 1}) {
      if (std::abs(s - b.level_height(l)) < 0.05) return l;
    }
    return -1;
  };
  // Force transition labelling inside the annex (y < 0) even when standing on a landing level.
  walk_path(pts, surface,
            [&](double x, double y, double s) { return y >= 0 ? label(x, y, s) : -1; });
  level_ = up ? level + 1 : level;
  return *this;
}

TrajectoryBuilder& TrajectoryBuilder::ramp(int level, bool up) {
  const Building& b = building_;
  const double ya = 9.1, yb = 10.9, xe = b.ramp_x1() + 0.8;
  std::vector<Vector2d> pts;
  if (up)
    pts = {{33.0, ya}, {xe, ya}, {xe, yb}, {33.0, yb}, {31.0, yb}};
  else
    pts = {{33.0, yb}, {xe, yb}, {xe, ya}, {33.0, ya}, {31.0, ya}};
  const double H = b.config().floor_height;
  const auto surface = [&b, level, H](double x, double y) {
    if (x < 32.4 || y < 8.4 || y > 11.6) return b.level_height(level) + (y > 10.0 ? H : 0.0);
    return b.ramp_surface(level, x, y).value_or(b.level_height(level));
  };
  const auto label = [&b, level](double x, double, double s) {
    if (x >= 32.0) return -1;
    for (int l : {level, level + 1}) {
      if (std::abs(s - b.level_height(l)) < 0.05) return l;
    }
    return -1;
  };
  walk_path(pts, surface, label);
  level_ = up ? level + 1 : level;
  return *this;
}

TrajectoryBuilder& TrajectoryBuilder::elevator(int from_level, int to_level) {
  const double zf = building_.level_height(from_level), zt = building_.level_height(to_level);
  const double dist = std::abs(zt - zf);
  double dur;
  trapezoid(0, dist, cfg_.elevator_speed, cfg_.elevator_accel, &dur);
  const double pre = 1.5, post = 4.5;  // door closing / opening; the robot stays put while it opens
  const double h = cfg_.sensor_height;
  const int n_pre = static_cast<int>(pre / cfg_.dt), n_post = static_cast<int>(post / cfg_.dt);
  const int n = static_cast<int>(std::ceil(dur / cfg_.dt));
  // The label is "in transit" from the moment the door starts closing until it has opened again.
  for (int i = 0; i < n_pre; ++i)
    emit({x_, y_, zf + h}, yaw_, i > static_cast<int>(0.5 / cfg_.dt) ? -1 : from_level, zf,
         i > static_cast<int>(0.5 / cfg_.dt));
  for (int i = 0; i <= n; ++i) {
    double d2;
    const double s = trapezoid(i * cfg_.dt, dist, cfg_.elevator_speed, cfg_.elevator_accel, &d2);
    const double z = zf + std::copysign(s, zt - zf);
    emit({x_, y_, z + h}, yaw_, -1, z, true);
  }
  // The door opens 1 s after the cabin stops; the robot keeps standing for the rest of `post`.
  for (int i = 0; i < n_post; ++i)
    emit({x_, y_, zt + h}, yaw_, i < static_cast<int>(1.0 / cfg_.dt) ? -1 : to_level, zt,
         i < static_cast<int>(1.0 / cfg_.dt));
  surface_z_ = zt;
  cabin_z_ = zt;
  cabin_level_ = to_level;
  level_ = to_level;
  return *this;
}

TrajectoryBuilder& TrajectoryBuilder::fly_to_level(int target_level, double agl, double yaw_rate) {
  if (!aerial_) {
    aerial_ = true;
    agl_ = agl;
  }
  const double z0 = surface_z_, z1 = building_.level_height(target_level);
  const double dist = std::abs(z1 - z0);
  double dur;
  trapezoid(0, dist, 0.5, 0.5, &dur);
  const int n = static_cast<int>(std::ceil(dur / cfg_.dt));
  const double y0 = yaw_;
  for (int i = 0; i <= n; ++i) {
    double d2;
    const double s = trapezoid(i * cfg_.dt, dist, 0.5, 0.5, &d2);
    const double z = z0 + std::copysign(s, z1 - z0);
    const int label = std::abs(z - z1) < 0.05 || std::abs(z - z0) < 0.05
                          ? (std::abs(z - z1) < 0.05 ? target_level : level_)
                          : -1;
    emit({x_, y_, z + agl_}, y0 + yaw_rate * i * cfg_.dt, label, cabin_z_, false);
    yaw_ = y0 + yaw_rate * i * cfg_.dt;
  }
  surface_z_ = z1;
  level_ = target_level;
  return *this;
}

Trajectory TrajectoryBuilder::build() const {
  std::vector<TrajectorySample> s = samples_;
  const size_t n = s.size();
  const double dt = cfg_.dt;
  for (size_t i = 0; i < n; ++i) {
    const size_t a = i == 0 ? 0 : i - 1, c = i + 1 >= n ? n - 1 : i + 1;
    const double span = (c - a) * dt;
    s[i].v = (samples_[c].p - samples_[a].p) / span;
    if (i > 0 && i + 1 < n)
      s[i].a = (samples_[i + 1].p - 2 * samples_[i].p + samples_[i - 1].p) / (dt * dt);
    else
      s[i].a.setZero();
    const Matrix3d dR = samples_[a].R.transpose() * samples_[c].R;
    const Eigen::AngleAxisd aa(dR);
    s[i].omega_body = aa.axis() * (aa.angle() / span);
    if (!std::isfinite(s[i].omega_body.x())) s[i].omega_body.setZero();
  }
  return Trajectory(std::move(s), cfg_.dt);
}

}  // namespace strataslam_sim
