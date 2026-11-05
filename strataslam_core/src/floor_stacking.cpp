#include "strataslam/fusion/floor_stacking.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace strataslam {

namespace {

using Pts = std::vector<Eigen::Vector2d>;

Pts structure_points(const PointCloud& c, const StackingOptions& o) {
  Pts out;
  for (const auto& p : c.points)
    if (p.z() >= o.z_min && p.z() <= o.z_max) out.emplace_back(p.x(), p.y());
  return out;
}

int count_cells(const Pts& p, double cell) {
  std::vector<std::pair<int, int>> k;
  k.reserve(p.size());
  for (const auto& q : p)
    k.emplace_back(static_cast<int>(std::floor(q.x() / cell)),
                   static_cast<int>(std::floor(q.y() / cell)));
  std::sort(k.begin(), k.end());
  return static_cast<int>(std::unique(k.begin(), k.end()) - k.begin());
}

struct Raster {
  cv::Mat img;             ///< CV_32F, 1 where occupied
  Eigen::Vector2d origin;  ///< metric position of pixel (0, 0)
  int cells = 0;
};

Raster rasterize(const Pts& pts, double cell, bool dilate) {
  Eigen::Vector2d lo(1e18, 1e18), hi(-1e18, -1e18);
  for (const auto& p : pts) {
    lo = lo.cwiseMin(p);
    hi = hi.cwiseMax(p);
  }
  Raster r;
  r.origin = lo - Eigen::Vector2d(cell, cell);
  const int w = static_cast<int>(std::ceil((hi.x() - lo.x()) / cell)) + 3,
            h = static_cast<int>(std::ceil((hi.y() - lo.y()) / cell)) + 3;
  r.img = cv::Mat::zeros(h, w, CV_32F);
  for (const auto& p : pts) {
    const int u = static_cast<int>((p.x() - r.origin.x()) / cell),
              v = static_cast<int>((p.y() - r.origin.y()) / cell);
    if (u >= 0 && u < w && v >= 0 && v < h) r.img.at<float>(v, u) = 1.f;
  }
  r.cells = cv::countNonZero(r.img);
  if (dilate) cv::dilate(r.img, r.img, cv::Mat::ones(3, 3, CV_8U));
  return r;
}

struct Peak {
  double score = -1;
  double yaw = 0;
  Eigen::Vector2d t = Eigen::Vector2d::Zero();
};

/// Scores of every candidate (yaw, translation): the fraction of b's (rotated) cells that fall on
/// a's dilated structure. `a_dilated` is the larger structure, `b` the smaller one.
struct Search {
  std::vector<Peak>
      candidates;  ///< best translation of every yaw, plus every local maximum above `floor_score`
  Peak best;
};

Search search(const Pts& a_pts, const Pts& b_pts, double cell, double yaw0, double yaw_window,
              double yaw_step, const Eigen::Vector2d& t0, double radius, int cells_b_for_score,
              double floor_score) {
  Search s;
  const Raster A = rasterize(a_pts, cell, true);
  const int R = static_cast<int>(std::ceil(radius / cell));
  const int n_yaw = static_cast<int>(std::floor(2.0 * yaw_window / yaw_step + 1e-9));
  for (int iy = 0; iy <= n_yaw; ++iy) {
    const double yaw = yaw0 - yaw_window + iy * yaw_step;
    const double c = std::cos(yaw), sn = std::sin(yaw);
    Pts rb;
    rb.reserve(b_pts.size());
    for (const auto& p : b_pts) rb.emplace_back(c * p.x() - sn * p.y(), sn * p.x() + c * p.y());
    const Raster B = rasterize(rb, cell, false);
    // b placed at translation t has its pixel (0,0) at metric B.origin + t; in A's raster that is
    // ((B.origin + t - A.origin) / cell). Search t in a window of +-R cells around t0.
    const Eigen::Vector2d tl0 = (B.origin + t0 - A.origin) / cell;
    const int x0 = static_cast<int>(std::lround(tl0.x())) - R,
              y0 = static_cast<int>(std::lround(tl0.y())) - R;
    const int cw = 2 * R + B.img.cols, ch = 2 * R + B.img.rows;
    cv::Mat crop = cv::Mat::zeros(ch, cw, CV_32F);
    const cv::Rect want(x0, y0, cw, ch), have(0, 0, A.img.cols, A.img.rows);
    const cv::Rect inter = want & have;
    if (inter.area() <= 0) continue;
    A.img(inter).copyTo(crop(cv::Rect(inter.x - x0, inter.y - y0, inter.width, inter.height)));
    cv::Mat resp;
    cv::matchTemplate(crop, B.img, resp, cv::TM_CCORR);
    resp /= static_cast<float>(std::max(1, cells_b_for_score));
    double mx;
    cv::Point loc;
    cv::minMaxLoc(resp, nullptr, &mx, nullptr, &loc);
    auto to_peak = [&](double score, cv::Point l) {
      Peak p;
      p.score = score;
      p.yaw = yaw;
      p.t = A.origin + Eigen::Vector2d((x0 + l.x) * cell, (y0 + l.y) * cell) - B.origin;
      return p;
    };
    if (mx > s.best.score) s.best = to_peak(mx, loc);
    s.candidates.push_back(to_peak(mx, loc));
    // Other maxima of this yaw: cells that beat all their neighbours within the exclusion radius.
    cv::Mat dil;
    const int k = 2 * static_cast<int>(std::ceil(3.0 / cell)) + 1;
    cv::dilate(resp, dil, cv::Mat::ones(k, k, CV_8U));
    for (int v = 0; v < resp.rows; ++v)
      for (int u = 0; u < resp.cols; ++u)
        if (resp.at<float>(v, u) >= dil.at<float>(v, u) && resp.at<float>(v, u) > floor_score)
          s.candidates.push_back(to_peak(resp.at<float>(v, u), cv::Point(u, v)));
  }
  return s;
}

}  // namespace

std::optional<StackingMatch> match_floor_structure(const PointCloud& a_cloud,
                                                   const PointCloud& b_cloud, const Pose4& initial,
                                                   const StackingOptions& o, std::string* why) {
  const auto fail = [&](const std::string& m) -> std::optional<StackingMatch> {
    if (why) *why = m;
    return std::nullopt;
  };
  Pts a = structure_points(a_cloud, o), b = structure_points(b_cloud, o);
  const int ca = count_cells(a, o.cell), cb = count_cells(b, o.cell);
  if (ca < o.min_cells || cb < o.min_cells)
    return fail("too little structure (" + std::to_string(ca) + " and " + std::to_string(cb) +
                " cells)");
  // The smaller structure is the template: its fraction on the larger one is the score.
  if (cb > ca) {
    std::string w;
    auto r = match_floor_structure(b_cloud, a_cloud, initial.inverse(), o, &w);
    if (!r) return fail(w);
    StackingMatch m = *r;
    m.a_to_b = r->a_to_b.inverse();
    std::swap(m.cells_a, m.cells_b);
    // The inverse keeps (yaw, x, y) uncertainty up to a rotation: the covariance is isotropic in
    // xy.
    return m;
  }
  const double yaw0 = initial.theta();
  const Eigen::Vector2d t0(initial.translation().x(), initial.translation().y());
  const double window = o.yaw_window_deg * M_PI / 180.0, step = o.yaw_step_deg * M_PI / 180.0;
  Search coarse = search(a, b, o.cell, yaw0, window, step, t0, o.search_radius, cb, 0.0);
  if (coarse.best.score < o.min_score)
    return fail("the structures do not overlap enough (best score " +
                std::to_string(coarse.best.score) + ")");
  // Best score at a distinct pose (another place or another heading).
  double runner = 0;
  for (const auto& c : coarse.candidates) {
    const double dyaw = std::abs(Pose4::WrapAngle(c.yaw - coarse.best.yaw)) * 180.0 / M_PI;
    if ((c.t - coarse.best.t).norm() > o.exclusion_m || dyaw > o.exclusion_deg)
      runner = std::max(runner, c.score);
  }
  const bool ambiguous = runner > 0 && coarse.best.score < o.peak_ratio * runner;
  if (ambiguous && !o.allow_ambiguous)
    return fail("ambiguous: best score " + std::to_string(coarse.best.score) + " against " +
                std::to_string(runner) + " at another pose");
  // Refinement on a finer raster around the coarse optimum.
  const double fine = o.cell * 0.5;
  const int cb_fine = count_cells(b, fine);
  Search refined = search(a, b, fine, coarse.best.yaw, 1.5 * step, 0.25 * step, coarse.best.t,
                          1.5 * o.cell + 0.5, cb_fine, 0.0);
  const Peak& p = refined.best.score >= 0 ? refined.best : coarse.best;
  StackingMatch m;
  m.a_to_b = Pose4(p.yaw, Eigen::Vector3d(p.t.x(), p.t.y(), initial.translation().z()));
  m.score = coarse.best.score;
  m.runner_up = runner;
  m.ambiguous = ambiguous;
  m.cells_a = ca;
  m.cells_b = cb;
  const double sxy = std::max(0.25, o.cell), syaw = 0.75 * M_PI / 180.0;
  m.covariance = Eigen::Vector4d(syaw * syaw, sxy * sxy, sxy * sxy, 1e4).asDiagonal();
  return m;
}

}  // namespace strataslam
