#include "strataslam/sensors/barometer.hpp"

#include <cmath>

namespace strataslam {

// The hypsometric equation of an isothermal layer of dry air: dh = (Rd T / g) ln(p1 / p2), e.g. J.
// M. Wallace and P. V. Hobbs, "Atmospheric Science: An Introductory Survey", 2nd ed., Academic
// Press 2006, eq. 3.29.
double hypsometric_height(double p_ref_pa, double p_pa, double temperature_k) {
  return kDryAirGasConstant * temperature_k / kStandardGravity * std::log(p_ref_pa / p_pa);
}

BaroFilter::BaroFilter(double temperature_k, double settle_time_s)
    : temperature_k_(temperature_k), tau_s_(settle_time_s / std::log(100.0)) {}

void BaroFilter::reset() {
  p_ref_.reset();
  height_.reset();
  base_station_ref_.reset();
  last_stamp_ = 0;
}

void BaroFilter::set_base_station(double base_pressure_pa) { base_station_pa_ = base_pressure_pa; }

double BaroFilter::update(Timestamp stamp, double pressure_pa) {
  if (!p_ref_) {
    p_ref_ = pressure_pa;
    base_station_ref_ = base_station_pa_;
  } else if (base_station_pa_ && !base_station_ref_) {
    // The base station's first sample came after the rover's: its reference is taken now, so the
    // weather is cancelled from here on (before 2026-11-04 it was never applied; audit A4A5-02).
    base_station_ref_ = base_station_pa_;
  }
  // Differential altitude: weather-induced pressure changes seen by the base station are removed.
  double p = pressure_pa;
  if (base_station_pa_ && base_station_ref_) p *= *base_station_ref_ / *base_station_pa_;
  const double h = hypsometric_height(*p_ref_, p, temperature_k_);
  if (!height_) {
    height_ = h;
  } else {
    const double dt = (stamp - last_stamp_) * 1e-9;
    const double a = dt > 0 ? 1.0 - std::exp(-dt / tau_s_) : 1.0;
    height_ = *height_ + a * (h - *height_);
  }
  last_stamp_ = stamp;
  return *height_;
}

}  // namespace strataslam
