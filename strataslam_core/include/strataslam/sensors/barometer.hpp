#pragma once

#include <optional>

#include "strataslam/common/types.hpp"

namespace strataslam {

constexpr double kDryAirGasConstant = 287.05;  ///< J/(kg K)
constexpr double kStandardGravity = 9.80665;   ///< m/s^2

/// Hypsometric height of the sensor above the reference level: dh = Rd T / g * ln(p_ref / p).
double hypsometric_height(double p_ref_pa, double p_pa, double temperature_k);

/// First-order low-pass over the barometric height, reference taken at start-up or from a base
/// station (differential altitude, which cancels the weather drift).
class BaroFilter {
 public:
  /// `settle_time_s`: time for a step input to settle within 1 %.
  BaroFilter(double temperature_k, double settle_time_s);

  /// Feeds a pressure sample; returns the filtered height relative to the reference (m).
  double update(Timestamp stamp, double pressure_pa);
  /// Reference pressure from a fixed base station sample, applied to the next update.
  void set_base_station(double base_pressure_pa);
  std::optional<double> height() const { return height_; }
  void reset();

 private:
  double temperature_k_, tau_s_;
  std::optional<double> p_ref_;
  std::optional<double> base_station_pa_;  ///< latest base-station pressure, if any
  std::optional<double>
      base_station_ref_;  ///< base-station pressure when the rover reference was set
  std::optional<double> height_;
  Timestamp last_stamp_ = 0;
};

}  // namespace strataslam
