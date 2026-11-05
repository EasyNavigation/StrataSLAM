#pragma once

#include <memory>

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>

#include "strataslam/config/config.hpp"
#include "strataslam/sensors/measurements.hpp"

namespace strataslam {

/// On-manifold IMU preintegration (gtsam::PreintegratedCombinedMeasurements) between two instants,
/// with the noise densities of the profile. World frame is z-up; the accelerometer reads +g at rest.
class ImuPreintegrator {
 public:
  explicit ImuPreintegrator(const ImuNoise& noise, double gravity = 9.80665,
                            const gtsam::imuBias::ConstantBias& bias = gtsam::imuBias::ConstantBias());

  /// Integrates one sample over `dt` seconds.
  void add(const ImuSample& s, double dt);
  /// Integrates a time-ordered batch; `prev_stamp` is the instant the batch starts at.
  void add_batch(const std::vector<ImuSample>& samples, Timestamp prev_stamp);
  void reset(const gtsam::imuBias::ConstantBias& bias);

  double delta_t() const { return pim_->deltaTij(); }
  gtsam::Rot3 delta_rotation() const { return pim_->deltaRij(); }
  /// Predicted navigation state at the end of the window.
  gtsam::NavState predict(const gtsam::NavState& at_start) const { return pim_->predict(at_start, bias_); }
  const gtsam::PreintegratedCombinedMeasurements& measurements() const { return *pim_; }

 private:
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> params_;
  std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> pim_;
  gtsam::imuBias::ConstantBias bias_;
};

}  // namespace strataslam
