#include "strataslam/sensors/imu_preintegration.hpp"

namespace strataslam {

ImuPreintegrator::ImuPreintegrator(const ImuNoise& n, double gravity,
                                   const gtsam::imuBias::ConstantBias& bias)
    : bias_(bias) {
  params_ = gtsam::PreintegrationCombinedParams::MakeSharedU(gravity);
  // Continuous-time spectral densities -> covariances (the factor scales them by dt itself).
  params_->accelerometerCovariance =
      gtsam::Matrix33::Identity() * n.accel_noise_density * n.accel_noise_density;
  params_->gyroscopeCovariance =
      gtsam::Matrix33::Identity() * n.gyro_noise_density * n.gyro_noise_density;
  params_->biasAccCovariance =
      gtsam::Matrix33::Identity() * n.accel_random_walk * n.accel_random_walk;
  params_->biasOmegaCovariance =
      gtsam::Matrix33::Identity() * n.gyro_random_walk * n.gyro_random_walk;
  params_->integrationCovariance = gtsam::Matrix33::Identity() * 1e-8;
  params_->biasAccOmegaInt = gtsam::Matrix66::Identity() * 1e-5;
  pim_ = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(params_, bias_);
}

void ImuPreintegrator::add(const ImuSample& s, double dt) {
  if (dt <= 0) return;
  pim_->integrateMeasurement(s.accel, s.gyro, dt);
}

void ImuPreintegrator::add_batch(const std::vector<ImuSample>& samples, Timestamp prev_stamp) {
  Timestamp t = prev_stamp;
  for (const auto& s : samples) {
    add(s, (s.stamp - t) * 1e-9);
    t = s.stamp;
  }
}

void ImuPreintegrator::reset(const gtsam::imuBias::ConstantBias& bias) {
  bias_ = bias;
  pim_->resetIntegrationAndSetBias(bias);
}

}  // namespace strataslam
