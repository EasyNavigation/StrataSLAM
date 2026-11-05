#include "strataslam/frontend/feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <opencv2/features2d.hpp>

namespace strataslam {

namespace {
/// ORB, the strongest corners. A mask restricts detection, e.g. to the pixels that have a depth:
/// the strongest corners gather where the contrast is, which in a corridor is its far end, out of
/// the range of an RGB-D sensor (Tiago, pasillo_Tiago: 24-99 of 1000 keypoints had a depth without
/// it).
class OrbExtractor final : public FeatureExtractor {
 public:
  explicit OrbExtractor(int n) : orb_(cv::ORB::create(n)) {}
  std::vector<RgbKeypoint> extract(const GrayImage& image,
                                   const std::vector<uint8_t>* mask) override {
    const cv::Mat img(image.height, image.width, CV_8UC1, const_cast<uint8_t*>(image.data.data()));
    cv::Mat m;
    if (mask && mask->size() == image.data.size())
      m = cv::Mat(image.height, image.width, CV_8UC1, const_cast<uint8_t*>(mask->data()));
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    orb_->detectAndCompute(img, m, kps, desc);
    std::vector<RgbKeypoint> out;
    out.reserve(kps.size());
    for (size_t i = 0; i < kps.size(); ++i) {
      RgbKeypoint k;
      k.u = kps[i].pt.x;
      k.v = kps[i].pt.y;
      k.response = kps[i].response;
      std::memcpy(k.descriptor.data(), desc.ptr(static_cast<int>(i)), kDescriptorBytes);
      out.push_back(k);
    }
    std::sort(out.begin(), out.end(),
              [](const RgbKeypoint& a, const RgbKeypoint& b) { return a.response > b.response; });
    return out;
  }
  std::string name() const override { return "orb"; }

 private:
  cv::Ptr<cv::ORB> orb_;
};
}  // namespace

std::unique_ptr<FeatureExtractor> make_orb_extractor(int max_features) {
  return std::make_unique<OrbExtractor>(max_features);
}

std::vector<uint8_t> valid_depth_mask(const DepthImage& depth, double depth_min, double depth_max) {
  std::vector<uint8_t> m(depth.meters.size());
  for (size_t i = 0; i < m.size(); ++i) {
    const float d = depth.meters[i];
    m[i] = (std::isfinite(d) && d >= depth_min && d <= depth_max) ? 255 : 0;
  }
  return m;
}

}  // namespace strataslam
