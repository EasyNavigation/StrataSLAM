// Offline floor fusion of a saved map: reads <map_dir>, writes <map_dir>/fused/.
// Usage: fuse_tool <map_dir> [--formulation lie4|se3soft|euler] [--anchors anchors.yaml]
// [--floor-height-prior m]
#include <cstdio>
#include <sstream>
#include <string>

#include "strataslam/fusion/fusion.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <map_dir> [--formulation lie4|se3soft|euler] [--anchors file.yaml] "
                 "[--floor-height-prior m] [--stacking on|off] [--same-level A,B;C,D]\n",
                 argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  strataslam::FusionOptions opt;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--formulation") {
      if (!strataslam::parse_formulation(v, &opt.formulation)) {
        std::fprintf(stderr, "unknown formulation '%s'\n", v.c_str());
        return 2;
      }
    } else if (k == "--anchors") {
      const auto a = strataslam::load_anchors(v);
      if (!a) {
        std::fprintf(stderr, "%s\n", a.error().message.c_str());
        return 1;
      }
      opt.anchors = a.value();
    } else if (k == "--same-level") {  // "2,3;0,4"
      std::stringstream ss(v);
      std::string pair;
      while (std::getline(ss, pair, ';')) {
        strataslam::SameLevel s;
        if (std::sscanf(pair.c_str(), "%d,%d", &s.a, &s.b) != 2) {
          std::fprintf(stderr, "bad --same-level pair '%s' (expected A,B)\n", pair.c_str());
          return 2;
        }
        opt.same_level.push_back(s);
      }
    } else if (k == "--stacking") {
      opt.auto_stacking = (v != "off");
    } else if (k == "--floor-height-prior") {
      opt.floor_height_prior = std::stod(v);
    }
  }
  const auto map = strataslam::read_map(dir);
  if (!map) {
    std::fprintf(stderr, "%s\n", map.error().message.c_str());
    return 1;
  }
  const auto result = strataslam::fuse(map.value(), opt);
  if (!result) {
    std::fprintf(stderr, "%s\n", result.error().message.c_str());
    return 1;
  }
  const auto w = strataslam::write_fusion_outputs(dir + "/fused", map.value(), result.value());
  if (!w) {
    std::fprintf(stderr, "%s\n", w.error().message.c_str());
    return 1;
  }
  for (const auto& warn : result.value().warnings)
    std::fprintf(stderr, "warning: %s\n", warn.c_str());
  std::printf("%s", result.value().transforms_yaml().c_str());
  return 0;
}
