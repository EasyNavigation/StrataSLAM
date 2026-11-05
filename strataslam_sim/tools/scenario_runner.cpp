// Runs one scenario with a given seed and thread count and prints one JSON line with every reported
// metric and the acceptance failures; optionally writes the map. Used by scripts/run_seeds.py for
// the multi-seed acceptance (ADR-0035) and for bit-exact determinism checks. Usage:
// strataslam_scenario_runner <Sxx> <seed> [threads] [map_out_dir] [--pixel-sigma px] [--image WxH]
//   --pixel-sigma / --image change the simulated camera (default 0.5 px, 320x240) for noise
//   studies.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "strataslam/io/map_io.hpp"
#include "strataslam_sim/scenarios.hpp"

namespace {

std::string json_escape(const std::string& s) {
  std::string e;
  for (char c : s) {
    if (c == '"' || c == '\\') e += '\\';
    if (c == '\n') {
      e += "\\n";
      continue;
    }
    e += c;
  }
  return e;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> pos;
  double pixel_sigma = -1;
  int width = 0, height = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--pixel-sigma" && i + 1 < argc)
      pixel_sigma = std::atof(argv[++i]);
    else if (a == "--image" && i + 1 < argc &&
             std::sscanf(argv[++i], "%dx%d", &width, &height) == 2) {
    } else
      pos.push_back(a);
  }
  if (pos.size() < 2) {
    std::fprintf(
        stderr, "usage: %s <Sxx> <seed> [threads] [map_out_dir] [--pixel-sigma px] [--image WxH]\n",
        argv[0]);
    return 2;
  }
  const std::string id = pos[0];
  const uint64_t seed = std::strtoull(pos[1].c_str(), nullptr, 10);
  auto spec = strataslam_sim::make_scenario(id, seed);
  if (pos.size() > 2) spec.slam.runtime.threads = std::atoi(pos[2].c_str());
  if (pixel_sigma >= 0) spec.sim.pixel_sigma = pixel_sigma;
  if (width > 0 && height > 0) {
    spec.sim.image_width = width;
    spec.sim.image_height = height;
  }
  const auto r = strataslam_sim::run_scenario(spec);
  std::printf("{\"scenario\":\"%s\",\"seed\":%llu,\"threads\":%d,\"passed\":%s,\"values\":{",
              json_escape(id).c_str(), static_cast<unsigned long long>(seed),
              spec.slam.runtime.threads, r.passed() ? "true" : "false");
  bool first = true;
  for (const auto& [k, v] : r.report.values) {
    // Python's json module reads NaN/Infinity; a bare "nan" would not parse.
    if (std::isfinite(v))
      std::printf("%s\"%s\":%.10g", first ? "" : ",", json_escape(k).c_str(), v);
    else
      std::printf("%s\"%s\":%s", first ? "" : ",", json_escape(k).c_str(),
                  std::isnan(v) ? "NaN" : (v > 0 ? "Infinity" : "-Infinity"));
    first = false;
  }
  std::printf("},\"failures\":[");
  first = true;
  for (const auto& f : r.failures) {
    std::printf("%s\"%s\"", first ? "" : ",", json_escape(f).c_str());
    first = false;
  }
  std::printf("]}\n");
  if (pos.size() > 3) {
    const auto w = strataslam::write_map(pos[3], r.map);
    if (!w) {
      std::fprintf(stderr, "write_map: %s\n", w.error().message.c_str());
      return 1;
    }
  }
  return 0;
}
