#pragma once

#include <cmath>
#include <cstdint>

namespace strataslam_sim {

inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline uint64_t hash_combine(uint64_t a, uint64_t b) {
  return splitmix64(a ^ (splitmix64(b) + 0x517cc1b727220a95ULL));
}
inline uint64_t hash_combine(uint64_t a, uint64_t b, uint64_t c) {
  return hash_combine(hash_combine(a, b), c);
}
inline uint64_t hash_combine(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  return hash_combine(hash_combine(a, b, c), d);
}

/// Small deterministic generator; cheap to construct from a (seed, stream, index) hash so that
/// noise does not depend on iteration order or thread count.
class Rng {
 public:
  explicit Rng(uint64_t seed) : s_(seed) {}
  uint64_t next() { return s_ = splitmix64(s_); }
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }  // [0,1)
  double uniform(double a, double b) { return a + (b - a) * uniform(); }
  double normal() {
    const double u1 = 1.0 - uniform(), u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
  }
  uint32_t below(uint32_t n) { return static_cast<uint32_t>(next() % n); }

 private:
  uint64_t s_;
};

}  // namespace strataslam_sim
