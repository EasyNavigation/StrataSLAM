#include "strataslam/common/logging.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <unistd.h>

namespace strataslam {

int debug_channel(const char* channel) {
  static std::mutex m;
  static std::map<std::string, int> cache;
  const std::lock_guard<std::mutex> lock(m);
  const auto it = cache.find(channel);
  if (it != cache.end()) return it->second;
  const char* v = std::getenv((std::string("STRATASLAM_DEBUG_") + channel).c_str());
  int n = 0;
  if (v && *v) {
    n = std::atoi(v);
    if (n <= 0) n = 1;
  }
  cache.emplace(channel, n);
  return n;
}

bool any_debug_channel() {
  for (char** e = environ; e && *e; ++e)
    if (std::string(*e).rfind("STRATASLAM_DEBUG_", 0) == 0) return true;
  return false;
}

std::string strprintf(const char* format, ...) {
  va_list a;
  va_start(a, format);
  va_list b;
  va_copy(b, a);
  const int n = std::vsnprintf(nullptr, 0, format, a);
  va_end(a);
  std::string out(n > 0 ? static_cast<size_t>(n) : 0, '\0');
  if (n > 0) std::vsnprintf(out.data(), out.size() + 1, format, b);
  va_end(b);
  return out;
}

}  // namespace strataslam
