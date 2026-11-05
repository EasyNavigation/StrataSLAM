#pragma once

#include <string>

namespace strataslam {

enum class LogLevel { Debug, Info, Warn, Error };

/// Core logging interface. The ROS node adapts it to rclcpp; no global state.
class Logger {
 public:
  virtual ~Logger() = default;
  virtual void log(LogLevel level, const std::string& message) = 0;
};

/// Logger that discards everything.
class NullLogger final : public Logger {
 public:
  void log(LogLevel, const std::string&) override {}
};

/// Developer traces, off by default: `STRATASLAM_DEBUG_<CHANNEL>=<n>` in the environment (read once
/// per channel) enables a channel; the traces are sent to the Logger at Debug level. Returns n (1
/// when the value is not a number), or 0 when the channel is off. Channels: ALT, BACKEND, FLOORS,
/// IMU, LOOPS, TRACK.
int debug_channel(const char* channel);
/// True if any STRATASLAM_DEBUG_* variable is set (tools then log at Debug level to stderr).
bool any_debug_channel();

/// printf-style formatting for log messages.
std::string strprintf(const char* format, ...) __attribute__((format(printf, 1, 2)));

}  // namespace strataslam
