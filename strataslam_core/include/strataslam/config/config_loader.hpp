#pragma once

#include <optional>
#include <string>
#include <vector>

#include "strataslam/config/config.hpp"

namespace strataslam {

struct Diagnostic {
  std::string path;  ///< dotted key path, or "(profile)" for the file as a whole
  std::string message;

  /// "config error: <path>: <message>" (path omitted when empty).
  std::string format() const;
};

struct ConfigResult {
  std::optional<Config> config;  ///< set only when `errors` is empty
  std::vector<Diagnostic> errors;
  std::vector<Diagnostic> warnings;

  bool ok() const { return errors.empty(); }
};

/// Parse, apply defaults and validate. All errors are accumulated. The profile is the whole
/// configuration (ADR-0053).
ConfigResult load_profile_string(const std::string& yaml);
ConfigResult load_profile_file(const std::string& path);

/// Every key path a profile can hold ("sensors.camera.rgb_topic"), sections included, as the loader
/// knows them.
std::vector<std::string> profile_keys();

}  // namespace strataslam
