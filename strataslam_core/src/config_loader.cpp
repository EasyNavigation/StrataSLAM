#include "strataslam/config/config_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <set>
#include <sstream>
#include <utility>

namespace strataslam {

std::string Diagnostic::format() const {
  return path.empty() ? "config error: " + message : "config error: " + path + ": " + message;
}

namespace {

struct Diagnostics {
  std::vector<Diagnostic> errors;
  std::vector<Diagnostic> warnings;
  void error(const std::string& path, const std::string& msg) { errors.push_back({path, msg}); }
  void warn(const std::string& path, const std::string& msg) { warnings.push_back({path, msg}); }
  std::set<std::string>* keys =
      nullptr;  ///< when set, every key path a section knows (profile_keys())
};

size_t edit_distance(const std::string& a, const std::string& b) {
  std::vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
  for (size_t j = 0; j <= b.size(); ++j) prev[j] = j;
  for (size_t i = 1; i <= a.size(); ++i) {
    cur[0] = i;
    for (size_t j = 1; j <= b.size(); ++j) {
      const size_t sub = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
      cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, sub});
    }
    std::swap(prev, cur);
  }
  return prev[b.size()];
}

/// Typed, key-tracking view over a YAML mapping. `finish()` reports unknown keys.
class Section {
 public:
  Section(const YAML::Node& node, std::string path, Diagnostics* diag)
      : node_(node), path_(std::move(path)), diag_(diag) {
    if (node_.IsDefined() && !node_.IsNull() && !node_.IsMap()) {
      // The top level is reported once, by the loader ("the profile must be a YAML map").
      if (!path_.empty()) diag_->error(path_, "expected a map");
      node_ = YAML::Node();
    }
  }

  const std::string& path() const { return path_; }
  bool present(const char* key) {
    known_.insert(key);
    return node_.IsMap() && node_[key].IsDefined();
  }
  std::string child_path(const char* key) const { return path_.empty() ? key : path_ + "." + key; }

  Section sub(const char* key) {
    known_.insert(key);
    YAML::Node n = node_.IsMap() ? node_[key] : YAML::Node();
    return Section(n, child_path(key), diag_);
  }
  bool has_sub(const char* key) {
    known_.insert(key);
    return node_.IsMap() && node_[key].IsDefined();
  }

  template <typename T>
  void get(const char* key, T& out) {
    known_.insert(key);
    if (!node_.IsMap()) return;
    YAML::Node n = node_[key];
    if (!n.IsDefined()) return;
    if (!n.IsScalar()) {
      diag_->error(child_path(key), "expected a scalar value");
      return;
    }
    try {
      out = n.as<T>();
    } catch (const YAML::Exception&) {
      diag_->error(child_path(key), std::string("invalid value: \"") + n.Scalar() + "\"");
    }
  }

  void get(const char* key, std::optional<double>& out) {
    known_.insert(key);
    if (!node_.IsMap()) return;
    YAML::Node n = node_[key];
    if (!n.IsDefined()) return;
    if (n.IsNull() || n.Scalar() == "null" || n.Scalar() == "~") {
      out.reset();
      return;
    }
    if (!n.IsScalar()) {
      diag_->error(child_path(key), "expected a scalar value");
      return;
    }
    try {
      out = n.as<double>();
    } catch (const YAML::Exception&) {
      diag_->error(child_path(key), std::string("invalid value: \"") + n.Scalar() + "\"");
    }
  }

  template <typename E>
  void get_enum(const char* key, E& out, std::initializer_list<std::pair<const char*, E>> table) {
    known_.insert(key);
    if (!node_.IsMap()) return;
    YAML::Node n = node_[key];
    if (!n.IsDefined()) return;
    const std::string s = n.IsScalar() ? n.Scalar() : "";
    std::string allowed;
    for (const auto& kv : table) {
      if (s == kv.first) {
        out = kv.second;
        return;
      }
      allowed += (allowed.empty() ? "" : ", ") + std::string(kv.first);
    }
    diag_->error(child_path(key), "invalid value: \"" + s + "\" (allowed: " + allowed + ")");
  }

  /// A key that profiles no longer have: an error that says what to write instead (not listed by
  /// profile_keys()).
  void removed(const char* key, const std::string& instead) {
    removed_.insert(key);
    if (node_.IsMap() && node_[key].IsDefined())
      diag_->error(child_path(key), "removed: " + instead);
  }

  void finish() {
    if (diag_->keys) {
      for (const auto& k : known_) diag_->keys->insert(child_path(k.c_str()));
    }
    if (!node_.IsMap()) return;
    for (const auto& kv : node_) {
      if (!kv.first.IsScalar()) {  // `? {a: 1} : x` (a complex key): not a profile key
        diag_->error(path_.empty() ? std::string("(root)") : path_, "keys must be plain names");
        continue;
      }
      const std::string key = kv.first.as<std::string>();
      if (known_.count(key) || removed_.count(key)) continue;
      std::string msg = "unknown key";
      size_t best = 3;
      std::string suggestion;
      for (const auto& k : known_) {
        const size_t d = edit_distance(key, k);
        if (d < best) {
          best = d;
          suggestion = k;
        }
      }
      if (!suggestion.empty()) msg += "; did you mean \"" + suggestion + "\"?";
      diag_->error(child_path(key.c_str()), msg);
    }
  }

 private:
  YAML::Node node_;
  std::string path_;
  Diagnostics* diag_;
  std::set<std::string> known_;
  std::set<std::string> removed_;
};

void check_positive(Diagnostics* d, const std::string& path, double v) {
  if (!(v > 0.0)) d->error(path, "must be positive");
}

void parse_robot(Section s, RobotConfig& r) {
  s.get_enum("platform", r.platform,
             {{"handheld", Platform::Handheld},
              {"ground", Platform::Ground},
              {"aerial", Platform::Aerial}});
  s.get("base_frame", r.base_frame);
  s.get("odom_frame", r.odom_frame);
  s.get("map_frame", r.map_frame);
  s.finish();
}

void parse_camera(Section s, CameraConfig& c, Diagnostics* d) {
  c.enabled = true;  // a present section means enabled unless stated otherwise
  s.get("enabled", c.enabled);
  s.removed("preset",
            "camera presets were removed (ADR-0053): write the camera's topics, depth range and "
            "IMU in the profile "
            "(docs/user/profiles.md lists the values of the tested cameras)");
  s.get("rgb_topic", c.rgb_topic);
  s.get("depth_topic", c.depth_topic);
  s.get("info_topic", c.info_topic);
  s.get("depth_min", c.depth_min);
  s.get("depth_max", c.depth_max);
  s.get_enum("features", c.features, {{"orb", FeatureType::Orb}, {"xfeat", FeatureType::Xfeat}});
  s.get("max_features", c.max_features);
  if (c.enabled) {
    const char* topics[] = {"rgb_topic", "depth_topic", "info_topic"};
    const std::string* vals[] = {&c.rgb_topic, &c.depth_topic, &c.info_topic};
    for (int i = 0; i < 3; ++i) {
      if (vals[i]->empty()) {
        d->error(s.child_path(topics[i]), "required");
      }
    }
    check_positive(d, s.child_path("depth_min"), c.depth_min);
    if (!(c.depth_min < c.depth_max)) {
      d->error(s.child_path("depth_min"), "must be less than depth_max");
    }
    if (c.max_features <= 0) d->error(s.child_path("max_features"), "must be positive");
  }
  s.finish();
}

void parse_lidar(Section s, LidarConfig& l, Diagnostics* d) {
  l.enabled = true;
  s.get("enabled", l.enabled);
  s.get("topic", l.topic);
  std::string type = "3d";
  s.get("type", type);
  if (type != "3d") d->error(s.child_path("type"), "only \"3d\" is supported in v1");
  s.get("min_range", l.min_range);
  s.get("max_range", l.max_range);
  s.get_enum(
      "deskew", l.deskew,
      {{"auto", DeskewMode::Auto}, {"true", DeskewMode::True}, {"false", DeskewMode::False}});
  if (l.enabled) {
    if (l.topic.empty()) d->error(s.child_path("topic"), "required");
    if (l.min_range < 0.0) d->error(s.child_path("min_range"), "must not be negative");
    if (!(l.min_range < l.max_range)) {
      d->error(s.child_path("min_range"), "must be less than max_range");
    }
  }
  s.finish();
}

void parse_imu(Section s, ImuConfig& i, Diagnostics* d) {
  i.enabled = true;
  s.get("enabled", i.enabled);
  s.get("topic", i.topic);
  s.get("floor_changes", i.floor_changes);
  s.removed("noise_preset",
            "camera presets were removed (ADR-0053): give the IMU noise in `noise:` (or leave it "
            "out for the "
            "nominal values)");
  if (s.has_sub("noise")) {
    Section n = s.sub("noise");
    n.get("accel_noise_density", i.noise.accel_noise_density);
    n.get("gyro_noise_density", i.noise.gyro_noise_density);
    n.get("accel_random_walk", i.noise.accel_random_walk);
    n.get("gyro_random_walk", i.noise.gyro_random_walk);
    n.finish();
    const std::pair<const char*, double> vals[] = {
        {"accel_noise_density", i.noise.accel_noise_density},
        {"gyro_noise_density", i.noise.gyro_noise_density},
        {"accel_random_walk", i.noise.accel_random_walk},
        {"gyro_random_walk", i.noise.gyro_random_walk}};
    for (const auto& kv : vals) check_positive(d, n.child_path(kv.first), kv.second);
  }
  if (i.enabled && i.topic.empty()) d->error(s.child_path("topic"), "required");
  s.finish();
}

void parse_barometer(Section s, BarometerConfig& b, Diagnostics* d) {
  b.enabled = true;
  s.get("enabled", b.enabled);
  s.get("topic", b.topic);
  s.get("reference_topic", b.reference_topic);
  s.get("temperature_k", b.temperature_k);
  s.get("settle_time_s", b.settle_time_s);
  if (b.enabled) {
    if (b.topic.empty()) d->error(s.child_path("topic"), "required");
    check_positive(d, s.child_path("temperature_k"), b.temperature_k);
    check_positive(d, s.child_path("settle_time_s"), b.settle_time_s);
  }
  s.finish();
}

void parse_external_odometry(Section s, ExternalOdometryConfig& e, Diagnostics* d) {
  e.enabled = true;
  s.get("enabled", e.enabled);
  s.get("topic", e.topic);
  s.get("frame", e.frame);
  if (e.enabled && e.topic.empty() == e.frame.empty()) {
    d->error(s.path(),
             "needs either a topic (nav_msgs/Odometry) or a frame (TF frame -> robot.base_frame)");
  }
  s.finish();
}

void parse_sensors(Section ms, Config& cfg, Diagnostics* d) {
  if (ms.has_sub("camera")) parse_camera(ms.sub("camera"), cfg.sensors.camera, d);
  if (ms.has_sub("lidar")) parse_lidar(ms.sub("lidar"), cfg.sensors.lidar, d);
  if (ms.has_sub("imu")) parse_imu(ms.sub("imu"), cfg.sensors.imu, d);
  if (ms.has_sub("barometer")) parse_barometer(ms.sub("barometer"), cfg.sensors.barometer, d);
  if (ms.has_sub("external_odometry")) {
    parse_external_odometry(ms.sub("external_odometry"), cfg.sensors.external_odometry, d);
  }
  ms.finish();
}

void parse_fusion(Section s, FusionConfig& f) {
  s.get_enum("odometry_from", f.odometry_from,
             {{"auto", OdometryFrom::Auto},
              {"camera", OdometryFrom::Camera},
              {"lidar", OdometryFrom::Lidar},
              {"external", OdometryFrom::External}});
  s.get_enum("loop_closure_from", f.loop_closure_from,
             {{"auto", LoopClosureFrom::Auto},
              {"camera", LoopClosureFrom::Camera},
              {"lidar", LoopClosureFrom::Lidar},
              {"both", LoopClosureFrom::Both}});
  f.colorize_cloud_explicit = s.present("colorize_cloud");
  s.get("colorize_cloud", f.colorize_cloud);
  s.get("vocabulary", f.vocabulary);
  s.finish();
}

void parse_multifloor(Section s, MultifloorConfig& m, Diagnostics* d) {
  const bool has_enabled = s.present("enabled"), has_mode = s.present("mode");
  s.get("enabled", m.enabled);
  s.get_enum("mode", m.mode,
             {{"floors", LevelMode::Floors},
              {"single", LevelMode::Single},
              {"planar", LevelMode::Planar}});
  if (has_enabled && has_mode && m.enabled != (m.mode == LevelMode::Floors)) {
    d->error(s.child_path("enabled"),
             "contradicts mode (enabled: false is mode: single; give only mode)");
  }
  if (has_enabled && !has_mode && !m.enabled) m.mode = LevelMode::Single;
  m.enabled = m.mode == LevelMode::Floors;
  s.get("floor_height_prior", m.floor_height_prior);
  s.get("split_on_tracking_loss", m.split_on_tracking_loss);
  s.get("tracking_loss_timeout", m.tracking_loss_timeout);
  s.get("min_vertical_change", m.min_vertical_change);
  if (m.floor_height_prior)
    check_positive(d, s.child_path("floor_height_prior"), *m.floor_height_prior);
  check_positive(d, s.child_path("tracking_loss_timeout"), m.tracking_loss_timeout);
  check_positive(d, s.child_path("min_vertical_change"), m.min_vertical_change);
  s.finish();
}

void parse_output(Section s, OutputConfig& o, Diagnostics* d) {
  s.get("map_dir", o.map_dir);
  s.get("voxel_size", o.voxel_size);
  s.get("keyframe_translation", o.keyframe_translation);
  s.get("keyframe_rotation_deg", o.keyframe_rotation_deg);
  s.get("publish_tf", o.publish_tf);
  if (o.map_dir.empty()) d->error(s.child_path("map_dir"), "must not be empty");
  if (!(o.voxel_size >= 0.005 && o.voxel_size <= 1.0)) {
    d->error(s.child_path("voxel_size"), "must be in [0.005, 1.0]");
  }
  check_positive(d, s.child_path("keyframe_translation"), o.keyframe_translation);
  if (!(o.keyframe_rotation_deg > 0.0 && o.keyframe_rotation_deg <= 180.0)) {
    d->error(s.child_path("keyframe_rotation_deg"), "must be in (0, 180]");
  }
  s.finish();
}

void parse_runtime(Section s, RuntimeConfig& r, Diagnostics* d) {
  s.get("seed", r.seed);
  s.get("threads", r.threads);
  if (r.threads < 0) d->error(s.child_path("threads"), "must not be negative");
  s.finish();
}

void cross_validate(const Config& c, Diagnostics* d) {
  const auto& s = c.sensors;
  if (!s.camera.enabled && !s.lidar.enabled) {
    d->error("sensors", "at least one sensor must be enabled (camera or lidar)");
  }
  const auto require = [&](bool cond, const char* path, const char* value, const char* needs) {
    if (!cond) {
      d->error(path, std::string("\"") + value + "\" requires " + needs + ": true");
    }
  };
  switch (c.fusion.odometry_from) {
    case OdometryFrom::Camera:
      require(s.camera.enabled, "fusion.odometry_from", "camera", "sensors.camera.enabled");
      break;
    case OdometryFrom::Lidar:
      require(s.lidar.enabled, "fusion.odometry_from", "lidar", "sensors.lidar.enabled");
      break;
    case OdometryFrom::External:
      d->error("fusion.odometry_from",
               "\"external\" is not an odometry source: the external odometry is the motion prior "
               "of the camera tracking (keep odometry_from camera or auto)");
      break;
    case OdometryFrom::Auto:
      break;
  }
  if (s.external_odometry.enabled && resolve_modes(c).odometry != OdometrySource::Camera) {
    d->warn("sensors.external_odometry",
            "only the camera tracking uses it: with LiDAR odometry it is ignored");
  }
  switch (c.fusion.loop_closure_from) {
    case LoopClosureFrom::Camera:
      require(s.camera.enabled, "fusion.loop_closure_from", "camera", "sensors.camera.enabled");
      break;
    case LoopClosureFrom::Lidar:
      require(s.lidar.enabled, "fusion.loop_closure_from", "lidar", "sensors.lidar.enabled");
      break;
    case LoopClosureFrom::Both:
      require(s.camera.enabled, "fusion.loop_closure_from", "both", "sensors.camera.enabled");
      require(s.lidar.enabled, "fusion.loop_closure_from", "both", "sensors.lidar.enabled");
      break;
    case LoopClosureFrom::Auto:
      break;
  }
  // Only when asked for: it is on by default, and a LiDAR-only profile need not say no (audit
  // A7-06).
  if (c.fusion.colorize_cloud && c.fusion.colorize_cloud_explicit && !s.camera.enabled) {
    d->warn("fusion.colorize_cloud", "without an enabled camera the cloud is not colourized");
  }
  if (c.multifloor.mode == LevelMode::Planar && c.robot.platform != Platform::Ground) {
    d->error(
        "multifloor.mode",
        "\"planar\" requires robot.platform: ground (a hand-held or flying sensor changes height)");
  }
  if (c.robot.platform == Platform::Aerial && !s.imu.enabled) {
    d->warn("robot.platform",
            "aerial platform without an IMU: the flight estimate is not reliable (strong warning)");
  }
}

/// The sections of a profile, in order; `top` is the document's root section.
void parse_root(Section& top, const YAML::Node& root, const char* head_key, Config& cfg,
                Diagnostics* d) {
  Section head = top.sub(head_key);
  head.get("version", cfg.version);
  head.finish();
  if (!root[head_key].IsDefined()) {
    d->error("strataslam", "required section is missing");
  } else if (cfg.version != 1) {
    d->error(std::string(head_key) + ".version", "unsupported profile version (supported: 1)");
  }
  parse_robot(top.sub("robot"), cfg.robot);
  parse_sensors(top.sub("sensors"), cfg, d);
  parse_fusion(top.sub("fusion"), cfg.fusion);
  parse_multifloor(top.sub("multifloor"), cfg.multifloor, d);
  parse_output(top.sub("output"), cfg.output, d);
  parse_runtime(top.sub("runtime"), cfg.runtime, d);
  top.finish();
}

}  // namespace

std::vector<std::string> profile_keys() {
  // Every section present, so that every parser runs and says which keys it knows; the sensors are
  // disabled, so nothing is required of them. A sensor section added to the loader shows up here by
  // its name (`has_sub`).
  const YAML::Node root = YAML::Load(
      "{strataslam: {version: 1}, robot: {}, sensors: {camera: {enabled: false}, lidar: {enabled: "
      "false}, "
      "imu: {enabled: false, noise: {}}, barometer: {enabled: false}, external_odometry: {enabled: "
      "false}}, "
      "fusion: {}, multifloor: {}, output: {}, runtime: {}}");
  std::set<std::string> keys;
  Diagnostics d;
  d.keys = &keys;
  Config cfg;
  Section top(root, "", &d);
  parse_root(top, root, "strataslam", cfg, &d);
  return {keys.begin(), keys.end()};
}

ResolvedModes resolve_modes(const Config& c) {
  const auto& s = c.sensors;
  ResolvedModes m{};
  switch (c.fusion.odometry_from) {
    case OdometryFrom::Camera:
      m.odometry = OdometrySource::Camera;
      break;
    case OdometryFrom::Lidar:
      m.odometry = OdometrySource::Lidar;
      break;
    case OdometryFrom::External:
      m.odometry = OdometrySource::External;
      break;
    case OdometryFrom::Auto:  // the external odometry is a prior of the camera, never the odometry
      m.odometry = s.lidar.enabled ? OdometrySource::Lidar : OdometrySource::Camera;
      break;
  }
  m.verify_loop_with_lidar = false;
  switch (c.fusion.loop_closure_from) {
    case LoopClosureFrom::Camera:
      m.loop_closure = LoopSource::Camera;
      break;
    case LoopClosureFrom::Lidar:
      m.loop_closure = LoopSource::Lidar;
      break;
    case LoopClosureFrom::Both:
      m.loop_closure = LoopSource::Both;
      break;
    case LoopClosureFrom::Auto:
      if (s.camera.enabled && s.lidar.enabled) {
        m.loop_closure = LoopSource::Camera;
        m.verify_loop_with_lidar = true;
      } else {
        m.loop_closure = s.lidar.enabled ? LoopSource::Lidar : LoopSource::Camera;
      }
      break;
  }
  return m;
}

namespace {

/// A key written twice in one map: YAML says it is an error, yaml-cpp keeps the first silently (a
/// block of settings pasted at the end of a profile did nothing; audit A4A5-07, A7-04). Reported
/// with its path.
void find_duplicate_keys(const YAML::Node& n, const std::string& path,
                         std::vector<std::string>* dups) {
  if (n.IsMap()) {
    std::set<std::string> seen;
    for (const auto& kv : n) {
      const std::string key = kv.first.IsScalar() ? kv.first.Scalar() : std::string("?");
      std::string child = path;
      if (!child.empty()) child += '.';
      child += key;
      if (!seen.insert(key).second) dups->push_back(child);
      find_duplicate_keys(kv.second, child, dups);
    }
  } else if (n.IsSequence()) {
    for (size_t i = 0; i < n.size(); ++i) {
      std::string item = path;
      item += '[';
      item += std::to_string(i);
      item += ']';
      find_duplicate_keys(n[i], item, dups);
    }
  }
}

}  // namespace

ConfigResult load_profile_string(const std::string& yaml) {
  Diagnostics d;
  ConfigResult result;
  YAML::Node root;
  // Errors about the file as a whole carry the path "(profile)", never an empty one (audit A7-05).
  try {
    const std::vector<YAML::Node> docs = YAML::LoadAll(yaml);
    if (docs.size() > 1) {
      result.errors.push_back(
          {"(profile)", "the file holds " + std::to_string(docs.size()) +
                            " YAML documents (separated by ---): a profile is one document"});
      return result;
    }
    root = docs.empty() ? YAML::Node() : docs.front();
  } catch (const YAML::Exception& e) {
    result.errors.push_back({"(profile)", std::string("invalid YAML: ") + e.what()});
    return result;
  }
  {
    std::vector<std::string> dups;
    find_duplicate_keys(root, "", &dups);
    for (const auto& k : dups)
      result.errors.push_back({k, "written twice (only one value can hold)"});
    if (!dups.empty()) return result;
  }

  Config cfg;
  try {
    Section top(root, "", &d);
    if (!root.IsMap()) {
      d.error("(profile)", "the profile must be a YAML map");
    } else {
      // Profiles written before the project was renamed (0.2.x) have an `easyslam:` header: still
      // accepted, with a warning, unless the new header is also present.
      const bool legacy_head = !root["strataslam"].IsDefined() && root["easyslam"].IsDefined();
      const char* head_key = legacy_head ? "easyslam" : "strataslam";
      if (legacy_head)
        d.warn("easyslam", "the project is now StrataSLAM: rename this section to \"strataslam\"");
      parse_root(top, root, head_key, cfg, &d);
      cross_validate(cfg, &d);
    }
  } catch (const YAML::Exception& e) {
    // Every malformed input should have its own message above; this keeps the promise that
    // validation returns errors and never throws, for the shapes nobody thought of.
    d.error("", std::string("invalid profile structure: ") + e.what());
  }

  result.errors = std::move(d.errors);
  result.warnings = std::move(d.warnings);
  if (result.errors.empty()) result.config = cfg;
  return result;
}

ConfigResult load_profile_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    ConfigResult r;
    r.errors.push_back({"", "no se pudo abrir el perfil: " + path});
    return r;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  ConfigResult r = load_profile_string(ss.str());
  // A vocabulary next to the profile is found wherever the command is run from ("~" is the node's
  // to expand).
  if (r.config) {
    std::string& v = r.config->fusion.vocabulary;
    if (!v.empty() && v != "default" && v[0] != '/' && v[0] != '~') {
      v = (std::filesystem::path(path).parent_path() / v).lexically_normal().string();
    }
  }
  return r;
}

}  // namespace strataslam
