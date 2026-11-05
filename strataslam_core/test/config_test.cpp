#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>

#include "strataslam/config/config_loader.hpp"

using namespace strataslam;

namespace {

const char* kBase = R"(
strataslam: {version: 1}
sensors:
  lidar: {enabled: true}
)";

std::string with(const std::string& extra) { return std::string(kBase) + extra; }

std::vector<std::string> errors_of(const ConfigResult& r) {
  std::vector<std::string> out;
  for (const auto& e : r.errors) out.push_back(e.format());
  return out;
}

bool has_error(const ConfigResult& r, const std::string& text) {
  const auto e = errors_of(r);
  return std::find(e.begin(), e.end(), text) != e.end();
}

}  // namespace

TEST(Profiles, EveryShippedProfileValidates) {
  for (const char* name : {"handheld_rgbd", "handheld_oak_d_lite", "lidar_only", "combined_aerial", "summit_lidar", "tiago_lidar_camera",
                           "tiago_camera_only"}) {
    const auto r = load_profile_file(std::string(STRATASLAM_PROFILE_DIR) + "/" + name + ".yaml");
    EXPECT_TRUE(r.ok()) << name << ": " << (r.errors.empty() ? "" : r.errors[0].format());
  }
}

TEST(Profiles, ExampleValuesParsed) {
  const auto r = load_profile_file(std::string(STRATASLAM_PROFILE_DIR) + "/handheld_rgbd.yaml");
  ASSERT_TRUE(r.ok());
  const Config& c = *r.config;
  EXPECT_TRUE(c.sensors.camera.enabled);
  EXPECT_EQ(c.sensors.camera.rgb_topic, "/camera/color/image_raw");
  EXPECT_EQ(c.sensors.imu.topic, "/camera/imu");
  EXPECT_EQ(c.runtime.seed, 42u);
  EXPECT_DOUBLE_EQ(*c.multifloor.floor_height_prior, 3.0);
}

TEST(Profiles, NullFloorHeightPrior) {
  const auto r = load_profile_file(std::string(STRATASLAM_PROFILE_DIR) + "/lidar_only.yaml");
  ASSERT_TRUE(r.ok());
  EXPECT_FALSE(r.config->multifloor.floor_height_prior.has_value());
}

namespace {

// The generic example profiles are templates: every option of a profile, active when the example uses it, commented
// (`# key: value`) with its default otherwise. The Summit and Tiago profiles are particular robots and bags.
const char* kTemplates[] = {"handheld_rgbd", "handheld_oak_d_lite", "lidar_only", "combined_aerial"};

struct OptionLine {
  size_t index;  ///< line number in the file
  int indent;    ///< YAML indentation of the key, `#` included for a commented one
  std::string path;
  bool commented;
  bool commented_parent;  ///< inside a commented section: uncommented with it, not alone
};

std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream in(path);
  std::vector<std::string> lines;
  for (std::string l; std::getline(in, l);) lines.push_back(l);
  return lines;
}

/// The options of a profile, active or commented, with their dotted paths. Explanatory comments never start with
/// "name:", so a line that does is an option (and the test below checks that the loader knows it).
std::vector<OptionLine> option_lines(const std::vector<std::string>& lines) {
  static const std::regex re(R"(^( *)(#( +))?([a-z_]+):(\s.*)?$)");
  std::vector<OptionLine> out;
  std::vector<OptionLine> stack;
  for (size_t i = 0; i < lines.size(); ++i) {
    std::smatch m;
    if (!std::regex_match(lines[i], m, re)) continue;
    const bool commented = m[2].matched;
    const int indent = static_cast<int>(m[1].length() + (commented ? m[3].length() - 1 : 0));
    while (!stack.empty() && stack.back().indent >= indent) stack.pop_back();
    OptionLine o{i, indent, m[4].str(), commented, !stack.empty() && stack.back().commented};
    if (!stack.empty()) o.path = stack.back().path + "." + o.path;
    stack.push_back(o);
    out.push_back(o);
  }
  return out;
}

std::string join(const std::vector<std::string>& lines) {
  std::string t;
  for (const auto& l : lines) t += l + "\n";
  return t;
}

}  // namespace

// Every option the loader knows is in every template, once (a new option has to be added to them), and every
// commented line that looks like an option is one.
TEST(Profiles, TheTemplatesHoldEveryOption) {
  const auto keys = profile_keys();
  ASSERT_GT(keys.size(), 40u);
  ASSERT_TRUE(std::count(keys.begin(), keys.end(), "sensors.imu.noise.gyro_random_walk"));
  for (const char* name : kTemplates) {
    const auto opts_ = option_lines(read_lines(std::string(STRATASLAM_PROFILE_DIR) + "/" + name + ".yaml"));
    std::map<std::string, int> seen;
    for (const auto& o : opts_) {
      ++seen[o.path];
      EXPECT_TRUE(std::count(keys.begin(), keys.end(), o.path)) << name << ": \"" << o.path << "\" is not an option";
    }
    for (const auto& k : keys) {
      if (k == "multifloor.enabled") continue;  // the old spelling of `mode` (false = single), contradicts it
      EXPECT_EQ(seen[k], 1) << name << ": option \"" << k << "\" should appear once, active or commented";
    }
  }
}

// Uncommenting any one commented option (a commented section with its keys) leaves a valid profile: the defaults and
// examples in the comments are real values, and no option is commented out of a place where it cannot go.
// The values of a disabled sensor are only checked once it is enabled: the loader skips them, as it does for the user.
TEST(Profiles, EveryCommentedOptionOfTheTemplatesIsValidWhenUncommented) {
  for (const char* name : kTemplates) {
    const auto lines = read_lines(std::string(STRATASLAM_PROFILE_DIR) + "/" + name + ".yaml");
    ASSERT_TRUE(load_profile_string(join(lines)).ok()) << name;
    const auto opts_ = option_lines(lines);
    int tried = 0;
    for (size_t k = 0; k < opts_.size(); ++k) {
      const auto& o = opts_[k];
      if (!o.commented || o.commented_parent) continue;
      auto edited = lines;
      const auto uncomment = [&](size_t i) { edited[i].erase(edited[i].find('#'), 2); };
      uncomment(o.index);
      for (size_t c = k + 1; c < opts_.size() && opts_[c].commented_parent && opts_[c].indent > o.indent; ++c) uncomment(opts_[c].index);
      const auto r = load_profile_string(join(edited));
      EXPECT_TRUE(r.ok()) << name << ": uncommenting \"" << o.path << "\": " << (r.errors.empty() ? "" : r.errors[0].format());
      ++tried;
    }
    EXPECT_GT(tried, 5) << name;
  }
}

TEST(Rules, NeedsCameraOrLidar) {
  const auto r = load_profile_string("strataslam: {version: 1}\nsensors:\n  imu: {enabled: true}\n");
  EXPECT_TRUE(has_error(r, "config error: sensors: at least one sensor must be enabled (camera or lidar)"));
}

TEST(Rules, ExternalOdometryNeedsATopicOrAFrame) {
  auto r = load_profile_string(with("  external_odometry: {enabled: true}\n"));
  EXPECT_TRUE(has_error(r, "config error: sensors.external_odometry: needs either a topic (nav_msgs/Odometry) or a frame (TF frame -> robot.base_frame)"));
  r = load_profile_string(with("  external_odometry: {enabled: true, topic: /odom, frame: odom}\n"));
  EXPECT_TRUE(has_error(r, "config error: sensors.external_odometry: needs either a topic (nav_msgs/Odometry) or a frame (TF frame -> robot.base_frame)"));
  r = load_profile_string(with("  external_odometry: {enabled: true, frame: odom}\n"));
  EXPECT_TRUE(r.ok());
  EXPECT_EQ(r.config->sensors.external_odometry.frame, "odom");
}

TEST(Rules, ExternalOdometryIsAPriorOfTheCameraNotAnOdometry) {
  auto r = load_profile_string(with("  external_odometry: {enabled: true, topic: /odom}\nfusion: {odometry_from: external}\n"));
  EXPECT_TRUE(has_error(r, "config error: fusion.odometry_from: \"external\" is not an odometry source: the external odometry is the "
                           "motion prior of the camera tracking (keep odometry_from camera or auto)"));
  r = load_profile_string("strataslam: {version: 1}\nsensors: {lidar: {enabled: true}, external_odometry: {enabled: true, frame: odom}}\n");
  EXPECT_TRUE(r.ok());
  EXPECT_TRUE(std::any_of(r.warnings.begin(), r.warnings.end(), [](const auto& w) {
    return w.format().find("only the camera tracking uses it") != std::string::npos; }));
}

TEST(Rules, OdometryFromRequiresSensor) {
  auto r = load_profile_string(with("fusion: {odometry_from: camera}\n"));
  EXPECT_TRUE(has_error(r, "config error: fusion.odometry_from: \"camera\" requires sensors.camera.enabled: true"));
  r = load_profile_string("strataslam: {version: 1}\nsensors: {camera: {rgb_topic: a, depth_topic: b, info_topic: c}}\nfusion: {odometry_from: lidar}\n");
  EXPECT_TRUE(has_error(r, "config error: fusion.odometry_from: \"lidar\" requires sensors.lidar.enabled: true"));
}

TEST(Rules, LevelModes) {
  auto r = load_profile_string(with(""));
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.config->multifloor.mode, LevelMode::Floors);
  EXPECT_TRUE(r.config->multifloor.enabled);
  r = load_profile_string(with("multifloor: {enabled: false}\n"));  // profiles from before `mode`
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.config->multifloor.mode, LevelMode::Single);
  EXPECT_FALSE(r.config->multifloor.enabled);
  r = load_profile_string(with("multifloor: {mode: single}\n"));
  ASSERT_TRUE(r.ok());
  EXPECT_FALSE(r.config->multifloor.enabled);
  r = load_profile_string(with("multifloor: {enabled: true, mode: planar}\n"));
  EXPECT_TRUE(has_error(r, "config error: multifloor.enabled: contradicts mode (enabled: false is mode: single; give only mode)"));
  r = load_profile_string(with("multifloor: {mode: planar}\n"));  // the default platform is handheld
  EXPECT_TRUE(has_error(r, "config error: multifloor.mode: \"planar\" requires robot.platform: ground (a hand-held or flying sensor "
                           "changes height)"));
  r = load_profile_string(with("robot: {platform: ground}\nmultifloor: {mode: planar}\n"));
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.config->multifloor.mode, LevelMode::Planar);
  EXPECT_FALSE(r.config->multifloor.enabled);
}

TEST(Rules, ColorizeWithoutCameraWarnsOnly) {
  const auto r = load_profile_string(with("fusion: {colorize_cloud: true}\n"));
  EXPECT_TRUE(r.ok());
  ASSERT_EQ(r.warnings.size(), 1u);
  EXPECT_EQ(r.warnings[0].path, "fusion.colorize_cloud");
}

// The profile can name its visual vocabulary (before, only the launch could: a camera-only profile run without
// `vocabulary:=` closed no loop, with one warning at start-up). A relative path is the profile's folder, so a profile
// and its vocabulary can travel together; absolute and "~" paths are kept (the node expands "~").
TEST(Rules, TheVocabularyComesFromTheProfileRelativeToItsFolder) {
  EXPECT_EQ(load_profile_string(with("")).config->fusion.vocabulary, "");
  EXPECT_EQ(load_profile_string(with("fusion: {vocabulary: vocab.bin}\n")).config->fusion.vocabulary, "vocab.bin");
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("strataslam_vocab_test_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir / "profiles");
  const auto load = [&](const std::string& vocabulary) {
    std::ofstream(dir / "profiles" / "p.yaml") << with("fusion: {vocabulary: \"" + vocabulary + "\"}\n");
    const auto r = load_profile_file((dir / "profiles" / "p.yaml").string());
    EXPECT_TRUE(r.ok());
    return r.config ? r.config->fusion.vocabulary : std::string("<invalid>");
  };
  EXPECT_EQ(load("vocab.bin"), (dir / "profiles" / "vocab.bin").string());
  EXPECT_EQ(load("../vocabularies/office.bin"), (dir / "vocabularies" / "office.bin").string());
  EXPECT_EQ(load("/opt/vocab.bin"), "/opt/vocab.bin");
  EXPECT_EQ(load("~/vocab.bin"), "~/vocab.bin");
  EXPECT_EQ(load("default"), "default");  // the shipped one, found by the node (ADR-0054), not a file here
  std::filesystem::remove_all(dir);
}

TEST(Rules, AerialWithoutImuWarns) {
  const auto r = load_profile_string(with("robot: {platform: aerial}\n"));
  EXPECT_TRUE(r.ok());
  const bool found = std::any_of(r.warnings.begin(), r.warnings.end(),
                                 [](const Diagnostic& d) { return d.path == "robot.platform"; });
  EXPECT_TRUE(found);
}

TEST(Rules, NumericRanges) {
  auto r = load_profile_string(with("output: {voxel_size: 2.0}\n"));
  EXPECT_TRUE(has_error(r, "config error: output.voxel_size: must be in [0.005, 1.0]"));
  r = load_profile_string(with("output: {voxel_size: 0.001}\n"));
  EXPECT_TRUE(has_error(r, "config error: output.voxel_size: must be in [0.005, 1.0]"));
  r = load_profile_string("strataslam: {version: 1}\nsensors:\n  lidar: {min_range: 5, max_range: 2}\n");
  EXPECT_TRUE(has_error(r, "config error: sensors.lidar.min_range: must be less than max_range"));
  r = load_profile_string("strataslam: {version: 1}\nsensors:\n  camera: {rgb_topic: a, depth_topic: b, info_topic: c, depth_min: 7, depth_max: 6}\n");
  EXPECT_TRUE(has_error(r, "config error: sensors.camera.depth_min: must be less than depth_max"));
  r = load_profile_string(with("multifloor: {tracking_loss_timeout: -1}\n"));
  EXPECT_TRUE(has_error(r, "config error: multifloor.tracking_loss_timeout: must be positive"));
}

TEST(Rules, UnknownKeySuggestion) {
  const auto r = load_profile_string("strataslam: {version: 1}\nsensors:\n  camra: {enabled: true}\n  lidar: {enabled: true}\n");
  EXPECT_TRUE(has_error(r, "config error: sensors.camra: unknown key; did you mean \"camera\"?"));
  EXPECT_FALSE(r.config.has_value());
}

TEST(Rules, UnknownKeyWithoutSuggestion) {
  const auto r = load_profile_string(with("zzzzzzzz: 1\n"));
  EXPECT_TRUE(has_error(r, "config error: zzzzzzzz: unknown key"));
}

TEST(Rules, AllErrorsAreReported) {
  const auto r = load_profile_string(R"(
strataslam: {version: 1}
sensors:
  camra: {enabled: true}
  external_odometry: {enabled: true}
output: {voxel_size: 5}
)");
  EXPECT_GE(r.errors.size(), 4u);  // unknown key, no sensor, missing topic, voxel size
  EXPECT_TRUE(has_error(r, "config error: sensors: at least one sensor must be enabled (camera or lidar)"));
  EXPECT_TRUE(has_error(r, "config error: sensors.external_odometry: needs either a topic (nav_msgs/Odometry) or a frame (TF frame -> robot.base_frame)"));
  EXPECT_TRUE(has_error(r, "config error: output.voxel_size: must be in [0.005, 1.0]"));
}

TEST(Rules, InvalidEnumAndLidar2d) {
  auto r = load_profile_string(with("robot: {platform: submarine}\n"));
  EXPECT_TRUE(has_error(r, "config error: robot.platform: invalid value: \"submarine\" (allowed: handheld, ground, aerial)"));
  r = load_profile_string("strataslam: {version: 1}\nsensors:\n  lidar: {type: 2d}\n");
  EXPECT_TRUE(has_error(r, "config error: sensors.lidar.type: only \"3d\" is supported in v1"));
}

TEST(Rules, BadYamlAndMissingHeader) {
  auto r = load_profile_string("strataslam: [unclosed");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.errors[0].message.rfind("invalid YAML", 0), 0u);
  r = load_profile_string("sensors:\n  lidar: {enabled: true}\n");
  EXPECT_TRUE(has_error(r, "config error: strataslam: required section is missing"));
  r = load_profile_string("strataslam: {version: 2}\nsensors:\n  lidar: {enabled: true}\n");
  EXPECT_TRUE(has_error(r, "config error: strataslam.version: unsupported profile version (supported: 1)"));
}

TEST(Rules, ProfilesFromBeforeTheRenameLoadWithAWarning) {
  // 0.2.x profiles start with `easyslam:`.
  auto r = load_profile_string("easyslam: {version: 1}\nsensors:\n  lidar: {enabled: true}\n");
  ASSERT_TRUE(r.ok());
  EXPECT_TRUE(std::any_of(r.warnings.begin(), r.warnings.end(), [](const Diagnostic& w) { return w.path == "easyslam"; }));
  r = load_profile_string("easyslam: {version: 2}\nsensors:\n  lidar: {enabled: true}\n");
  EXPECT_TRUE(has_error(r, "config error: easyslam.version: unsupported profile version (supported: 1)"));
  // Both headers: the new one is read and the old one is an unknown key, as any other.
  r = load_profile_string("strataslam: {version: 1}\neasyslam: {version: 1}\nsensors:\n  lidar: {enabled: true}\n");
  EXPECT_FALSE(r.ok());
}

// Camera presets were removed (ADR-0053): the profile is the whole configuration. A profile written with them says what
// to write instead, rather than "unknown key", and is not accepted with values it no longer gets.
TEST(Rules, PresetKeysOfOldProfilesSayWhatToWriteInstead) {
  auto r = load_profile_string("strataslam: {version: 1}\nsensors:\n  camera: {preset: realsense_d435i}\n  imu: {noise_preset: explicit}\n");
  EXPECT_FALSE(r.ok());
  const auto e = errors_of(r);
  const auto find = [&](const std::string& prefix) {
    return std::any_of(e.begin(), e.end(), [&](const std::string& m) { return m.rfind(prefix, 0) == 0; });
  };
  EXPECT_TRUE(find("config error: sensors.camera.preset: removed: camera presets were removed (ADR-0053)")) << ::testing::PrintToString(e);
  EXPECT_TRUE(find("config error: sensors.imu.noise_preset: removed: camera presets were removed (ADR-0053)")) << ::testing::PrintToString(e);
  for (const auto& m : e) EXPECT_EQ(m.find("unknown key"), std::string::npos) << m;
  const auto keys = profile_keys();
  EXPECT_FALSE(std::count(keys.begin(), keys.end(), "sensors.camera.preset"));  // not an option any more
  EXPECT_FALSE(std::count(keys.begin(), keys.end(), "sensors.imu.noise_preset"));
}

TEST(Rules, ACameraNeedsItsThreeTopics) {
  const auto r = load_profile_string("strataslam: {version: 1}\nsensors:\n  camera: {enabled: true}\n");
  for (const char* t : {"rgb_topic", "depth_topic", "info_topic"}) {
    EXPECT_TRUE(has_error(r, std::string("config error: sensors.camera.") + t + ": required")) << t;
  }
}

TEST(Rules, TheImuNoiseIsTheProfilesOrTheNominalDefaults) {
  const char* cam = "strataslam: {version: 1}\nsensors:\n  camera: {rgb_topic: /c, depth_topic: /d, info_topic: /i}\n";
  auto r = load_profile_string(std::string(cam) + "  imu:\n");  // an empty section: enabled, with every default
  ASSERT_TRUE(r.ok()) << (r.errors.empty() ? "" : r.errors[0].format());
  EXPECT_TRUE(r.config->sensors.imu.enabled);
  EXPECT_EQ(r.config->sensors.imu.topic, "/imu");
  EXPECT_DOUBLE_EQ(r.config->sensors.imu.noise.gyro_noise_density, 1.7e-4);
  r = load_profile_string(std::string(cam) + "  imu: {topic: /my/imu, noise: {gyro_noise_density: 0.01}}\n");
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.config->sensors.imu.topic, "/my/imu");
  EXPECT_DOUBLE_EQ(r.config->sensors.imu.noise.gyro_noise_density, 0.01);
  EXPECT_DOUBLE_EQ(r.config->sensors.imu.noise.accel_noise_density, 2.0e-3);  // the others keep their defaults
  // Audit AUD-007 (a camera without an imu section raised a yaml-cpp exception): no section, no IMU (ADR-0002).
  r = load_profile_string(cam);
  ASSERT_TRUE(r.ok());
  EXPECT_FALSE(r.config->sensors.imu.enabled);
}

TEST(Rules, ComplexKeysAreErrorsNotExceptions) {
  // Found by the audit's fuzzer: `? key: value` makes a map the key, which used to throw.
  const auto r = load_profile_string("strataslam: {version: 1}\nrobot:\n  platform: ground\n? odom_frame: odom\n  map_frame: map\n"
                                     "sensors:\n  lidar: {topic: /points}\n");
  EXPECT_FALSE(r.ok());
  EXPECT_TRUE(has_error(r, "config error: (root): keys must be plain names")) << (r.errors.empty() ? "" : r.errors[0].format());
}

TEST(Rules, EverySensorSectionMustBeAMap) {
  // Audit AUD-018: a scalar or a list in place of a sensor section was accepted as "present".
  for (const std::string preset : {"", "  camera: {rgb_topic: /c, depth_topic: /d, info_topic: /i}\n"}) {
    for (const std::string section : {"imu", "barometer", "lidar", "external_odometry"}) {
      for (const std::string value : {"5", "true", "[1, 2]", "/imu"}) {
        const std::string yaml = "strataslam: {version: 1}\nsensors:\n" + preset +
                                 (preset.empty() && section != "lidar" ? "  lidar: {topic: /points}\n" : "") + "  " + section + ": " + value + "\n";
        const auto r = load_profile_string(yaml);
        EXPECT_FALSE(r.ok()) << yaml;
        EXPECT_TRUE(has_error(r, "config error: sensors." + section + ": expected a map")) << yaml;
      }
    }
  }
  const auto r = load_profile_string("strataslam: {version: 1}\nsensors:\n  camera: 5\n");
  EXPECT_TRUE(has_error(r, "config error: sensors.camera: expected a map"));
}

TEST(Modes, EveryCombinationOfTheSpecTable) {
  // Spec 8.3, all eight camera x LiDAR x external-odometry combinations with `auto` (audit AUD-024).
  struct Row {
    bool cam, lidar, ext;
    OdometrySource odom;
    LoopSource loop;
    bool verify_with_lidar;
  };
  const Row rows[] = {
      {true, false, false, OdometrySource::Camera, LoopSource::Camera, false},
      {false, true, false, OdometrySource::Lidar, LoopSource::Lidar, false},
      {true, true, false, OdometrySource::Lidar, LoopSource::Camera, true},
      {false, false, false, OdometrySource::Camera, LoopSource::Camera, false},  // rejected by the validator
      // The external odometry is a prior of the camera tracking, never the odometry (ADR-0043).
      {true, false, true, OdometrySource::Camera, LoopSource::Camera, false},
      {false, true, true, OdometrySource::Lidar, LoopSource::Lidar, false},
      {true, true, true, OdometrySource::Lidar, LoopSource::Camera, true},
      {false, false, true, OdometrySource::Camera, LoopSource::Camera, false},  // rejected by the validator
  };
  for (const auto& r : rows) {
    Config c;
    c.sensors.camera.enabled = r.cam;
    c.sensors.lidar.enabled = r.lidar;
    c.sensors.external_odometry.enabled = r.ext;
    const auto m = resolve_modes(c);
    EXPECT_EQ(m.odometry, r.odom) << r.cam << r.lidar << r.ext;
    EXPECT_EQ(m.loop_closure, r.loop) << r.cam << r.lidar << r.ext;
    EXPECT_EQ(m.verify_loop_with_lidar, r.verify_with_lidar) << r.cam << r.lidar << r.ext;
  }
  // An explicit `loop_closure_from` wins and never adds the LiDAR verification.
  for (const auto& [from, loop] : {std::pair{LoopClosureFrom::Camera, LoopSource::Camera}, std::pair{LoopClosureFrom::Lidar, LoopSource::Lidar},
                                   std::pair{LoopClosureFrom::Both, LoopSource::Both}}) {
    Config c;
    c.sensors.camera.enabled = c.sensors.lidar.enabled = true;
    c.fusion.loop_closure_from = from;
    const auto m = resolve_modes(c);
    EXPECT_EQ(m.loop_closure, loop);
    EXPECT_FALSE(m.verify_loop_with_lidar);
  }
  // An explicit `odometry_from` wins over the external odometry.
  for (const auto& [from, odom] : {std::pair{OdometryFrom::Camera, OdometrySource::Camera}, std::pair{OdometryFrom::Lidar, OdometrySource::Lidar}}) {
    Config c;
    c.sensors.camera.enabled = c.sensors.lidar.enabled = c.sensors.external_odometry.enabled = true;
    c.fusion.odometry_from = from;
    EXPECT_EQ(resolve_modes(c).odometry, odom);
  }
}

TEST(Modes, ResolutionTable) {
  const auto cam = [] {
    Config c; c.sensors.camera.enabled = true; return c; };
  const auto lid = [] {
    Config c; c.sensors.lidar.enabled = true; return c; };

  auto m = resolve_modes(cam());
  EXPECT_EQ(m.odometry, OdometrySource::Camera);
  EXPECT_EQ(m.loop_closure, LoopSource::Camera);
  EXPECT_FALSE(m.verify_loop_with_lidar);

  m = resolve_modes(lid());
  EXPECT_EQ(m.odometry, OdometrySource::Lidar);
  EXPECT_EQ(m.loop_closure, LoopSource::Lidar);

  Config both = cam();
  both.sensors.lidar.enabled = true;
  m = resolve_modes(both);
  EXPECT_EQ(m.odometry, OdometrySource::Lidar);
  EXPECT_EQ(m.loop_closure, LoopSource::Camera);
  EXPECT_TRUE(m.verify_loop_with_lidar);

  Config ext = lid();
  ext.sensors.external_odometry.enabled = true;
  m = resolve_modes(ext);
  EXPECT_EQ(m.odometry, OdometrySource::Lidar);
  EXPECT_EQ(m.loop_closure, LoopSource::Lidar);

  Config forced = both;
  forced.fusion.odometry_from = OdometryFrom::Camera;
  forced.fusion.loop_closure_from = LoopClosureFrom::Both;
  m = resolve_modes(forced);
  EXPECT_EQ(m.odometry, OdometrySource::Camera);
  EXPECT_EQ(m.loop_closure, LoopSource::Both);
}

namespace {
bool error_at(const ConfigResult& r, const std::string& path) {
  return std::any_of(r.errors.begin(), r.errors.end(), [&](const Diagnostic& e) { return e.path == path; });
}
}  // namespace

// Audit A7-08: ranges that had no test of their own.
TEST(Rules, MoreNumericRanges) {
  const char* cam = "strataslam: {version: 1}\nsensors:\n  camera: {rgb_topic: a, depth_topic: b, info_topic: c, ";
  EXPECT_TRUE(error_at(load_profile_string(std::string(cam) + "depth_min: 0, depth_max: 6}\n"), "sensors.camera.depth_min"));
  EXPECT_TRUE(error_at(load_profile_string(std::string(cam) + "depth_min: -1, depth_max: 6}\n"), "sensors.camera.depth_min"));
  EXPECT_TRUE(error_at(load_profile_string("strataslam: {version: 1}\nsensors:\n  lidar: {min_range: -1}\n"), "sensors.lidar.min_range"));
  EXPECT_TRUE(error_at(load_profile_string(with("output: {keyframe_translation: 0}\n")), "output.keyframe_translation"));
  EXPECT_TRUE(error_at(load_profile_string(with("output: {keyframe_rotation_deg: -5}\n")), "output.keyframe_rotation_deg"));
  EXPECT_TRUE(error_at(load_profile_string(with("multifloor: {min_vertical_change: 0}\n")), "multifloor.min_vertical_change"));
  EXPECT_TRUE(error_at(load_profile_string(with("multifloor: {floor_height_prior: -3}\n")), "multifloor.floor_height_prior"));
  EXPECT_TRUE(error_at(load_profile_string(with("runtime: {threads: -2}\n")), "runtime.threads"));
  EXPECT_TRUE(error_at(load_profile_string("strataslam: {version: 1}\nsensors:\n  lidar: {enabled: true}\n  barometer: {settle_time_s: 0}\n"),
                       "sensors.barometer.settle_time_s"));
}

// Audit A4A5-07 / A7-04 / A7-05: a key written twice, a second YAML document and a top level that is not a map are
// errors with a path, not silently ignored.
TEST(Rules, DuplicateKeysExtraDocumentsAndTopLevelErrors) {
  auto r = load_profile_string(with("output: {voxel_size: 0.05}\noutput: {voxel_size: 0.5}\n"));
  EXPECT_TRUE(has_error(r, "config error: output: written twice (only one value can hold)")) << errors_of(r).front();
  r = load_profile_string(with("debug:\n  disable_floor_gating: true\nstrataslam: {version: 1}\n"));
  EXPECT_TRUE(has_error(r, "config error: strataslam: written twice (only one value can hold)"));
  r = load_profile_string(with("output: {voxel_size: 0.05, voxel_size: 0.1}\n"));
  EXPECT_TRUE(has_error(r, "config error: output.voxel_size: written twice (only one value can hold)"));
  r = load_profile_string(std::string(kBase) + "---\noutput: {voxel_size: 0.5}\n");
  ASSERT_EQ(r.errors.size(), 1u);
  EXPECT_EQ(r.errors[0].path, "(profile)");
  r = load_profile_string("- a\n- b\n");
  ASSERT_EQ(r.errors.size(), 1u) << errors_of(r).front();
  EXPECT_EQ(errors_of(r)[0], "config error: (profile): the profile must be a YAML map");
  r = load_profile_string("strataslam: [unclosed\n");
  ASSERT_FALSE(r.errors.empty());
  EXPECT_EQ(r.errors[0].path, "(profile)");
}

// Audit A7-06: colourising is on by default; a LiDAR-only profile gets the "no camera" warning only if it asked for it.
TEST(Rules, ColourWarningOnlyWhenAskedFor) {
  const auto has_warning = [](const ConfigResult& r) {
    return std::any_of(r.warnings.begin(), r.warnings.end(), [](const Diagnostic& w) { return w.path == "fusion.colorize_cloud"; });
  };
  EXPECT_FALSE(has_warning(load_profile_string(kBase)));
  EXPECT_TRUE(has_warning(load_profile_string(with("fusion: {colorize_cloud: true}\n"))));
}
