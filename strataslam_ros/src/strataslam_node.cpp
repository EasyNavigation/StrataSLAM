// strataslam_node: hosts the ROS-free core pipeline (SlamSystem) behind the topics, services and TF
// of the plan (section 11). All SLAM logic lives in strataslam_core; this file only converts
// messages.
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pwd.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <unistd.h>

#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <initializer_list>
#include <iomanip>
#include <map>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <optional>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/topic_metadata.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <set>
#include <sstream>
#include <std_srvs/srv/trigger.hpp>
#include <string>
#include <tf2_msgs/msg/tf_message.hpp>
#include <thread>

#include "strataslam/config/config_loader.hpp"
#include "strataslam/frontend/depth_utils.hpp"
#include "strataslam/frontend/feature_extractor.hpp"
#include "strataslam/fusion/fusion.hpp"
#include "strataslam/io/map_io.hpp"
#include "strataslam/pipeline/slam_system.hpp"
#include "strataslam_msgs/msg/camera_view.hpp"
#include "strataslam_msgs/msg/feature_frame.hpp"
#include "strataslam_msgs/msg/floor_state.hpp"
#include "strataslam_msgs/msg/instance.hpp"
#include "strataslam_msgs/srv/mark_floor_change.hpp"

namespace strataslam_ros {

using strataslam::Timestamp;
namespace sm = sensor_msgs::msg;

namespace {

/// Core logging interface wired to rclcpp.
class RclcppLogger final : public strataslam::Logger {
 public:
  explicit RclcppLogger(rclcpp::Logger l) : l_(l) {}
  void log(strataslam::LogLevel level, const std::string& m) override {
    switch (level) {
      case strataslam::LogLevel::Debug:
        RCLCPP_DEBUG(l_, "%s", m.c_str());
        break;
      case strataslam::LogLevel::Info:
        RCLCPP_INFO(l_, "%s", m.c_str());
        break;
      case strataslam::LogLevel::Warn:
        RCLCPP_WARN(l_, "%s", m.c_str());
        break;
      case strataslam::LogLevel::Error:
        RCLCPP_ERROR(l_, "%s", m.c_str());
        break;
    }
  }

 private:
  rclcpp::Logger l_;
};

Timestamp stamp_ns(const builtin_interfaces::msg::Time& t) {
  return static_cast<Timestamp>(t.sec) * 1000000000LL + t.nanosec;
}

gtsam::Pose3 to_pose(const geometry_msgs::msg::Transform& t) {
  return gtsam::Pose3(
      gtsam::Rot3::Quaternion(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z),
      gtsam::Point3(t.translation.x, t.translation.y, t.translation.z));
}

gtsam::Pose3 from_transform(const geometry_msgs::msg::Transform& t) {
  return gtsam::Pose3(
      gtsam::Rot3::Quaternion(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z),
      gtsam::Point3(t.translation.x, t.translation.y, t.translation.z));
}

geometry_msgs::msg::Transform to_transform(const gtsam::Pose3& p) {
  geometry_msgs::msg::Transform t;
  const auto q = p.rotation().toQuaternion();
  t.translation.x = p.x();
  t.translation.y = p.y();
  t.translation.z = p.z();
  t.rotation.w = q.w();
  t.rotation.x = q.x();
  t.rotation.y = q.y();
  t.rotation.z = q.z();
  return t;
}

std::string expand_home(std::string dir);

std::string expand_map_dir(std::string dir) {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  localtime_r(&now, &tm);
  char date[16], tim[16];
  std::strftime(date, sizeof date, "%Y%m%d", &tm);
  std::strftime(tim, sizeof tim, "%H%M%S", &tm);
  const auto replace = [&](const std::string& from, const std::string& to) {
    for (size_t p = dir.find(from); p != std::string::npos; p = dir.find(from, p + to.size()))
      dir.replace(p, from.size(), to);
  };
  replace("{date}", date);
  replace("{time}", tim);
  return expand_home(dir);
}

/// "~" and "~/..." are the user's home (HOME, else the password database); "~name/..." is that
/// user's home.
std::string expand_home(std::string dir) {
  // "~" and "~/..." are the user's home (HOME, else the password database); "~name/..." is that
  // user's home. Before, any leading "~" became HOME ("~root/maps" -> "/home/userroot/maps"; audit
  // A6-08).
  if (!dir.empty() && dir[0] == '~') {
    const size_t slash = dir.find('/');
    const std::string user =
        dir.substr(1, slash == std::string::npos ? std::string::npos : slash - 1);
    const std::string rest = slash == std::string::npos ? "" : dir.substr(slash);
    std::string home;
    if (user.empty()) {
      if (const char* h = std::getenv("HOME"); h && *h)
        home = h;
      else if (const passwd* pw = getpwuid(getuid()))
        home = pw->pw_dir;
    } else if (const passwd* pw = getpwnam(user.c_str())) {
      home = pw->pw_dir;
    }
    if (!home.empty()) dir = home + rest;
  }
  return dir;
}

/// Reads one scalar field of a PointCloud2 point as double.
/// Bytes a supported PointField occupies (0 = unsupported type).
size_t field_size(const sm::PointField& f) {
  switch (f.datatype) {
    case sm::PointField::FLOAT32:
    case sm::PointField::UINT32:
    case sm::PointField::INT32:
      return 4;
    case sm::PointField::FLOAT64:
      return 8;
    default:
      return 0;
  }
}

/// Why a PointCloud2 cannot be read safely, or an empty string. The layout comes from the message
/// itself (a driver, a bag), so it is checked before any byte is read: every field must lie inside
/// a point, and the buffer must hold width x height points of point_step bytes.
std::string cloud_layout_error(const sm::PointCloud2& m,
                               std::initializer_list<const sm::PointField*> fields) {
  for (const auto* f : fields) {
    if (!f) continue;
    const size_t size = field_size(*f);
    if (size == 0)
      return "field '" + f->name + "' has an unsupported datatype " + std::to_string(f->datatype);
    if (f->count != 1)
      return "field '" + f->name + "' has count " + std::to_string(f->count) + " (expected 1)";
    if (static_cast<size_t>(f->offset) + size > m.point_step)
      return "field '" + f->name + "' (offset " + std::to_string(f->offset) +
             ") does not fit in point_step " + std::to_string(m.point_step);
  }
  const uint64_t n = static_cast<uint64_t>(m.width) * m.height;
  if (n > 0 && m.point_step == 0) return "point_step is 0";
  if (m.height > 1 && m.row_step < static_cast<uint64_t>(m.width) * m.point_step)
    return "row_step " + std::to_string(m.row_step) + " < width x point_step";
  const uint64_t row = m.height > 1 ? m.row_step : static_cast<uint64_t>(m.width) * m.point_step;
  const uint64_t needed =
      n == 0 ? 0 : (m.height - 1) * row + static_cast<uint64_t>(m.width) * m.point_step;
  if (m.data.size() < needed)
    return "data holds " + std::to_string(m.data.size()) +
           " bytes, width x height x point_step needs " + std::to_string(needed);
  return {};
}

double read_field(const uint8_t* p, const sm::PointField& f) {
  switch (f.datatype) {
    case sm::PointField::FLOAT32: {
      float v;
      std::memcpy(&v, p + f.offset, 4);
      return v;
    }
    case sm::PointField::FLOAT64: {
      double v;
      std::memcpy(&v, p + f.offset, 8);
      return v;
    }
    case sm::PointField::UINT32: {
      uint32_t v;
      std::memcpy(&v, p + f.offset, 4);
      return v;
    }
    case sm::PointField::INT32: {
      int32_t v;
      std::memcpy(&v, p + f.offset, 4);
      return v;
    }
    default:
      return 0.0;
  }
}

}  // namespace

class StrataslamNode : public rclcpp::Node {
 public:
  explicit StrataslamNode(const rclcpp::NodeOptions& options)
      : rclcpp::Node("strataslam_node", options) {
    profile_path_ = declare_parameter<std::string>("profile", "");
    vocabulary_path_ = declare_parameter<std::string>("vocabulary", "");
    features_topic_ = declare_parameter<std::string>("features_topic", "");
    cloud_period_s_ = declare_parameter<double>("cloud_period_s", 5.0);
    bag_path_ = declare_parameter<std::string>("bag", "");
    wait_for_viewer_s_ = declare_parameter<double>("wait_for_viewer_s", 0.0);

    try {
      load_profile();
    } catch (const std::exception& e) {
      // A node that cannot start must not leave its container (and `ros2 launch`) waiting forever.
      // Shutting down here, inside the container's load request, could end the process before it
      // answers the launch, which then waits for that answer: the node is created, does nothing,
      // and ends the process just after.
      RCLCPP_FATAL(get_logger(), "cannot start: %s", e.what());
      startup_failed_timer_ =
          create_wall_timer(std::chrono::milliseconds(500), [] { rclcpp::shutdown(); });
      return;
    }
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    if (bag_path_.empty()) tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("strataslam/odometry",
                                                          rclcpp::QoS(10).reliable());
    path_pub_ =
        create_publisher<nav_msgs::msg::Path>("strataslam/path", rclcpp::QoS(10).reliable());
    state_pub_ = create_publisher<strataslam_msgs::msg::FloorState>("strataslam/floor_state",
                                                                    rclcpp::QoS(10).reliable());
    cloud_pub_ = create_publisher<sm::PointCloud2>("strataslam/floor_cloud",
                                                   rclcpp::QoS(1).transient_local().reliable());
    // What the sensors see right now, in the odometry frame (for viewers): latest only, best
    // effort.
    scan_pub_ =
        create_publisher<sm::PointCloud2>("strataslam/scan", rclcpp::SensorDataQoS().keep_last(1));
    depth_cloud_pub_ = create_publisher<sm::PointCloud2>("strataslam/depth_cloud",
                                                         rclcpp::SensorDataQoS().keep_last(1));
    camera_view_pub_ = create_publisher<strataslam_msgs::msg::CameraView>(
        "strataslam/camera_view", rclcpp::SensorDataQoS().keep_last(1));

    if (bag_path_.empty()) create_subscriptions();
    save_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/save_map", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
          if (!guard("save_map", [&] { on_save_map(res); })) {
            res->success = false;
            res->message = "internal error (see the node log)";
          }
        });
    mark_srv_ = create_service<strataslam_msgs::srv::MarkFloorChange>(
        "~/mark_floor_change",
        [this](const std::shared_ptr<strataslam_msgs::srv::MarkFloorChange::Request> req,
               std::shared_ptr<strataslam_msgs::srv::MarkFloorChange::Response> res) {
          if (!guard("mark_floor_change", [&] { on_mark_floor_change(*req, *res); })) {
            res->success = false;
            res->message = "internal error (see the node log)";
          }
        });
    session_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/new_session", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                                std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
          const bool ok = guard("new_session", [&] {
            if (slam_) {
              slam_->new_session();
              res->success = true;
              res->message = "new session";
            } else {
              res->success = false;
              res->message = "the pipeline has not started yet";
            }
          });
          if (!ok) {
            res->success = false;
            res->message = "internal error (see the node log)";
          }
        });
    cloud_timer_ = create_wall_timer(std::chrono::duration<double>(cloud_period_s_),
                                     [this] { guard("publish_cloud", [&] { publish_cloud(); }); });
    RCLCPP_INFO(get_logger(), "%s started with profile %s", get_fully_qualified_name(),
                profile_path_.c_str());
    if (!bag_path_.empty()) {
      // Offline mode: every message of the bag is processed, in order, one after the other (no
      // subscription queues, so no scan is dropped and the result does not depend on the timing).
      offline_timer_ = create_wall_timer(std::chrono::milliseconds(0), [this] {
        offline_timer_->cancel();
        check_single_instance(false);
        guard("offline bag", [&] { run_bag(); });
        rclcpp::shutdown();
      });
    } else {
      announce_instance();
    }
  }

  /// Ctrl+C online: the map of the run is saved as with a bag (`bag:=` saved it, a live run lost it
  /// silently; audit AUD-092), unless nothing new was mapped since the last ~/save_map.
  ~StrataslamNode() override {
    if (!bag_path_.empty() || !slam_) return;
    const size_t kfs = slam_->keyframes().size();
    if (kfs == 0 || kfs == keyframes_at_last_save_) return;
    RCLCPP_WARN(get_logger(), "stopping with %zu keyframe(s) not saved: saving the map",
                kfs - keyframes_at_last_save_);
    try {
      auto res = std::make_shared<std_srvs::srv::Trigger::Response>();
      on_save_map(res);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "could not save the map: %s", e.what());
    }
  }

 private:
  /// Runs the body of a ROS callback. An exception from the core or from a malformed message is
  /// logged (throttled) and the message dropped: nothing unwinds into the executor and stops the
  /// node.
  template <class F>
  bool guard(const char* where, F&& body) {
    try {
      body();
      return true;
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "%s: %s (message dropped)", where,
                            e.what());
    } catch (...) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "%s: unknown exception (message dropped)", where);
    }
    return false;
  }

  std::string duplicate_hint() const {
    const char* domain = std::getenv("ROS_DOMAIN_ID");
    return std::string("another ") + get_fully_qualified_name() + " is running in ROS_DOMAIN_ID " +
           (domain ? domain : "0") +
           ": start this one with namespace:=<name> (or another ROS_DOMAIN_ID)";
  }

  /// Online, a second node with the same name would share the topics and answer the same service
  /// calls (a save_map would write two maps). Every node announces when it started on ~/instance
  /// (transient local); a node that hears of an older one with its name refuses to run, whenever
  /// that happens. The older one never stops because of a newcomer.
  void announce_instance() {
    instance_.started = now();
    instance_.id = std::random_device{}() * 4294967296ULL + std::random_device{}();
    const auto qos = rclcpp::QoS(10).reliable().transient_local();
    instance_pub_ = create_publisher<strataslam_msgs::msg::Instance>("~/instance", qos);
    instance_sub_ = create_subscription<strataslam_msgs::msg::Instance>(
        "~/instance", qos, [this](strataslam_msgs::msg::Instance::ConstSharedPtr m) {
          if (m->id == instance_.id || refused_) return;
          const auto older = [](const strataslam_msgs::msg::Instance& a,
                                const strataslam_msgs::msg::Instance& b) {
            if (a.started.sec != b.started.sec) return a.started.sec < b.started.sec;
            if (a.started.nanosec != b.started.nanosec)
              return a.started.nanosec < b.started.nanosec;
            return a.id < b.id;
          };
          if (!older(*m, instance_)) return;
          refused_ = true;
          RCLCPP_FATAL(get_logger(), "%s", duplicate_hint().c_str());
          rclcpp::shutdown();
        });
    instance_pub_->publish(instance_);
  }

  /// Offline (a bag) nothing is called and the map is saved by the node itself: another instance
  /// with the same name only shares the topics, which is worth a warning, not a refusal.
  void check_single_instance(bool /*fatal*/) {
    const std::string me = get_fully_qualified_name();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
    do {
      size_t same = 0;
      for (const auto& n : get_node_names()) same += n == me;
      if (same > 1) {
        RCLCPP_WARN(get_logger(), "%s; both publish on the same topics", duplicate_hint().c_str());
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } while (std::chrono::steady_clock::now() < deadline);
  }

  // ---------------- profile and pipeline ----------------

  void load_profile() {
    if (profile_path_.empty()) throw std::runtime_error("parameter 'profile' is required");
    const auto r = strataslam::load_profile_file(profile_path_);
    for (const auto& w : r.warnings)
      RCLCPP_WARN(get_logger(), "config warning: %s: %s", w.path.c_str(), w.message.c_str());
    if (!r.ok()) {
      for (const auto& e : r.errors) RCLCPP_ERROR(get_logger(), "%s", e.format().c_str());
      throw std::runtime_error("invalid profile " + profile_path_ + " (" +
                               std::to_string(r.errors.size()) + " error(s))");
    }
    cfg_ = *r.config;
    std::ifstream in(profile_path_);
    profile_text_.assign(std::istreambuf_iterator<char>(in), {});
    // The launch's `vocabulary:=` overrides the profile's `fusion.vocabulary`.
    const bool from_launch = !vocabulary_path_.empty();
    std::string vocabulary = from_launch ? vocabulary_path_ : cfg_.fusion.vocabulary;
    if (vocabulary == "default") {  // the one shipped with the package (ADR-0054)
      vocabulary = ament_index_cpp::get_package_share_directory("strataslam_ros") +
                   "/vocabularies/tum_rgbd.bin";
    }
    vocabulary = expand_home(vocabulary);
    if (!vocabulary.empty()) {
      auto v = strataslam::BowVocabulary::load(vocabulary);
      if (!v) throw std::runtime_error(v.error().message);
      tuning_.vocabulary = std::make_shared<const strataslam::BowVocabulary>(std::move(v.value()));
      RCLCPP_INFO(get_logger(), "visual vocabulary: %s (%zu words, from the %s)",
                  vocabulary.c_str(), tuning_.vocabulary->words(),
                  from_launch ? "launch" : "profile");
    } else if (cfg_.sensors.camera.enabled) {
      RCLCPP_WARN(get_logger(),
                  "no vocabulary: visual loop closure is disabled (train one with `strataslam "
                  "vocab train`, then "
                  "set fusion.vocabulary in the profile or pass vocabulary:=)");
    }
    if (cfg_.sensors.camera.features == strataslam::FeatureType::Xfeat) {
      RCLCPP_WARN(get_logger(), "XFeat is not built in this version; using ORB");
    }
    orb_ = strataslam::make_orb_extractor(cfg_.sensors.camera.max_features);
  }

  bool lookup(const std::string& frame, gtsam::Pose3* out) {
    const std::string base = cfg_.robot.base_frame;
    if (frame.empty() || frame == base) {
      *out = gtsam::Pose3();
      return true;
    }
    try {
      const auto t = tf_buffer_->lookupTransform(base, frame, tf2::TimePointZero);
      *out = to_pose(t.transform);
      return true;
    } catch (const tf2::TransformException& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for TF %s -> %s: %s",
                           base.c_str(), frame.c_str(), e.what());
      return false;
    }
  }

  /// The pipeline needs the sensor extrinsics at construction: it starts with the first sensor data
  /// for which every enabled sensor's transform is available.
  bool ensure_system(const std::string& camera_frame, const std::string& lidar_frame,
                     const std::string& imu_frame) {
    if (slam_) return true;
    if (cfg_.sensors.camera.enabled && camera_frame.empty()) return false;
    if (cfg_.sensors.lidar.enabled && lidar_frame.empty()) return false;
    strataslam::Extrinsics ext;
    if (cfg_.sensors.camera.enabled && !lookup(camera_frame, &ext.T_base_camera)) return false;
    if (cfg_.sensors.lidar.enabled && !lookup(lidar_frame, &ext.T_base_lidar)) return false;
    if (cfg_.sensors.imu.enabled && !lookup(imu_frame, &ext.T_base_imu)) return false;
    camera_frame_ = camera_frame;
    lidar_frame_ = lidar_frame;
    imu_frame_ = imu_frame;
    ext_ = ext;
    slam_ = std::make_unique<strataslam::SlamSystem>(cfg_, ext, tuning_,
                                                     std::make_shared<RclcppLogger>(get_logger()));
    for (const auto& m : imu_backlog_)
      slam_->add_imu(m);  // data that arrived while the TF tree filled in
    if (last_baro_reference_)
      slam_->add_baro_reference(*last_baro_reference_);  // before the rover's first sample
    for (const auto& m : baro_backlog_) slam_->add_baro(m);
    imu_backlog_.clear();
    baro_backlog_.clear();
    // Which sensor tracks the motion and which closes loops, with `auto` resolved: the profile may
    // not say.
    const auto modes = strataslam::resolve_modes(cfg_);
    const auto odom = modes.odometry == strataslam::OdometrySource::Camera  ? "camera"
                      : modes.odometry == strataslam::OdometrySource::Lidar ? "lidar"
                                                                            : "external";
    const auto loops = modes.loop_closure == strataslam::LoopSource::Camera  ? "camera"
                       : modes.loop_closure == strataslam::LoopSource::Lidar ? "lidar"
                                                                             : "camera and lidar";
    RCLCPP_INFO(
        get_logger(), "pipeline started: odometry from the %s%s, loop closures from the %s%s", odom,
        cfg_.fusion.odometry_from == strataslam::OdometryFrom::Auto ? " (auto)" : "", loops,
        cfg_.fusion.loop_closure_from == strataslam::LoopClosureFrom::Auto ? " (auto)" : "");
    visual_loops_off_ = modes.loop_closure != strataslam::LoopSource::Lidar && !tuning_.vocabulary;
    return true;
  }

  // ---------------- subscriptions ----------------

  void create_subscriptions() {
    const auto qos = rclcpp::SensorDataQoS();
    const auto& s = cfg_.sensors;
    if (s.camera.enabled) {
      depth_sub_.subscribe(this, s.camera.depth_topic, qos.get_rmw_qos_profile());
      info_sub_.subscribe(this, s.camera.info_topic, qos.get_rmw_qos_profile());
      if (!features_topic_.empty()) {
        feat_sub_.subscribe(this, features_topic_, qos.get_rmw_qos_profile());
        feat_sync_ = std::make_unique<message_filters::Synchronizer<FeatPolicy>>(
            FeatPolicy(10), depth_sub_, feat_sub_, info_sub_);
        feat_sync_->registerCallback(std::bind(&StrataslamNode::on_features_guarded, this,
                                               std::placeholders::_1, std::placeholders::_2,
                                               std::placeholders::_3));
      } else {
        rgb_sub_.subscribe(this, s.camera.rgb_topic, qos.get_rmw_qos_profile());
        rgb_sync_ = std::make_unique<message_filters::Synchronizer<RgbPolicy>>(
            RgbPolicy(10), depth_sub_, rgb_sub_, info_sub_);
        rgb_sync_->registerCallback(std::bind(&StrataslamNode::on_rgb_guarded, this,
                                              std::placeholders::_1, std::placeholders::_2,
                                              std::placeholders::_3));
      }
    }
    if (s.lidar.enabled)
      lidar_sub_ = create_subscription<sm::PointCloud2>(
          s.lidar.topic, qos,
          [this](sm::PointCloud2::ConstSharedPtr m) { guard("lidar", [&] { on_lidar(*m); }); });
    if (s.imu.enabled)
      imu_sub_ = create_subscription<sm::Imu>(s.imu.topic, qos, [this](sm::Imu::ConstSharedPtr m) {
        guard("imu", [&] { on_imu(*m); });
      });
    if (s.barometer.enabled)
      baro_sub_ = create_subscription<sm::FluidPressure>(
          s.barometer.topic, qos, [this](sm::FluidPressure::ConstSharedPtr m) {
            guard("barometer", [&] { on_baro(*m); });
          });
    if (s.barometer.enabled && !s.barometer.reference_topic.empty())
      baro_ref_sub_ = create_subscription<sm::FluidPressure>(
          s.barometer.reference_topic, qos, [this](sm::FluidPressure::ConstSharedPtr m) {
            guard("barometer reference", [&] { on_baro_reference(*m); });
          });
    if (s.external_odometry.enabled && !s.external_odometry.topic.empty())
      ext_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
          s.external_odometry.topic, qos, [this](nav_msgs::msg::Odometry::ConstSharedPtr m) {
            guard("external odometry", [&] { on_ext_odom(*m); });
          });
    if (s.external_odometry.enabled && !s.external_odometry.frame.empty())
      ext_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
          "/tf", rclcpp::QoS(100), [this](tf2_msgs::msg::TFMessage::ConstSharedPtr m) {
            guard("external odometry", [&] { on_ext_tf(*m); });
          });
  }

  // message_filters takes a bound member function, not a lambda.
  void on_features_guarded(const sm::Image::ConstSharedPtr& depth,
                           const strataslam_msgs::msg::FeatureFrame::ConstSharedPtr& feat,
                           const sm::CameraInfo::ConstSharedPtr& info) {
    queue_camera(stamp_ns(depth->header.stamp), [this, depth, feat, info] {
      guard("camera features", [&] { on_features(depth, feat, info); });
    });
  }
  void on_rgb_guarded(const sm::Image::ConstSharedPtr& depth, const sm::Image::ConstSharedPtr& rgb,
                      const sm::CameraInfo::ConstSharedPtr& info) {
    queue_camera(stamp_ns(depth->header.stamp),
                 [this, depth, rgb, info] { guard("camera", [&] { on_rgb(depth, rgb, info); }); });
  }

  /// With an external odometry, a camera frame waits until it covers the frame's stamp (wheel
  /// odometry trails the camera by a fraction of a second), at most kCameraWait behind the newest
  /// frame. In arrival order, so a bag gives the same result run after run. `all`: process every
  /// frame waiting (end of a bag).
  void queue_camera(Timestamp stamp, std::function<void()> process) {
    camera_queue_.emplace_back(stamp, std::move(process));
    newest_camera_ = std::max(newest_camera_, stamp);
    drain_camera(false);
  }
  void drain_camera(bool all) {
    constexpr Timestamp kCameraWait = 500'000'000;  // ns
    const bool wait = cfg_.sensors.external_odometry.enabled && !all;
    while (!camera_queue_.empty()) {
      const Timestamp stamp = camera_queue_.front().first;
      if (wait && stamp > last_ext_stamp_ && stamp + kCameraWait > newest_camera_) break;
      if (cfg_.sensors.external_odometry.enabled) warn_if_no_ext_odometry(stamp);
      const auto process = std::move(camera_queue_.front().second);
      camera_queue_.pop_front();
      process();
    }
  }

  void on_ext_odom(const nav_msgs::msg::Odometry& m) {
    strataslam::OdomMeasurement o;
    o.stamp = stamp_ns(m.header.stamp);
    const auto& p = m.pose.pose;
    o.pose = gtsam::Pose3(
        gtsam::Rot3::Quaternion(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z),
        gtsam::Point3(p.position.x, p.position.y, p.position.z));
    if (slam_) slam_->add_odom(o);
    last_ext_stamp_ = std::max(last_ext_stamp_, o.stamp);
    drain_camera(false);
  }

  /// The external odometry from TF: the transforms `external_odometry.frame` -> robot.base_frame
  /// (wheels). The odometry may move a frame above the base instead (odom -> base_footprint, with
  /// base_footprint -> base_link fixed in the URDF, while the profile names base_link): the moving
  /// transform is composed with the fixed chain below it. A transform from the odometry frame to a
  /// frame that is not an ancestor of the base is not the robot's motion and is skipped.
  void on_ext_tf(const tf2_msgs::msg::TFMessage& m) {
    const std::string& odom = cfg_.sensors.external_odometry.frame;
    const std::string& base = cfg_.robot.base_frame;
    for (const auto& t : m.transforms) {
      if (t.header.frame_id != odom) continue;
      gtsam::Pose3 T_odom_base = to_pose(t.transform);
      if (t.child_frame_id != base) {
        const auto below = fixed_chain_to_base(t.child_frame_id);
        if (!below) continue;
        T_odom_base = T_odom_base * *below;
      }
      if (!ext_tf_reported_) {
        ext_tf_reported_ = true;
        const std::string chain = t.child_frame_id == base ? ""
                                                           : " composed with the fixed " +
                                                                 t.child_frame_id + " -> " + base;
        RCLCPP_INFO(get_logger(), "wheel odometry: TF %s -> %s%s", odom.c_str(),
                    t.child_frame_id.c_str(), chain.c_str());
      }
      if (slam_) slam_->add_odom({stamp_ns(t.header.stamp), T_odom_base});
      last_ext_stamp_ = std::max(last_ext_stamp_, stamp_ns(t.header.stamp));
    }
    drain_camera(false);
  }

  /// `frame` -> robot.base_frame when `frame` is an ancestor of the base below the odometry frame
  /// (the base's parents are followed up to it); nullopt otherwise, or while TF does not have the
  /// chain yet (its static transforms arrive after the first odometry).
  std::optional<gtsam::Pose3> fixed_chain_to_base(const std::string& frame) {
    const std::string& base = cfg_.robot.base_frame;
    const std::string& odom = cfg_.sensors.external_odometry.frame;
    std::string f = base, parent;
    bool ancestor = false;
    for (int depth = 0; depth < 32 && tf_buffer_->_getParent(f, tf2::TimePointZero, parent);
         ++depth) {
      if (parent == frame) {
        ancestor = true;
        break;
      }
      if (parent == odom) break;
      f = parent;
    }
    if (!ancestor) return std::nullopt;
    try {
      return to_pose(tf_buffer_->lookupTransform(frame, base, tf2::TimePointZero).transform);
    } catch (const tf2::TransformException&) {
      return std::nullopt;
    }
  }

  /// Frames that went on without the external odometry because none had arrived: after 2 s of them
  /// (camera time) the profile and the robot's TF disagree, which is said once with what TF has.
  void warn_if_no_ext_odometry(Timestamp stamp) {
    constexpr Timestamp kGrace = 2'000'000'000;  // ns
    if (ext_odom_warned_ || last_ext_stamp_ != 0) return;
    if (!first_unodometered_) first_unodometered_ = stamp;
    if (stamp - *first_unodometered_ < kGrace) return;
    ext_odom_warned_ = true;
    const auto& ext = cfg_.sensors.external_odometry;
    if (!ext.topic.empty()) {
      RCLCPP_WARN(get_logger(),
                  "external_odometry: nothing on %s after 2 s of camera frames: the camera "
                  "tracks without the wheel odometry (check the topic)",
                  ext.topic.c_str());
      return;
    }
    std::string children;
    for (const auto& name : tf_buffer_->getAllFrameNames()) {
      std::string parent;
      if (tf_buffer_->_getParent(name, tf2::TimePointZero, parent) && parent == ext.frame)
        children += (children.empty() ? "" : ", ") + name;
    }
    RCLCPP_WARN(get_logger(),
                "external_odometry: no TF %s -> %s after 2 s of camera frames: the camera tracks "
                "without the wheel odometry. TF has %s -> {%s}: set robot.base_frame to the frame "
                "the odometry moves, or one fixed below it",
                ext.frame.c_str(), cfg_.robot.base_frame.c_str(), ext.frame.c_str(),
                children.empty() ? "nothing" : children.c_str());
  }

  /// A sensor topic of the profile that the bag does not have is a mistake (a namespace, a typo):
  /// the run would end with an empty map. Each one is reported with the bag's topics of that type.
  void warn_missing_topics(const std::vector<rosbag2_storage::TopicMetadata>& topics) {
    const auto& s = cfg_.sensors;
    std::vector<std::pair<std::string, std::string>> wanted;  // topic, message type
    if (s.lidar.enabled) wanted.push_back({s.lidar.topic, "sensor_msgs/msg/PointCloud2"});
    if (s.imu.enabled) wanted.push_back({s.imu.topic, "sensor_msgs/msg/Imu"});
    if (s.barometer.enabled) wanted.push_back({s.barometer.topic, "sensor_msgs/msg/FluidPressure"});
    if (s.external_odometry.enabled && !s.external_odometry.topic.empty())
      wanted.push_back({s.external_odometry.topic, "nav_msgs/msg/Odometry"});
    if (s.camera.enabled) {
      if (features_topic_.empty())
        wanted.push_back({s.camera.rgb_topic, "sensor_msgs/msg/Image"});
      else
        wanted.push_back({features_topic_, "strataslam_msgs/msg/FeatureFrame"});
      wanted.push_back({s.camera.depth_topic, "sensor_msgs/msg/Image"});
      wanted.push_back({s.camera.info_topic, "sensor_msgs/msg/CameraInfo"});
    }
    for (const auto& [topic, type] : wanted) {
      std::string same_type;
      bool found = false;
      for (const auto& t : topics) {
        if (t.name == topic) found = true;
        if (t.type == type) same_type += (same_type.empty() ? "" : ", ") + t.name;
      }
      if (!found)
        RCLCPP_ERROR(get_logger(), "the bag has no topic %s (%s), so that sensor gets no data; %s",
                     topic.c_str(), type.c_str(),
                     same_type.empty() ? "the bag has no topic of that type"
                                       : ("topics of that type in the bag: " + same_type).c_str());
    }
  }

  /// Reads the bag at `bag_path_` and feeds the pipeline synchronously, then saves the map.
  void run_bag() {
    // With a viewer (launch argument viewer:=true) the bag starts once it listens, so it sees every
    // pose.
    if (wait_for_viewer_s_ > 0.0) {
      RCLCPP_INFO(get_logger(), "waiting up to %.0f s for a viewer on /strataslam/odometry",
                  wait_for_viewer_s_);
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::duration<double>(wait_for_viewer_s_);
      while (rclcpp::ok() && odom_pub_->get_subscription_count() == 0 &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
    rosbag2_cpp::Reader reader;
    reader.open(bag_path_);
    const auto& s = cfg_.sensors;
    warn_missing_topics(reader.get_all_topics_and_types());
    rclcpp::Serialization<sm::PointCloud2> ser_cloud;
    rclcpp::Serialization<sm::Imu> ser_imu;
    rclcpp::Serialization<sm::FluidPressure> ser_baro;
    rclcpp::Serialization<nav_msgs::msg::Odometry> ser_odom;
    rclcpp::Serialization<tf2_msgs::msg::TFMessage> ser_tf;
    rclcpp::Serialization<sm::Image> ser_image;
    rclcpp::Serialization<sm::CameraInfo> ser_info;
    rclcpp::Serialization<strataslam_msgs::msg::FeatureFrame> ser_feat;
    size_t n_lidar = 0, n_total = 0, n_camera = 0;
    // Camera: each colour image (or feature frame) is paired with the depth image of the closest
    // stamp within kPairTolerance, and with the latest camera_info; a pair is processed as soon as
    // both are read, in bag order. Deterministic, unlike the time-window policy of message_filters.
    constexpr Timestamp kPairTolerance =
        25'000'000;  // ns: below half the period of a 15-30 Hz camera
    const bool use_features = !features_topic_.empty();
    std::deque<sm::Image::SharedPtr> depth_q, rgb_q;
    std::deque<strataslam_msgs::msg::FeatureFrame::SharedPtr> feat_q;
    sm::CameraInfo::SharedPtr info;
    const auto pair_camera = [&]() {
      while (!depth_q.empty() && (use_features ? !feat_q.empty() : !rgb_q.empty())) {
        const Timestamp tc = use_features ? stamp_ns(feat_q.front()->header.stamp)
                                          : stamp_ns(rgb_q.front()->header.stamp);
        const Timestamp td = stamp_ns(depth_q.front()->header.stamp);
        if (std::llabs(tc - td) <= kPairTolerance) {
          if (info) {
            if (use_features)
              on_features_guarded(depth_q.front(), feat_q.front(), info);
            else
              on_rgb_guarded(depth_q.front(), rgb_q.front(), info);
            ++n_camera;
          }
          depth_q.pop_front();
          if (use_features)
            feat_q.pop_front();
          else
            rgb_q.pop_front();
        } else if (td < tc) {
          depth_q.pop_front();  // no colour image this close: the older one is dropped
        } else if (use_features) {
          feat_q.pop_front();
        } else {
          rgb_q.pop_front();
        }
      }
    };
    // The cloud timer cannot fire while this callback runs: the floor cloud is published from the
    // loop, at the same wall-clock period.
    const auto cloud_period = std::chrono::duration<double>(cloud_period_s_);
    auto next_cloud = std::chrono::steady_clock::now() + cloud_period;
    // Ctrl+C (SIGINT, or the launch's SIGTERM) stops reading: the map of what was processed is
    // saved.
    bool interrupted = false;
    while (reader.has_next()) {
      if (!rclcpp::ok()) {
        interrupted = true;
        break;
      }
      if (std::chrono::steady_clock::now() >= next_cloud) {
        guard("publish_cloud", [&] { publish_cloud(); });
        next_cloud = std::chrono::steady_clock::now() + cloud_period;
      }
      const auto bm = reader.read_next();
      ++n_total;
      rclcpp::SerializedMessage raw(*bm->serialized_data);
      const std::string& topic = bm->topic_name;
      guard(topic.c_str(), [&] {
        if (topic == "/tf" || topic == "/tf_static") {
          tf2_msgs::msg::TFMessage m;
          ser_tf.deserialize_message(&raw, &m);
          for (const auto& tr : m.transforms)
            tf_buffer_->setTransform(tr, "bag", topic == "/tf_static");
          if (s.external_odometry.enabled && !s.external_odometry.frame.empty()) on_ext_tf(m);
        } else if (s.lidar.enabled && topic == s.lidar.topic) {
          sm::PointCloud2 m;
          ser_cloud.deserialize_message(&raw, &m);
          ++n_lidar;
          on_lidar(m);
        } else if (s.imu.enabled && topic == s.imu.topic) {
          sm::Imu m;
          ser_imu.deserialize_message(&raw, &m);
          on_imu(m);
        } else if (s.barometer.enabled && topic == s.barometer.topic) {
          sm::FluidPressure m;
          ser_baro.deserialize_message(&raw, &m);
          on_baro(m);
        } else if (s.barometer.enabled && !s.barometer.reference_topic.empty() &&
                   topic == s.barometer.reference_topic) {
          sm::FluidPressure m;
          ser_baro.deserialize_message(&raw, &m);
          on_baro_reference(m);
        } else if (s.external_odometry.enabled && topic == s.external_odometry.topic) {
          nav_msgs::msg::Odometry m;
          ser_odom.deserialize_message(&raw, &m);
          on_ext_odom(m);
        } else if (s.camera.enabled && topic == s.camera.info_topic) {
          auto m = std::make_shared<sm::CameraInfo>();
          ser_info.deserialize_message(&raw, m.get());
          info = m;
        } else if (s.camera.enabled && topic == s.camera.depth_topic) {
          auto m = std::make_shared<sm::Image>();
          ser_image.deserialize_message(&raw, m.get());
          depth_q.push_back(m);
          pair_camera();
        } else if (s.camera.enabled && !use_features && topic == s.camera.rgb_topic) {
          auto m = std::make_shared<sm::Image>();
          ser_image.deserialize_message(&raw, m.get());
          rgb_q.push_back(m);
          pair_camera();
        } else if (s.camera.enabled && use_features && topic == features_topic_) {
          auto m = std::make_shared<strataslam_msgs::msg::FeatureFrame>();
          ser_feat.deserialize_message(&raw, m.get());
          feat_q.push_back(m);
          pair_camera();
        }
      });
    }
    drain_camera(true);
    if (interrupted) {
      RCLCPP_WARN(get_logger(),
                  "bag interrupted after %zu of %zu messages: saving the map of what was processed",
                  n_total, static_cast<size_t>(reader.get_metadata().message_count));
    } else {  // with the context shut down there is nobody to publish to
      guard("same level", [&] { check_same_level(true); });
    }
    RCLCPP_INFO(get_logger(),
                "bag %s: %zu messages, %zu LiDAR scans and %zu camera frames processed",
                bag_path_.c_str(), n_total, n_lidar, n_camera);
    auto res = std::make_shared<std_srvs::srv::Trigger::Response>();
    on_save_map(res);
    if (!res->success) RCLCPP_ERROR(get_logger(), "no map saved: %s", res->message.c_str());
    if (!interrupted && slam_) guard("final view", [&] { publish_final_view(); });
  }

  /// The view of the saved map: the last loop closures and the final optimization of `save_map`
  /// move the path and the cloud, and the node shuts down right after a bag, so the path, the
  /// floor -> odom transform and the cloud are published once more (the cloud past its rate limit)
  /// and delivered before it goes.
  void publish_final_view() {
    const auto path = slam_->floor_path(last_floor_);
    const Timestamp t =
        path.empty() ? static_cast<Timestamp>(now().nanoseconds()) : path.back().first;
    publish_state(t);
    publish_path(rclcpp::Time(t, RCL_ROS_TIME));
    publish_cloud(true);
    const auto timeout = std::chrono::seconds(10);  // a cloud of 100 MB to a viewer on this machine
    if (path_pub_->get_subscription_count() > 0) path_pub_->wait_for_all_acked(timeout);
    if (cloud_pub_->get_subscription_count() > 0) cloud_pub_->wait_for_all_acked(timeout);
  }

  void on_imu(const sm::Imu& m) {
    imu_frame_pending_ = m.header.frame_id;
    strataslam::ImuSample s;
    s.stamp = stamp_ns(m.header.stamp);
    s.accel =
        Eigen::Vector3d(m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z);
    s.gyro = Eigen::Vector3d(m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z);
    if (slam_) {
      slam_->add_imu(s);
    } else {
      imu_backlog_.push_back(s);
      if (imu_backlog_.size() > 1200) imu_backlog_.pop_front();  // ~6 s at 200 Hz
    }
  }

  /// The fixed base station: kept until the pipeline starts, then given to it with every sample.
  void on_baro_reference(const sm::FluidPressure& m) {
    const strataslam::BaroSample b{stamp_ns(m.header.stamp), m.fluid_pressure};
    last_baro_reference_ = b;
    if (slam_) slam_->add_baro_reference(b);
  }

  void on_baro(const sm::FluidPressure& m) {
    const strataslam::BaroSample b{stamp_ns(m.header.stamp), m.fluid_pressure};
    if (slam_) {
      slam_->add_baro(b);
    } else {
      baro_backlog_.push_back(b);
      if (baro_backlog_.size() > 400) baro_backlog_.pop_front();
    }
  }

  static bool depth_to_meters(const sm::Image& img, strataslam::DepthImage* out) {
    out->width = static_cast<int>(img.width);
    out->height = static_cast<int>(img.height);
    const size_t n = static_cast<size_t>(img.width) * img.height;
    out->meters.assign(n, 0.f);
    if (img.encoding == "32FC1") {
      if (img.data.size() < n * 4) return false;
      for (size_t i = 0; i < n; ++i) {
        float v;
        std::memcpy(&v, &img.data[i * 4], 4);
        out->meters[i] = std::isfinite(v) ? v : 0.f;
      }
      return true;
    }
    if (img.encoding == "16UC1") {
      if (img.data.size() < n * 2) return false;
      for (size_t i = 0; i < n; ++i) {
        uint16_t v;
        std::memcpy(&v, &img.data[i * 2], 2);
        out->meters[i] = v * 0.001f;
      }
      return true;
    }
    return false;
  }

  bool fill_frame(const sm::Image& depth, const sm::CameraInfo& info, strataslam::RgbdFrame* f) {
    f->stamp = stamp_ns(depth.header.stamp);
    f->K.fx = info.k[0];
    f->K.fy = info.k[4];
    f->K.cx = info.k[2];
    f->K.cy = info.k[5];
    f->K.width = static_cast<int>(info.width);
    f->K.height = static_cast<int>(info.height);
    if (!(f->K.fx > 0) ||
        !(f->K.fy > 0)) {  // NaN or zero: some drivers publish a depth camera_info without K
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "camera_info on %s has no valid K (fx=%g): frames dropped; use the colour camera_info",
          cfg_.sensors.camera.info_topic.c_str(), f->K.fx);
      return false;
    }
    if (!depth_to_meters(depth, &f->depth)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "unsupported depth encoding '%s' (need 16UC1 or 32FC1)",
                           depth.encoding.c_str());
      return false;
    }
    return f->K.fx > 0 && f->depth.width == f->K.width && f->depth.height == f->K.height;
  }

  void on_features(const sm::Image::ConstSharedPtr& depth,
                   const strataslam_msgs::msg::FeatureFrame::ConstSharedPtr& feat,
                   const sm::CameraInfo::ConstSharedPtr& info) {
    camera_frame_pending_ = depth->header.frame_id;
    if (!ensure_system(depth->header.frame_id, lidar_frame_pending_, imu_frame_pending_)) return;
    strataslam::RgbdFrame f;
    if (!fill_frame(*depth, *info, &f)) return;
    const size_t n = std::min({feat->u.size(), feat->v.size(), feat->response.size(),
                               feat->descriptors.size() / strataslam::kDescriptorBytes});
    for (size_t i = 0; i < n; ++i) {
      strataslam::RgbKeypoint k;
      k.u = feat->u[i];
      k.v = feat->v[i];
      k.response = feat->response[i];
      std::memcpy(k.descriptor.data(), &feat->descriptors[i * strataslam::kDescriptorBytes],
                  strataslam::kDescriptorBytes);
      f.keypoints.push_back(k);
    }
    process_rgbd(f, nullptr);
  }

  void on_rgb(const sm::Image::ConstSharedPtr& depth, const sm::Image::ConstSharedPtr& rgb,
              const sm::CameraInfo::ConstSharedPtr& info) {
    camera_frame_pending_ = depth->header.frame_id;
    if (!ensure_system(depth->header.frame_id, lidar_frame_pending_, imu_frame_pending_)) return;
    strataslam::RgbdFrame f;
    if (!fill_frame(*depth, *info, &f)) return;
    strataslam::GrayImage g;
    g.width = static_cast<int>(rgb->width);
    g.height = static_cast<int>(rgb->height);
    g.data.resize(static_cast<size_t>(g.width) * g.height);
    const bool mono = rgb->encoding == "mono8";
    const bool bgr = rgb->encoding == "bgr8", rgb8 = rgb->encoding == "rgb8";
    if (!mono && !bgr && !rgb8) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "unsupported color encoding '%s' (mono8, rgb8, bgr8)",
                           rgb->encoding.c_str());
      return;
    }
    for (size_t i = 0; i < g.data.size(); ++i) {
      if (mono)
        g.data[i] = rgb->data[i];
      else
        g.data[i] = static_cast<uint8_t>(0.299 * rgb->data[3 * i + (bgr ? 2 : 0)] +
                                         0.587 * rgb->data[3 * i + 1] +
                                         0.114 * rgb->data[3 * i + (bgr ? 0 : 2)]);
    }
    // Only keypoints with a depth can become landmarks of the RGB-D tracker: detect them there.
    const auto with_depth = strataslam::valid_depth_mask(f.depth, cfg_.sensors.camera.depth_min,
                                                         cfg_.sensors.camera.depth_max);
    f.keypoints = orb_->extract(g, &with_depth);
    if (!mono && cfg_.fusion.colorize_cloud) {
      f.rgb.width = g.width;
      f.rgb.height = g.height;
      f.rgb.data.resize(g.data.size() * 3);
      for (size_t i = 0; i < g.data.size(); ++i)
        for (int c = 0; c < 3; ++c) f.rgb.data[3 * i + c] = rgb->data[3 * i + (bgr ? 2 - c : c)];
    }
    process_rgbd(f, rgb.get());
  }

  /// `rgb`: the colour image the keypoints come from (null with feature-frame input), for the live
  /// view.
  void process_rgbd(const strataslam::RgbdFrame& f, const sm::Image* rgb) {
    slam_->add_rgbd(f);
    publish_state(f.stamp);
    publish_camera_view(f, rgb);
    if (depth_cloud_pub_->get_subscription_count() > 0 &&
        std::chrono::steady_clock::now() >= next_depth_pub_) {
      strataslam::BackprojectionConfig bp;
      bp.depth_min = cfg_.sensors.camera.depth_min;
      bp.depth_max = cfg_.sensors.camera.depth_max;
      publish_live(*depth_cloud_pub_, strataslam::depth_to_cloud(f.depth, f.K, 4, bp),
                   ext_.T_base_camera, f.stamp, next_depth_pub_);
    }
  }

  /// The image, the depth and the keypoints of a camera frame with what the tracking did with each
  /// one (strataslam_msgs/CameraView), reduced to at most kViewWidth pixels wide: at most
  /// kViewPeriod apart (wall clock), and only when someone listens.
  void publish_camera_view(const strataslam::RgbdFrame& f, const sm::Image* rgb) {
    using strataslam_msgs::msg::CameraView;
    constexpr auto kViewPeriod = std::chrono::milliseconds(200);
    constexpr int kViewWidth = 320;
    const auto now = std::chrono::steady_clock::now();
    if (camera_view_pub_->get_subscription_count() == 0 || now < next_camera_view_) return;
    next_camera_view_ = now + kViewPeriod;
    const int step = std::max(1, (f.depth.width + kViewWidth - 1) / kViewWidth);
    CameraView v;
    v.header.stamp = rclcpp::Time(f.stamp, RCL_ROS_TIME);
    v.header.frame_id = camera_frame_pending_;
    const auto& cam = cfg_.sensors.camera;
    v.depth.width = static_cast<uint32_t>(f.depth.width / step);
    v.depth.height = static_cast<uint32_t>(f.depth.height / step);
    v.depth.encoding = "32FC1";
    v.depth.step = v.depth.width * 4;
    v.depth.data.resize(static_cast<size_t>(v.depth.step) * v.depth.height);
    for (uint32_t r = 0; r < v.depth.height; ++r) {
      for (uint32_t c = 0; c < v.depth.width; ++c) {
        float z = f.depth.meters[static_cast<size_t>(r * step) * f.depth.width + c * step];
        if (!(z >= cam.depth_min && z <= cam.depth_max)) z = 0.f;
        std::memcpy(&v.depth.data[(static_cast<size_t>(r) * v.depth.width + c) * 4], &z, 4);
      }
    }
    const bool bgr = rgb && rgb->encoding == "bgr8";
    if (rgb && static_cast<int>(rgb->width) == f.depth.width &&
        static_cast<int>(rgb->height) == f.depth.height &&
        (bgr || rgb->encoding == "rgb8" || rgb->encoding == "mono8")) {
      const int ch = rgb->encoding == "mono8" ? 1 : 3;
      v.image.width = v.depth.width;
      v.image.height = v.depth.height;
      v.image.encoding = ch == 1 ? "mono8" : "rgb8";
      v.image.step = v.image.width * ch;
      v.image.data.resize(static_cast<size_t>(v.image.step) * v.image.height);
      for (uint32_t r = 0; r < v.image.height; ++r) {
        const uint8_t* src = &rgb->data[static_cast<size_t>(r * step) * rgb->step];
        uint8_t* dst = &v.image.data[static_cast<size_t>(r) * v.image.step];
        for (uint32_t c = 0; c < v.image.width; ++c)
          for (int k = 0; k < ch; ++k) dst[c * ch + k] = src[c * step * ch + (bgr ? 2 - k : k)];
      }
    }
    // Without camera tracking (LiDAR odometry) the keypoints only serve loop closures.
    const auto use = slam_->keypoint_use();
    for (size_t i = 0; i < f.keypoints.size(); ++i) {
      v.u.push_back(f.keypoints[i].u / static_cast<float>(step));
      v.v.push_back(f.keypoints[i].v / static_cast<float>(step));
      uint8_t kind = use ? CameraView::UNMATCHED : CameraView::NOT_TRACKED;
      if (use && use->size() == f.keypoints.size()) {
        kind = (*use)[i] == strataslam::KeypointUse::Inlier     ? CameraView::INLIER
               : (*use)[i] == strataslam::KeypointUse::Rejected ? CameraView::REJECTED
                                                                : CameraView::UNMATCHED;
      }
      v.use.push_back(kind);
    }
    camera_view_pub_->publish(v);
  }

  /// The points a sensor sees now, moved to the odometry frame with the pose just estimated: at
  /// most kLivePeriod apart (wall clock) and kLiveMaxPoints per message, and only when someone
  /// listens.
  void publish_live(rclcpp::Publisher<sm::PointCloud2>& pub,
                    const std::vector<Eigen::Vector3f>& pts, const gtsam::Pose3& T_base_sensor,
                    Timestamp stamp, std::chrono::steady_clock::time_point& next) {
    constexpr auto kLivePeriod = std::chrono::milliseconds(100);
    constexpr size_t kLiveMaxPoints = 20000;
    const auto now = std::chrono::steady_clock::now();
    if (pts.empty() || pub.get_subscription_count() == 0 || now < next) return;
    const auto pose = slam_->current_pose_odom();
    if (!pose) return;
    next = now + kLivePeriod;
    const gtsam::Pose3 T_odom_sensor = *pose * T_base_sensor;
    const Eigen::Matrix3f R = T_odom_sensor.rotation().matrix().cast<float>();
    const Eigen::Vector3f tr = T_odom_sensor.translation().cast<float>();
    const size_t step = (pts.size() + kLiveMaxPoints - 1) / kLiveMaxPoints;
    std::vector<Eigen::Vector3f> out;
    out.reserve(pts.size() / step + 1);
    for (size_t i = 0; i < pts.size(); i += step) out.push_back(R * pts[i] + tr);
    pub.publish(xyz_cloud(out, cfg_.robot.odom_frame, rclcpp::Time(stamp, RCL_ROS_TIME)));
  }

  size_t lidar_received_ = 0;
  size_t lidar_malformed_ =
      0;  ///< PointCloud2 messages dropped because their layout is inconsistent
  void on_lidar(const sm::PointCloud2& m) {
    ++lidar_received_;
    lidar_frame_pending_ = m.header.frame_id;
    if (!ensure_system(camera_frame_pending_, m.header.frame_id, imu_frame_pending_)) return;
    const sm::PointField *fx = nullptr, *fy = nullptr, *fz = nullptr, *ft = nullptr;
    for (const auto& f : m.fields) {
      if (f.name == "x")
        fx = &f;
      else if (f.name == "y")
        fy = &f;
      else if (f.name == "z")
        fz = &f;
      else if (f.name == "t" || f.name == "time")
        ft = &f;
    }
    if (!fx || !fy || !fz) return;
    if (const std::string err = cloud_layout_error(m, {fx, fy, fz, ft}); !err.empty()) {
      ++lidar_malformed_;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "dropping malformed PointCloud2 on %s (%zu so far): %s",
                           cfg_.sensors.lidar.topic.c_str(), lidar_malformed_, err.c_str());
      return;
    }
    strataslam::LidarScan scan;
    scan.stamp = stamp_ns(m.header.stamp);
    const size_t n = static_cast<size_t>(m.width) * m.height;
    const size_t row = m.height > 1 ? m.row_step : static_cast<size_t>(m.width) * m.point_step;
    scan.points.reserve(n);
    std::vector<double> times;
    for (size_t i = 0; i < n; ++i) {
      const uint8_t* p = m.data.data() + (i / m.width) * row + (i % m.width) * m.point_step;
      const Eigen::Vector3f pt(static_cast<float>(read_field(p, *fx)),
                               static_cast<float>(read_field(p, *fy)),
                               static_cast<float>(read_field(p, *fz)));
      if (!pt.allFinite()) continue;
      scan.points.push_back(pt);
      if (ft)
        times.push_back(read_field(p, *ft) * (ft->datatype == sm::PointField::UINT32 ? 1e-9 : 1.0));
    }
    if (ft && !times.empty()) {
      const double t0 = *std::min_element(times.begin(), times.end());
      for (double t : times) scan.time_offsets_s.push_back(static_cast<float>(t - t0));
    }
    slam_->add_lidar(scan);
    publish_state(scan.stamp);
    publish_live(*scan_pub_, scan.points, ext_.T_base_lidar, scan.stamp, next_scan_pub_);
  }

  // ---------------- publishing ----------------

  /// Whether the robot itself publishes odom -> base on TF (its wheel odometry). Decided once,
  /// online, before the node publishes anything there: then the node cannot publish that transform
  /// too (a frame has one parent) and publishes floor_<k> -> odom as the correction on top of the
  /// robot's odometry, its own odometry going to a sibling frame `strataslam_odom`. Otherwise (no
  /// wheel odometry, or a bag processed offline) it publishes odom -> base itself and the tree is
  /// whole (audit AUD-044). Another node giving odom a parent (a localizer's map -> odom) and the
  /// SLAM's floor_<k> -> odom make two parents for one frame: TF keeps one, and the floors hang
  /// loose. Said once, with what to do; nothing is changed.
  void warn_if_odom_has_another_parent() {
    if (odom_conflict_warned_ || !tf_listener_) return;
    std::string parent;
    if (!tf_buffer_->_getParent(cfg_.robot.odom_frame, tf2::TimePointZero, parent) ||
        parent.rfind("floor_", 0) == 0)
      return;
    odom_conflict_warned_ = true;
    RCLCPP_WARN(get_logger(),
                "TF: %s also has the parent '%s', from another node (a localizer such as AMCL "
                "publishing %s -> %s?). A frame "
                "has one parent: if that node owns %s -> %s, set output.publish_tf: false in the "
                "profile (StrataSLAM then "
                "publishes only floor_<k> -> strataslam_odom); if StrataSLAM should localize, stop "
                "the other node",
                cfg_.robot.odom_frame.c_str(), parent.c_str(), parent.c_str(),
                cfg_.robot.odom_frame.c_str(), parent.c_str(), cfg_.robot.odom_frame.c_str());
  }

  bool robot_publishes_odom() {
    if (!robot_odom_tf_) {
      robot_odom_tf_ =
          tf_listener_ && tf_buffer_->canTransform(cfg_.robot.odom_frame, cfg_.robot.base_frame,
                                                   tf2::TimePointZero);
      RCLCPP_INFO(get_logger(),
                  *robot_odom_tf_ ? "TF %s -> %s comes from the robot: publishing floor_<k> -> %s "
                                    "as the correction on top of it, "
                                    "the SLAM odometry in strataslam_odom"
                                  : "no TF %s -> %s from the robot: publishing it (and floor_<k> "
                                    "-> %s) from the SLAM odometry",
                  cfg_.robot.odom_frame.c_str(), cfg_.robot.base_frame.c_str(),
                  cfg_.robot.odom_frame.c_str());
    }
    return *robot_odom_tf_;
  }

  void publish_state(Timestamp stamp_ns_value) {
    const rclcpp::Time stamp(stamp_ns_value, RCL_ROS_TIME);
    const auto pose = slam_->current_pose_odom();
    const auto st = slam_->floor_status();
    // output.publish_tf: false leaves map and odom to another node (a localizer that owns map ->
    // odom): the SLAM keeps only its own branch floor_<k> -> strataslam_odom.
    const bool tf_tree = cfg_.output.publish_tf;
    if (!tf_tree && !tf_mode_logged_) {
      tf_mode_logged_ = true;
      RCLCPP_INFO(get_logger(),
                  "output.publish_tf is false: publishing only floor_<k> -> strataslam_odom; %s "
                  "and %s are left to other nodes",
                  cfg_.robot.map_frame.c_str(), cfg_.robot.odom_frame.c_str());
    }
    const bool robot_odom = tf_tree && robot_publishes_odom();
    const std::string slam_odom_frame =
        (robot_odom || !tf_tree) ? std::string("strataslam_odom") : cfg_.robot.odom_frame;
    if (tf_tree) warn_if_odom_has_another_parent();
    if (pose) {
      nav_msgs::msg::Odometry o;
      o.header.stamp = stamp;
      o.header.frame_id = slam_odom_frame;
      o.child_frame_id = cfg_.robot.base_frame;
      const auto t = to_transform(*pose);
      o.pose.pose.position.x = t.translation.x;
      o.pose.pose.position.y = t.translation.y;
      o.pose.pose.position.z = t.translation.z;
      o.pose.pose.orientation = t.rotation;
      odom_pub_->publish(o);
    }
    if (const auto fo = slam_->floor_from_odom()) {
      const std::string floor_frame =
          "floor_" +
          std::to_string(std::max<int>(
              0, static_cast<int>(slam_->floor_status().floor < 0 ? last_floor_
                                                                  : slam_->floor_status().floor)));
      std::vector<geometry_msgs::msg::TransformStamped> out;
      geometry_msgs::msg::TransformStamped ts;
      ts.header.stamp = stamp;
      ts.header.frame_id = floor_frame;
      ts.child_frame_id = slam_odom_frame;
      ts.transform = to_transform(*fo);
      out.push_back(ts);
      if (tf_tree) {
        // map -> floor_<k> for every floor seen: map is the first floor, and each other floor sits
        // where the odometry entered it (its height included). `strataslam fuse` refines this
        // offline.
        auto floors = floor_from_odom_;
        if (st.floor >= 0) floors[st.floor] = *fo;
        if (!root_floor_ && !floors.empty()) root_floor_ = floors.begin()->first;
        if (root_floor_ && floors.count(*root_floor_)) {
          for (const auto& [k, T_floor_odom] : floors) {
            geometry_msgs::msg::TransformStamped tm;
            tm.header.stamp = stamp;
            tm.header.frame_id = cfg_.robot.map_frame;
            tm.child_frame_id = "floor_" + std::to_string(k);
            tm.transform = to_transform(floors.at(*root_floor_) * T_floor_odom.inverse());
            out.push_back(tm);
          }
        }
      }
      if (!tf_tree) {
        // nothing above or on the robot's frames
      } else if (pose && !robot_odom) {
        geometry_msgs::msg::TransformStamped
            tb;  // odom -> base from the SLAM odometry: the tree is whole
        tb.header.stamp = stamp;
        tb.header.frame_id = cfg_.robot.odom_frame;
        tb.child_frame_id = cfg_.robot.base_frame;
        tb.transform = to_transform(*pose);
        out.push_back(tb);
      } else if (pose && robot_odom) {
        // floor -> odom(robot) = (floor -> base, from the SLAM) * (odom(robot) -> base)^-1, at this
        // stamp if TF has it.
        try {
          geometry_msgs::msg::TransformStamped wheel;
          try {
            wheel =
                tf_buffer_->lookupTransform(cfg_.robot.odom_frame, cfg_.robot.base_frame, stamp);
          } catch (const tf2::TransformException&) {
            wheel = tf_buffer_->lookupTransform(cfg_.robot.odom_frame, cfg_.robot.base_frame,
                                                tf2::TimePointZero);
          }
          geometry_msgs::msg::TransformStamped tc;
          tc.header.stamp = stamp;
          tc.header.frame_id = floor_frame;
          tc.child_frame_id = cfg_.robot.odom_frame;
          tc.transform = to_transform((*fo) * (*pose) * from_transform(wheel.transform).inverse());
          out.push_back(tc);
        } catch (const tf2::TransformException& e) {
          RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 5000, "no TF %s -> %s for the correction: %s",
              cfg_.robot.odom_frame.c_str(), cfg_.robot.base_frame.c_str(), e.what());
        }
      }
      tf_broadcaster_->sendTransform(out);
    }
    if (st.floor >= 0) {
      last_floor_ = st.floor;
      if (const auto fo = slam_->floor_from_odom()) floor_from_odom_[st.floor] = *fo;
    }
    if (st.floor != last_state_floor_ || st.pending != last_pending_ ||
        st.num_floors != last_num_floors_ || state_counter_++ % 10 == 0) {
      strataslam_msgs::msg::FloorState fs;
      fs.header.stamp = stamp;
      fs.header.frame_id = cfg_.robot.map_frame;
      fs.floor = st.floor;
      fs.state = st.pending                                         ? "pending"
                 : st.state == strataslam::FloorState::InTransition ? "in_transition"
                 : st.state == strataslam::FloorState::Lost         ? "lost"
                                                                    : "on_floor";
      fs.transition_type = st.last_transition ? strataslam::to_string(*st.last_transition) : "";
      fs.session = st.session;
      fs.num_floors = static_cast<uint32_t>(st.num_floors);
      fs.altitude = st.altitude;
      if (const auto h = slam_->baro_height()) {
        fs.has_baro = true;
        fs.baro_height = *h;
      }
      state_pub_->publish(fs);
      last_state_floor_ = st.floor;
      last_pending_ = st.pending;
      last_num_floors_ = st.num_floors;
    }
    const size_t kfs = slam_->keyframes().size();
    if (kfs != last_kf_count_) {
      last_kf_count_ = kfs;
      publish_path(stamp);
      check_same_level(false);
    }
  }

  void publish_path(const rclcpp::Time& stamp) {
    const FloorIdLocal floor = last_floor_;
    nav_msgs::msg::Path path;
    path.header.stamp = stamp;
    path.header.frame_id = "floor_" + std::to_string(floor);
    for (const auto& [t, pose] : slam_->floor_path(floor)) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header.stamp = rclcpp::Time(t, RCL_ROS_TIME);
      ps.header.frame_id = path.header.frame_id;
      const auto tf = to_transform(pose);
      ps.pose.position.x = tf.translation.x;
      ps.pose.position.y = tf.translation.y;
      ps.pose.position.z = tf.translation.z;
      ps.pose.orientation = tf.rotation;
      path.poses.push_back(ps);
    }
    path_pub_->publish(path);
  }

  void publish_cloud(bool force = false) {
    // Building the cloud of a whole floor is not free: only when someone listens (a viewer, RViz),
    // and at most a tenth of the time. A floor of a large map (a 1 km drive: 9 million points, 100
    // MB) takes long to build and to send; at a fixed period of one second it took most of the run,
    // and the viewer's machine froze under the messages.
    constexpr double kCloudShare = 0.1;
    const auto start = std::chrono::steady_clock::now();
    if (!slam_ || slam_->keyframes().empty() || cloud_pub_->get_subscription_count() == 0 ||
        (!force && start < cloud_not_before_))
      return;
    const auto pts = slam_->floor_cloud(last_floor_, cfg_.output.voxel_size);
    if (!pts.empty())
      cloud_pub_->publish(xyz_cloud(pts, "floor_" + std::to_string(last_floor_), now()));
    const auto spent = std::chrono::steady_clock::now() - start;
    cloud_not_before_ = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                    spent * (1.0 / kCloudShare));
  }

  // ---------------- same-level candidates ----------------

  /// Floors that could be one level (a drift of the odometry in z split it, or the robot came back
  /// to a floor without relocalizing on it), checked as `strataslam fuse` checks them (ADR-0031)
  /// and only reported. While on a floor: that floor against the others within same_level_max_dz,
  /// every kLevelCheckKeyframes keyframes; at the end of a bag (`all`), every pair. The clouds are
  /// copied here (read-only, so the run is unchanged) and registered in a background thread; its
  /// findings are logged on the next call.
  void check_same_level(bool all) {
    constexpr size_t kLevelCheckKeyframes = 20;
    collect_same_level(all);
    if (!slam_) return;  // the pipeline never started (no sensor data, or the TF never arrived)
    if (level_job_.valid()) return;  // the previous check is still running
    const auto st = slam_->floor_status();
    std::vector<std::pair<int, int>> pairs;
    if (all) {
      for (auto i = floor_from_odom_.begin(); i != floor_from_odom_.end(); ++i)
        for (auto j = std::next(i); j != floor_from_odom_.end(); ++j)
          pairs.push_back({i->first, j->first});
    } else {
      if (st.floor < 0 || st.state != strataslam::FloorState::OnFloor ||
          !floor_from_odom_.count(st.floor))
        return;
      const size_t kfs = slam_->floor_path(st.floor).size();
      if (kfs < level_checked_kfs_[st.floor] + kLevelCheckKeyframes) return;
      level_checked_kfs_[st.floor] = kfs;
      for (const auto& [g, unused] : floor_from_odom_)
        if (g != st.floor)
          pairs.push_back({std::min<int>(g, st.floor), std::max<int>(g, st.floor)});
    }
    struct Job {
      int a, b;
      strataslam::Pose4 a_to_b;
    };
    std::vector<Job> jobs;
    std::map<int, strataslam::PointCloud> clouds;
    const strataslam::FusionOptions opt;
    for (const auto& [a, b] : pairs) {
      if (level_reported_.count({a, b})) continue;
      // Pose of b's floor frame in a's: in the plane through the odometry frame that both were
      // tracked in; in height from the floor manager's levels, since the odometry misses an
      // elevator or a change while lost.
      const gtsam::Pose3 a_to_b = floor_from_odom_.at(a) * floor_from_odom_.at(b).inverse();
      const auto alt_a = slam_->floor_altitude(a), alt_b = slam_->floor_altitude(b);
      if (!alt_a || !alt_b) continue;
      const double dz = *alt_b - *alt_a;
      if (std::abs(dz) > opt.same_level_max_dz) continue;
      jobs.push_back({a, b,
                      strataslam::Pose4(a_to_b.rotation().yaw(),
                                        Eigen::Vector3d(a_to_b.x(), a_to_b.y(), dz))});
      for (int f : {a, b})
        if (!clouds.count(f)) clouds[f].points = slam_->floor_cloud(f, cfg_.output.voxel_size);
    }
    if (jobs.empty()) return;
    level_job_ = std::async(std::launch::async, [jobs, clouds = std::move(clouds), opt] {
      std::vector<strataslam::LevelCandidate> found;
      for (const auto& j : jobs)
        if (auto c = strataslam::same_level_candidate(j.a, clouds.at(j.a), j.b, clouds.at(j.b),
                                                      j.a_to_b, opt))
          found.push_back(*c);
      return found;
    });
    if (all) collect_same_level(true);  // end of the bag: wait for it and log what it found
  }

  /// Logs the findings of the background check once it is done (`wait`: wait for it).
  void collect_same_level(bool wait) {
    if (!level_job_.valid() ||
        (!wait && level_job_.wait_for(std::chrono::seconds(0)) != std::future_status::ready))
      return;
    for (const auto& c : level_job_.get()) {
      if (!level_reported_.insert({static_cast<int>(c.a), static_cast<int>(c.b)}).second) continue;
      RCLCPP_WARN(get_logger(),
                  "floors %d and %d could be the same level: compatible structure (score %.2f%s), "
                  "heights %.2f m apart"
                  " (a candidate, not applied: confirm with `strataslam fuse --same-level %d,%d`)",
                  static_cast<int>(c.a), static_cast<int>(c.b), c.score,
                  c.structure_ambiguous ? ", but it also fits elsewhere" : "", std::abs(c.dz),
                  static_cast<int>(c.a), static_cast<int>(c.b));
    }
  }

  static sm::PointCloud2 xyz_cloud(const std::vector<Eigen::Vector3f>& pts,
                                   const std::string& frame, const rclcpp::Time& stamp) {
    sm::PointCloud2 c;
    c.header.stamp = stamp;
    c.header.frame_id = frame;
    c.height = 1;
    c.width = static_cast<uint32_t>(pts.size());
    c.is_bigendian = false;
    c.is_dense = true;
    c.point_step = 12;
    c.row_step = c.point_step * c.width;
    for (const char* name : {"x", "y", "z"}) {
      sm::PointField f;
      f.name = name;
      f.offset = static_cast<uint32_t>(c.fields.size() * 4);
      f.datatype = sm::PointField::FLOAT32;
      f.count = 1;
      c.fields.push_back(f);
    }
    c.data.resize(static_cast<size_t>(c.row_step));
    for (size_t i = 0; i < pts.size(); ++i) std::memcpy(c.data.data() + 12 * i, pts[i].data(), 12);
    return c;
  }

  // ---------------- services ----------------

  void on_save_map(std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
    if (!slam_) {
      res->success = false;
      res->message = "the pipeline has not started yet";
      return;
    }
    const std::string dir = expand_map_dir(cfg_.output.map_dir);
    const strataslam::MapData map = slam_->finalize(profile_text_);
    const auto w = strataslam::write_map(dir, map);
    res->success = static_cast<bool>(w);
    res->message = w ? dir : w.error().message;
    if (w) keyframes_at_last_save_ = slam_->keyframes().size();
    RCLCPP_INFO(
        get_logger(),
        "save_map: %s (%zu keyframes, %zu loop closures accepted, lidar scans received: %zu)",
        res->message.c_str(), slam_->keyframes().size(), slam_->loops().size(), lidar_received_);
    // Said again at the end: the warning at start-up scrolls away, and a map without loops looks
    // like a failed SLAM.
    if (visual_loops_off_) {
      RCLCPP_WARN(get_logger(),
                  "save_map: visual loop closure was disabled for the whole run: no vocabulary "
                  "(set fusion.vocabulary in "
                  "the profile or pass vocabulary:=)");
    }
  }

  void on_mark_floor_change(const strataslam_msgs::srv::MarkFloorChange::Request& req,
                            strataslam_msgs::srv::MarkFloorChange::Response& res) {
    if (!slam_) {
      res.success = false;
      res.message = "the pipeline has not started yet";
      return;
    }
    strataslam::TransitionType type = strataslam::TransitionType::Unknown;
    if (!req.transition_type.empty() &&
        !strataslam::parse_transition_type(req.transition_type, &type)) {
      res.success = false;
      res.message = "unknown transition type '" + req.transition_type + "'";
      return;
    }
    slam_->mark_floor_change(type);
    res.success = true;
    res.floor = slam_->floor_status().floor;
    res.message = "floor change recorded";
    publish_state(static_cast<Timestamp>(now().nanoseconds()));
  }

  using FloorIdLocal = int;
  using DepthSub = message_filters::Subscriber<sm::Image>;
  using InfoSub = message_filters::Subscriber<sm::CameraInfo>;
  using FeatSub = message_filters::Subscriber<strataslam_msgs::msg::FeatureFrame>;
  using FeatPolicy =
      message_filters::sync_policies::ApproximateTime<sm::Image, strataslam_msgs::msg::FeatureFrame,
                                                      sm::CameraInfo>;
  using RgbPolicy =
      message_filters::sync_policies::ApproximateTime<sm::Image, sm::Image, sm::CameraInfo>;

  std::string bag_path_;
  rclcpp::TimerBase::SharedPtr offline_timer_;
  strataslam_msgs::msg::Instance instance_;
  rclcpp::Publisher<strataslam_msgs::msg::Instance>::SharedPtr instance_pub_;
  rclcpp::Subscription<strataslam_msgs::msg::Instance>::SharedPtr instance_sub_;
  bool refused_ = false;
  std::optional<bool> robot_odom_tf_;
  std::optional<int> root_floor_;  ///< the floor that is map (the first one)
  bool tf_mode_logged_ = false, odom_conflict_warned_ = false;
  size_t keyframes_at_last_save_ =
      0;  ///< keyframes of the pipeline when the map was last saved  ///< the robot publishes odom
          ///< -> base itself (decided once)
  rclcpp::TimerBase::SharedPtr startup_failed_timer_;
  std::string profile_path_, vocabulary_path_, features_topic_, profile_text_;
  bool visual_loops_off_ = false;  ///< loops would come from the camera, but there is no vocabulary
  double cloud_period_s_ = 5.0;
  std::chrono::steady_clock::time_point cloud_not_before_{};  ///< see publish_cloud
  double wait_for_viewer_s_ = 0.0;
  strataslam::Extrinsics ext_;
  std::map<int, gtsam::Pose3>
      floor_from_odom_;  ///< latest T_floor_odom of every floor (same-level checks)
  std::map<int, size_t> level_checked_kfs_;  ///< keyframes of a floor at its last same-level check
  std::set<std::pair<int, int>> level_reported_;
  std::future<std::vector<strataslam::LevelCandidate>> level_job_;
  rclcpp::Publisher<sm::PointCloud2>::SharedPtr scan_pub_, depth_cloud_pub_;
  rclcpp::Publisher<strataslam_msgs::msg::CameraView>::SharedPtr camera_view_pub_;
  std::chrono::steady_clock::time_point next_scan_pub_{}, next_depth_pub_{}, next_camera_view_{};
  strataslam::Config cfg_;
  strataslam::SystemTuning tuning_;
  std::unique_ptr<strataslam::FeatureExtractor> orb_;
  std::unique_ptr<strataslam::SlamSystem> slam_;
  std::deque<strataslam::ImuSample> imu_backlog_;
  std::deque<strataslam::BaroSample> baro_backlog_;
  std::string camera_frame_, lidar_frame_, imu_frame_;
  std::string camera_frame_pending_, lidar_frame_pending_, imu_frame_pending_;
  int last_floor_ = 0, last_state_floor_ = -2;
  bool last_pending_ = false;
  size_t last_num_floors_ = 0, last_kf_count_ = 0, state_counter_ = 0;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<strataslam_msgs::msg::FloorState>::SharedPtr state_pub_;
  rclcpp::Publisher<sm::PointCloud2>::SharedPtr cloud_pub_;
  DepthSub depth_sub_, rgb_sub_;
  InfoSub info_sub_;
  FeatSub feat_sub_;
  std::unique_ptr<message_filters::Synchronizer<FeatPolicy>> feat_sync_;
  std::unique_ptr<message_filters::Synchronizer<RgbPolicy>> rgb_sync_;
  rclcpp::Subscription<sm::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sm::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sm::FluidPressure>::SharedPtr baro_sub_;
  rclcpp::Subscription<sm::FluidPressure>::SharedPtr baro_ref_sub_;
  std::optional<strataslam::BaroSample> last_baro_reference_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ext_odom_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr ext_tf_sub_;
  std::deque<std::pair<Timestamp, std::function<void()>>>
      camera_queue_;  ///< frames waiting for the external odometry
  Timestamp newest_camera_ = 0, last_ext_stamp_ = 0;
  bool ext_tf_reported_ = false, ext_odom_warned_ = false;
  std::optional<Timestamp> first_unodometered_;  ///< first camera frame without external odometry
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_, session_srv_;
  rclcpp::Service<strataslam_msgs::srv::MarkFloorChange>::SharedPtr mark_srv_;
  rclcpp::TimerBase::SharedPtr cloud_timer_;
};

}  // namespace strataslam_ros

RCLCPP_COMPONENTS_REGISTER_NODE(strataslam_ros::StrataslamNode)
