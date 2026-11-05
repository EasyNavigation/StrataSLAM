// strataslam_sim_export_bag: writes a scenario of the simulator as a rosbag2, with exact ground
// truth beside it, for the ROS 2 tests and for `strataslam doctor`. The simulator itself stays
// ROS-free.
//
//   strataslam_sim_export_bag <S01..S12> <out_dir> [--profile profile.yaml] [--corrupt <kind>]
//                           [--camera-hz 15] [--lidar-hz 12] [--seconds N]
// Writes <out_dir>/bag (mcap/sqlite3), <out_dir>/vocab.bin and <out_dir>/trajectory.tum (ground
// truth, bag time). Corruptions for the doctor tests: no_tf, depth_encoding, rgb_depth_offset,
// imu_z_flip, no_camera_info.
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <iostream>
#include <rclcpp/qos.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/topic_metadata.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <tf2_msgs/msg/tf_message.hpp>

#include "strataslam/config/config_loader.hpp"
#include "strataslam_msgs/msg/feature_frame.hpp"
#include "strataslam_sim/scenarios.hpp"

namespace sm = sensor_msgs::msg;

namespace {

builtin_interfaces::msg::Time to_time(int64_t ns) {
  builtin_interfaces::msg::Time t;
  t.sec = static_cast<int32_t>(ns / 1000000000LL);
  t.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
  return t;
}

template <typename Msg>
void write(rosbag2_cpp::Writer& w, const std::string& topic, const Msg& msg, int64_t ns) {
  w.write(msg, topic, rclcpp::Time(ns, RCL_ROS_TIME));
}

geometry_msgs::msg::TransformStamped static_tf(const std::string& parent, const std::string& child,
                                               const gtsam::Pose3& p, int64_t ns) {
  geometry_msgs::msg::TransformStamped t;
  t.header.stamp = to_time(ns);
  t.header.frame_id = parent;
  t.child_frame_id = child;
  const auto q = p.rotation().toQuaternion();
  t.transform.translation.x = p.x();
  t.transform.translation.y = p.y();
  t.transform.translation.z = p.z();
  t.transform.rotation.w = q.w();
  t.transform.rotation.x = q.x();
  t.transform.rotation.y = q.y();
  t.transform.rotation.z = q.z();
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " <scenario> <out_dir> [--profile p.yaml] [--corrupt kind] [--camera-hz hz] "
                 "[--seconds s]\n";
    return 2;
  }
  const std::string scenario = argv[1], out = argv[2];
  std::string profile_path, corrupt;
  double camera_hz = 15.0, lidar_hz = 12.0, seconds = 1e9;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--profile")
      profile_path = argv[i + 1];
    else if (k == "--corrupt")
      corrupt = argv[i + 1];
    else if (k == "--camera-hz")
      camera_hz = std::stod(argv[i + 1]);
    else if (k == "--lidar-hz")
      lidar_hz = std::stod(argv[i + 1]);
    else if (k == "--seconds")
      seconds = std::stod(argv[i + 1]);
  }
  auto spec = strataslam_sim::make_scenario(scenario);
  // Topics and sensors come from the profile when given (so every example profile can be
  // exercised).
  strataslam::Config cfg = spec.slam;
  std::string rgb_t = "/camera/color/image_raw",
              depth_t = "/camera/aligned_depth_to_color/image_raw",
              info_t = "/camera/color/camera_info", lidar_t = "/points", imu_t = "/camera/imu",
              baro_t = "/pressure";
  std::string base = "base_link";
  if (!profile_path.empty()) {
    const auto r = strataslam::load_profile_file(profile_path);
    if (!r.ok()) {
      for (const auto& e : r.errors) std::cerr << e.format() << "\n";
      return 1;
    }
    cfg = *r.config;
    rgb_t = cfg.sensors.camera.rgb_topic;
    depth_t = cfg.sensors.camera.depth_topic;
    info_t = cfg.sensors.camera.info_topic;
    lidar_t = cfg.sensors.lidar.topic;
    imu_t = cfg.sensors.imu.topic;
    baro_t = cfg.sensors.barometer.topic;
    base = cfg.robot.base_frame;
    spec.sim.camera = cfg.sensors.camera.enabled;
    spec.sim.lidar = cfg.sensors.lidar.enabled;
    spec.sim.imu = cfg.sensors.imu.enabled;
    spec.sim.baro = cfg.sensors.barometer.enabled;
  }
  spec.sim.camera_hz = camera_hz;
  spec.sim.lidar_hz = lidar_hz;
  strataslam_sim::Simulator sim(spec.building, spec.trajectory, spec.sim);

  std::filesystem::create_directories(out);
  const std::string bag = out + "/bag";
  std::filesystem::remove_all(bag);
  rosbag2_cpp::Writer writer;
  writer.open(bag);
  const int64_t t0 = sim.stamp_at(0);

  // Static TF (omitted for the `no_tf` corruption). The topic is registered with the
  // transient-local QoS that TF listeners require, so that `ros2 bag play` republishes it
  // compatibly.
  if (corrupt != "no_tf") {
    rosbag2_storage::TopicMetadata md;
    md.name = "/tf_static";
    md.type = "tf2_msgs/msg/TFMessage";
    md.serialization_format = "cdr";
    md.offered_qos_profiles = {rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable()};
    writer.create_topic(md);
    tf2_msgs::msg::TFMessage tf;
    tf.transforms.push_back(
        static_tf(base, "camera_color_optical_frame", spec.sim.extrinsics.T_base_camera, t0));
    tf.transforms.push_back(static_tf(base, "lidar_link", spec.sim.extrinsics.T_base_lidar, t0));
    tf.transforms.push_back(static_tf(base, "imu_link", spec.sim.extrinsics.T_base_imu, t0));
    write(writer, "/tf_static", tf, t0);
  }
  const auto K = sim.intrinsics();
  using T = strataslam_sim::SensorEvent::Type;
  const double t_end = std::min(seconds, sim.trajectory().duration());
  for (const auto& ev : sim.events()) {
    if (sim.time_of(ev.stamp) > t_end) break;
    switch (ev.type) {
      case T::Imu: {
        auto m = sim.imu(ev.index);
        sm::Imu msg;
        msg.header.stamp = to_time(m.stamp);
        msg.header.frame_id = "imu_link";
        const double zsign = corrupt == "imu_z_flip" ? -1.0 : 1.0;
        msg.linear_acceleration.x = m.accel.x();
        msg.linear_acceleration.y = m.accel.y();
        msg.linear_acceleration.z = zsign * m.accel.z();
        msg.angular_velocity.x = m.gyro.x();
        msg.angular_velocity.y = m.gyro.y();
        msg.angular_velocity.z = m.gyro.z();
        write(writer, imu_t, msg, m.stamp);
        break;
      }
      case T::Baro: {
        const auto b = sim.baro(ev.index);
        sm::FluidPressure msg;
        msg.header.stamp = to_time(b.stamp);
        msg.header.frame_id = base;
        msg.fluid_pressure = b.pressure_pa;
        write(writer, baro_t, msg, b.stamp);
        break;
      }
      case T::Rgbd: {
        const auto f = sim.rgbd(ev.index);
        const int64_t rgb_stamp =
            f.stamp + (corrupt == "rgb_depth_offset" ? 50000000LL : 0);  // 50 ms
        if (corrupt != "no_camera_info") {
          sm::CameraInfo ci;
          ci.header.stamp = to_time(f.stamp);
          ci.header.frame_id = "camera_color_optical_frame";
          ci.width = K.width;
          ci.height = K.height;
          ci.distortion_model = "plumb_bob";
          ci.d.assign(5, 0.0);
          ci.k = {K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1};
          ci.p = {K.fx, 0, K.cx, 0, 0, K.fy, K.cy, 0, 0, 0, 1, 0};
          write(writer, info_t, ci, f.stamp);
        }
        sm::Image depth;
        depth.header.stamp = to_time(f.stamp);
        depth.header.frame_id = "camera_color_optical_frame";
        depth.width = K.width;
        depth.height = K.height;
        const bool f32 =
            corrupt != "depth_encoding";  // the corruption publishes an unsupported encoding
        depth.encoding = f32 ? "32FC1" : "8UC1";
        depth.step = K.width * (f32 ? 4 : 1);
        depth.data.resize(static_cast<size_t>(depth.step) * K.height);
        for (size_t i = 0; i < f.depth.meters.size(); ++i) {
          if (f32)
            std::memcpy(&depth.data[i * 4], &f.depth.meters[i], 4);
          else
            depth.data[i] = static_cast<uint8_t>(std::min(255.f, f.depth.meters[i] * 20.f));
        }
        write(writer, depth_t, depth, f.stamp);
        sm::Image
            rgb;  // a flat grey image: the simulator works at feature level (see FeatureFrame)
        rgb.header.stamp = to_time(rgb_stamp);
        rgb.header.frame_id = "camera_color_optical_frame";
        rgb.width = K.width;
        rgb.height = K.height;
        rgb.encoding = "mono8";
        rgb.step = K.width;
        rgb.data.assign(static_cast<size_t>(K.width) * K.height, 128);
        write(writer, rgb_t, rgb, rgb_stamp);
        strataslam_msgs::msg::FeatureFrame ff;
        ff.header.stamp = to_time(f.stamp);
        ff.header.frame_id = "camera_color_optical_frame";
        for (const auto& kp : f.keypoints) {
          ff.u.push_back(kp.u);
          ff.v.push_back(kp.v);
          ff.response.push_back(kp.response);
          ff.descriptors.insert(ff.descriptors.end(), kp.descriptor.begin(), kp.descriptor.end());
        }
        write(writer, "/strataslam/features", ff, f.stamp);
        break;
      }
      case T::Lidar: {
        const auto s = sim.lidar(ev.index);
        sm::PointCloud2 c;
        c.header.stamp = to_time(s.stamp);
        c.header.frame_id = "lidar_link";
        c.height = 1;
        c.width = static_cast<uint32_t>(s.points.size());
        c.point_step = 16;
        c.row_step = c.point_step * c.width;
        c.is_dense = true;
        for (const auto& [name, off] :
             {std::pair<const char*, int>{"x", 0}, {"y", 4}, {"z", 8}, {"t", 12}}) {
          sm::PointField pf;
          pf.name = name;
          pf.offset = off;
          pf.datatype = sm::PointField::FLOAT32;
          pf.count = 1;
          c.fields.push_back(pf);
        }
        c.data.resize(static_cast<size_t>(c.row_step));
        for (size_t i = 0; i < s.points.size(); ++i) {
          std::memcpy(&c.data[i * 16], s.points[i].data(), 12);
          std::memcpy(&c.data[i * 16 + 12], &s.time_offsets_s[i], 4);
        }
        write(writer, lidar_t, c, s.stamp);
        break;
      }
      case T::Odom:
        break;
    }
  }
  writer.close();
  // The ground truth in the bag's time base, under the name `strataslam eval --gt <dir>` reads
  // (before 2026-11-04 it was groundtruth.tum in simulation time, 1000 s away from every stamp of
  // the bag; audit AUD-058).
  std::ofstream(out + "/trajectory.tum") << sim.trajectory().to_tum(spec.sim.start_stamp_ns * 1e-9);
  // The vocabulary the node needs for visual loop closure (the simulator's random descriptors).
  const auto vr = strataslam_sim::sim_vocabulary()->save(out + "/vocab.bin");
  if (!vr) {
    std::cerr << vr.error().message << "\n";
    return 1;
  }
  std::cout << "wrote " << bag << "\n";
  return 0;
}
