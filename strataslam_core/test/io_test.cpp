#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>

#include "strataslam/common/version.hpp"
#include "strataslam/io/map_io.hpp"
#include "strataslam/io/point_cloud.hpp"

using namespace strataslam;
namespace fs = std::filesystem;

namespace {

class TempDir {
 public:
  TempDir() {
    path_ = fs::temp_directory_path() / ("strataslam_io_" + std::to_string(std::random_device{}()));
    fs::create_directories(path_);
  }
  ~TempDir() { fs::remove_all(path_); }
  std::string str() const { return path_.string(); }
  fs::path path() const { return path_; }

 private:
  fs::path path_;
};

PointCloud random_cloud(std::mt19937& rng, size_t n, bool colored) {
  std::uniform_real_distribution<float> d(-50.f, 50.f);
  PointCloud c;
  for (size_t i = 0; i < n; ++i) {
    c.points.emplace_back(d(rng), d(rng), d(rng));
    if (colored) c.colors.push_back({uint8_t(rng()), uint8_t(rng()), uint8_t(rng())});
  }
  return c;
}

FloorData make_floor(std::mt19937& rng, FloorId id) {
  std::uniform_real_distribution<double> d(-1e3, 1e3);
  FloorData f;
  f.metadata.id = id;
  f.metadata.gravity = Eigen::Vector3d(d(rng) * 1e-6, d(rng) * 1e-6, -1.0);
  f.metadata.reference_pressure_pa = 101325.123456789 - id * 35.5;
  f.metadata.sensors = {"camera", "imu"};
  f.metadata.sessions = {0, 1};
  f.metadata.alignment_residual_rad = 1.2345678901234567e-4;
  f.metadata.altitude_m = 3.0 * id + 0.0123456789;
  for (KeyframeId k = 0; k < 5; ++k) {
    KeyframeRecord kf;
    kf.id = k + 100 * id;
    kf.stamp_ns = 1700000000123456789LL + int64_t(k) * 33333333;
    kf.session = k % 2;
    kf.t = Eigen::Vector3d(d(rng), d(rng), d(rng));
    kf.q = Eigen::Quaterniond(d(rng), d(rng), d(rng), d(rng)).normalized();
    f.keyframes.push_back(kf);
    f.clouds[kf.id] = random_cloud(rng, 50, k % 2 == 0);
  }
  EdgeRecord e;
  e.type = "odometry";
  e.from = 100 * id;
  e.to = 100 * id + 1;
  e.measurement = {d(rng), d(rng), d(rng), 0.0, 0.0, 0.0, 1.0};
  e.info_dim = 6;
  for (int i = 0; i < 21; ++i) e.information.push_back(d(rng));
  f.edges.push_back(e);
  EdgeRecord b;
  b.type = "baro";
  b.from = 100 * id + 1;
  b.to = 100 * id + 2;
  b.measurement = {3.0000000000000004};
  b.info_dim = 1;
  b.information = {4.0};
  f.edges.push_back(b);

  VisualPayload v;
  v.descriptor_cols = 32;
  for (int i = 0; i < 4; ++i) v.keypoints.push_back({float(i), float(2 * i), 0.5f, 1.5f * i});
  v.descriptors.resize(4 * 32);
  for (auto& x : v.descriptors) x = uint8_t(rng());
  f.visual[100 * id] = v;

  LidarPayload l;
  l.rings = 3;
  l.sectors = 4;
  l.num_points = 12345;
  l.max_range = 80.f;
  l.scan_context.assign(12, 0.25f);
  l.ring_key = {1.f, 2.f, 3.f};
  f.lidar[100 * id + 1] = l;
  f.floor_cloud = random_cloud(rng, 200, false);
  return f;
}

MapData make_map(unsigned seed) {
  std::mt19937 rng(seed);
  MapData m;
  m.strataslam_version = kVersion;
  m.profile_yaml = "strataslam:\n  version: 1\n";
  m.floors.push_back(make_floor(rng, 0));
  m.floors.push_back(make_floor(rng, 1));
  TransitionRecord t;
  t.type = TransitionType::Elevator;
  t.from = 0;
  t.to = 1;
  t.kf_before = 4;
  t.kf_after = 100;
  t.delta_z = 3.0000000000000004;
  t.covariance = Eigen::Matrix4d::Identity() * 0.01;
  t.covariance(0, 3) = 1.0 / 3.0;
  t.continuous_tracking = false;
  t.relative = std::array<double, 4>{0.1234567890123, 1.0 / 3.0, -2.5, 3.0000000000000004};
  m.transitions.push_back(t);
  TransitionRecord u;
  u.type = TransitionType::Stairs;
  u.continuous_tracking = true;
  m.transitions.push_back(u);  // no delta_z
  return m;
}

}  // namespace

TEST(MapIo, FullRoundTripIsExact) {
  TempDir dir;
  const MapData original = make_map(7);
  const auto w = write_map(dir.str() + "/map", original);
  ASSERT_TRUE(w.ok()) << w.error().message;
  const auto r = read_map(dir.str() + "/map");
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_TRUE(r.value() == original);
}

TEST(MapIo, LayoutMatchesSpec) {
  TempDir dir;
  ASSERT_TRUE(write_map(dir.str(), make_map(1)).ok());
  for (const char* p : {"manifest.yaml", "profile.yaml", "transitions.yaml",
                        "floor_000/metadata.yaml", "floor_000/keyframes.csv", "floor_000/edges.csv",
                        "floor_000/clouds/0.pcd", "floor_000/visual/0.bin", "floor_000/lidar/1.bin",
                        "floor_000/floor_cloud.pcd", "floor_001/metadata.yaml"}) {
    EXPECT_TRUE(fs::exists(dir.path() / p)) << p;
  }
}

TEST(MapIo, MapsSavedBeforeTheRenameStillLoad) {
  // 0.2.x wrote `easyslam_version` in the manifest; nothing else in the format changed.
  TempDir dir;
  const MapData original = make_map(3);
  ASSERT_TRUE(write_map(dir.str(), original).ok());
  std::string manifest;
  {
    std::ifstream in(dir.path() / "manifest.yaml");
    manifest.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  const auto at = manifest.find("strataslam_version");
  ASSERT_NE(at, std::string::npos);
  manifest.replace(at, std::string("strataslam_version").size(), "easyslam_version");
  std::ofstream(dir.path() / "manifest.yaml") << manifest;
  const auto r = read_map(dir.str());
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_TRUE(r.value() == original);
}

TEST(MapIo, KeyframeCloudsWithoutAFrameAreThoseOfOlderMapsInTheSensorFrame) {
  TempDir dir;
  const MapData original = make_map(1);
  ASSERT_TRUE(write_map(dir.str(), original).ok());
  const fs::path meta = dir.path() / "floor_000" / "metadata.yaml";
  std::string text;
  {
    std::ifstream in(meta);
    text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  const auto at = text.find("cloud_frame: base");
  ASSERT_NE(at, std::string::npos);
  // Written before the field: the clouds are in the sensor frame.
  std::ofstream(meta) << text.substr(0, at) +
                             text.substr(at + std::string("cloud_frame: base").size());
  auto r = read_map(dir.str());
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_EQ(r.value().floors[0].metadata.cloud_frame, "sensor");
  // Any other frame is an error, not a guess.
  std::ofstream(meta) << text.substr(0, at) + "cloud_frame: lidar" +
                             text.substr(at + std::string("cloud_frame: base").size());
  r = read_map(dir.str());
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().message.find("cloud_frame"), std::string::npos) << r.error().message;
}

TEST(MapIo, TheBuildingAsTheOdometryLeftItIsSavedWithTheFloors) {
  TempDir dir;
  MapData m = make_map(5);
  m.floors[0].metadata.placement = Eigen::Vector4d(0, 0, 0, 0);
  m.floors[1].metadata.placement = Eigen::Vector4d(M_PI / 2, 1, 2, 3);
  ASSERT_TRUE(write_map(dir.str(), m).ok());
  const auto r = read_map(dir.str());
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_TRUE(r.value() == m);
  // global_cloud.pcd: every floor cloud placed by its placement (floor 1 turned 90 deg, moved).
  const auto global = read_pcd((dir.path() / "global_cloud.pcd").string());
  ASSERT_TRUE(global.ok());
  const auto& c0 = *m.floors[0].floor_cloud;
  const auto& c1 = *m.floors[1].floor_cloud;
  ASSERT_EQ(global.value().points.size(), c0.points.size() + c1.points.size());
  EXPECT_TRUE(global.value().points[0].isApprox(c0.points[0]));
  const Eigen::Vector3f p = c1.points[0];
  EXPECT_TRUE(global.value().points[c0.points.size()].isApprox(
      Eigen::Vector3f(-p.y() + 1, p.x() + 2, p.z() + 3), 1e-5f));
  // Saved again with a floor that has no placement: no building to place, and none left behind.
  m.floors[1].metadata.placement.reset();
  ASSERT_TRUE(write_map(dir.str(), m).ok());
  EXPECT_FALSE(fs::exists(dir.path() / "global_cloud.pcd"));
}

TEST(MapIo, FloorsWithoutAPlacementAreThoseOfOlderMaps) {
  TempDir dir;
  MapData original = make_map(1);
  original.floors[0].metadata.placement = Eigen::Vector4d(0.5, 1, 2, 3);
  ASSERT_TRUE(write_map(dir.str(), original).ok());
  const fs::path meta = dir.path() / "floor_000" / "metadata.yaml";
  std::string text;
  {
    std::ifstream in(meta);
    text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  const auto at = text.find("placement:");
  ASSERT_NE(at, std::string::npos);
  std::ofstream(meta) << text.substr(0, at);  // the last field: written before it existed
  auto r = read_map(dir.str());
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_FALSE(r.value().floors[0].metadata.placement);
  std::ofstream(meta) << text.substr(0, at) + "placement: [1, 2]\n";
  r = read_map(dir.str());
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().message.find("placement"), std::string::npos) << r.error().message;
}

TEST(MapIo, UnknownFormatVersionIsRejected) {
  TempDir dir;
  ASSERT_TRUE(write_map(dir.str(), make_map(1)).ok());
  std::ofstream(dir.path() / "manifest.yaml")
      << "strataslam_version: x\nformat_version: 99\nfloors: []\n";
  const auto r = read_map(dir.str());
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().message, "unsupported map format version 99 (supported: 1)");
}

TEST(MapIo, BinaryHeaderVersionAndMagicChecked) {
  TempDir dir;
  VisualPayload v;
  v.descriptor_cols = 1;
  v.keypoints = {{1, 2, 3, 4}};
  v.descriptors = {7};
  const std::string p = dir.str() + "/v.bin";
  ASSERT_TRUE(write_visual_payload(p, v).ok());
  {
    std::fstream f(p, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(4);
    const uint32_t bad = 42;
    f.write(reinterpret_cast<const char*>(&bad), 4);
  }
  auto r = read_visual_payload(p);
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().message.find("unsupported format version 42"), std::string::npos);
  {
    std::fstream f(p, std::ios::in | std::ios::out | std::ios::binary);
    f.write("XXXX", 4);
  }
  r = read_visual_payload(p);
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().message.find("bad magic"), std::string::npos);
}

TEST(MapIo, TruncatedPayloadIsAnError) {
  TempDir dir;
  LidarPayload l;
  l.rings = 2;
  l.sectors = 2;
  l.scan_context.assign(4, 1.f);
  l.ring_key = {1.f, 2.f};
  const std::string p = dir.str() + "/l.bin";
  ASSERT_TRUE(write_lidar_payload(p, l).ok());
  fs::resize_file(p, fs::file_size(p) - 3);
  EXPECT_FALSE(read_lidar_payload(p).ok());
}

TEST(MapIo, HostileFilesAreErrorsNotCrashes) {
  // Audit AUD-008: a keyframes.csv cut inside its last field used to abort in Eigen's comma
  // initializer; a cloud declaring 4e9 points with 3 bytes of data reserved 48 GB.
  TempDir dir;
  ASSERT_TRUE(write_map(dir.str(), make_map(1)).ok());
  const std::string kf = dir.str() + "/floor_000/keyframes.csv";
  std::ifstream in(kf);
  std::string header, row;
  std::getline(in, header);
  std::getline(in, row);
  in.close();
  for (const std::string& bad :
       {row.substr(0, row.rfind(',') + 1), row.substr(0, row.rfind(',')) + ",x", row + ",1"}) {
    std::ofstream(kf) << header << "\n" << bad << "\n";
    const auto r = read_map(dir.str());
    EXPECT_FALSE(r.ok()) << bad;
  }
  const std::string pcd = dir.str() + "/huge.pcd";
  std::ofstream(pcd) << "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 "
                        "1\nWIDTH 4000000000\n"
                        "HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 4000000000\nDATA binary\nabc";
  auto c = read_pcd(pcd);
  ASSERT_FALSE(c.ok());
  EXPECT_NE(c.error().message.find("truncated"), std::string::npos);
  std::ofstream(pcd) << "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 8 8 8\nTYPE F F F\nCOUNT 1 1 "
                        "1\nPOINTS 1\nDATA binary\n";
  EXPECT_FALSE(read_pcd(pcd).ok());  // doubles: not the layout this reader understands
  const std::string ply = dir.str() + "/huge.ply";
  std::ofstream(ply)
      << "ply\nformat binary_little_endian 1.0\nelement vertex 4000000000\nproperty float x\n"
         "property float y\nproperty float z\nend_header\nabc";
  EXPECT_FALSE(read_ply(ply).ok());
  std::ofstream(ply) << "ply\nformat binary_little_endian 1.0\nelement vertex lots\nproperty float "
                        "x\nend_header\n";
  EXPECT_FALSE(read_ply(ply).ok());  // a non-numeric count is an error, not an exception
  std::ofstream(ply) << "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty double "
                        "x\nproperty double y\n"
                        "property double z\nend_header\n";
  EXPECT_FALSE(read_ply(ply).ok());
}

TEST(MapIo, RejectsInvalidInputs) {
  TempDir dir;
  MapData m = make_map(1);
  m.floors[1].metadata.id = 0;
  EXPECT_FALSE(write_map(dir.str(), m).ok());
  m = make_map(1);
  m.floors[0].metadata.id = -1;
  EXPECT_FALSE(write_map(dir.str(), m).ok());
  EXPECT_FALSE(read_map(dir.str() + "/missing").ok());
}

TEST(Clouds, PcdAndPlyRoundTrip) {
  TempDir dir;
  std::mt19937 rng(3);
  for (bool colored : {false, true}) {
    const PointCloud c = random_cloud(rng, 1000, colored);
    ASSERT_TRUE(write_pcd(dir.str() + "/a.pcd", c).ok());
    ASSERT_TRUE(write_ply(dir.str() + "/a.ply", c).ok());
    const auto p = read_pcd(dir.str() + "/a.pcd");
    const auto y = read_ply(dir.str() + "/a.ply");
    ASSERT_TRUE(p.ok());
    ASSERT_TRUE(y.ok());
    EXPECT_TRUE(p.value() == c);
    EXPECT_TRUE(y.value() == c);
  }
}

TEST(Clouds, EmptyCloudAndMismatchedColors) {
  TempDir dir;
  PointCloud empty;
  ASSERT_TRUE(write_pcd(dir.str() + "/e.pcd", empty).ok());
  EXPECT_TRUE(read_pcd(dir.str() + "/e.pcd").value() == empty);
  PointCloud bad;
  bad.points.emplace_back(0, 0, 0);
  bad.colors = {{1, 2, 3}, {4, 5, 6}};
  EXPECT_FALSE(write_pcd(dir.str() + "/b.pcd", bad).ok());
  EXPECT_FALSE(write_ply(dir.str() + "/b.ply", bad).ok());
}

// A manifest is checked before any floor is read (audit AUD-035: 100 000 copies of one floor took
// the reader to 11.9 GB and bad_alloc), and its version must be a number (A6-07: missing, -1 or
// "abc" read as "version 0").
TEST(MapIo, HostileManifestsAreRejectedBeforeReadingFloors) {
  TempDir dir;
  ASSERT_TRUE(write_map(dir.str(), make_map(2)).ok());
  const auto with_manifest = [&](const std::string& text) {
    std::ofstream(dir.path() / "manifest.yaml") << text;
    return read_map(dir.str());
  };
  const std::string head = "strataslam_version: x\nformat_version: 1\n";
  std::string many = head + "floors:\n";
  for (int i = 0; i < 100000; ++i) many += "  - {id: 0, dir: floor_000}\n";
  auto r = with_manifest(many);
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().message.find("at most 1000"), std::string::npos) << r.error().message;
  r = with_manifest(head + "floors:\n  - {id: 0, dir: floor_000}\n  - {id: 1, dir: floor_000}\n");
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().message.find("listed twice"), std::string::npos) << r.error().message;
  r = with_manifest(head + "floors: {a: 1}\n");
  EXPECT_FALSE(r.ok());
  for (const char* version : {"", "-1", "abc", "1.5"}) {
    r = with_manifest(std::string("strataslam_version: x\n") +
                      (*version ? std::string("format_version: ") + version + "\n" : "") +
                      "floors: []\n");
    ASSERT_FALSE(r.ok()) << version;
    EXPECT_EQ(r.error().message.find("unsupported map format version 0"), std::string::npos)
        << r.error().message;
  }
  // A floor whose metadata says id -1 (the writer never writes one; A6-02).
  ASSERT_TRUE(write_map(dir.str(), make_map(2)).ok());
  std::string meta;
  {
    std::ifstream in(dir.path() / "floor_001" / "metadata.yaml");
    meta.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  const auto at = meta.find("id: 1");
  ASSERT_NE(at, std::string::npos) << meta;
  meta.replace(at, 5, "id: -1");
  std::ofstream(dir.path() / "floor_001" / "metadata.yaml") << meta;
  r = read_map(dir.str());
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().message.find("negative floor id"), std::string::npos) << r.error().message;
}
