#include "strataslam/io/map_io.hpp"

#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <sstream>

#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace strataslam {
namespace {

// Exceptions are used only inside this file and converted to Result at the API boundary.
struct IoError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

std::string fmt_double(double v) {
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof(buf), v);  // shortest round-trip form
  return std::string(buf, r.ptr);
}

double parse_double(const std::string& s, const std::string& where) {
  double v = 0;
  const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
  if (r.ec != std::errc() || r.ptr != s.data() + s.size()) {
    throw IoError(where + ": invalid number \"" + s + "\"");
  }
  return v;
}

uint64_t parse_u64(const std::string& s, const std::string& where) {
  uint64_t v = 0;
  const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
  if (r.ec != std::errc() || r.ptr != s.data() + s.size()) {
    throw IoError(where + ": invalid integer \"" + s + "\"");
  }
  return v;
}

int64_t parse_i64(const std::string& s, const std::string& where) {
  int64_t v = 0;
  const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
  if (r.ec != std::errc() || r.ptr != s.data() + s.size()) {
    throw IoError(where + ": invalid integer \"" + s + "\"");
  }
  return v;
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  std::istringstream ss(s);
  while (std::getline(ss, cur, sep)) out.push_back(cur);
  if (!s.empty() && s.back() == sep) out.emplace_back();
  return out;
}

std::string join_doubles(const std::vector<double>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? ";" : "") + fmt_double(v[i]);
  return s;
}

std::string floor_dir_name(FloorId id) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "floor_%03d", id);
  return buf;
}

std::string read_text(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw IoError(p.string() + ": cannot open");
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void write_text(const fs::path& p, const std::string& text) {
  std::ofstream out(p, std::ios::binary);
  if (!out) throw IoError(p.string() + ": cannot open for writing");
  out << text;
  if (!out) throw IoError(p.string() + ": write failed");
}

void check(const Result<void>& r) {
  if (!r) throw IoError(r.error().message);
}

template <typename T>
T unwrap(Result<T> r) {
  if (!r) throw IoError(r.error().message);
  return std::move(r.value());
}

void setup_emitter(YAML::Emitter& out) { out.SetDoublePrecision(17); }

void emit_vec3(YAML::Emitter& out, const Eigen::Vector3d& v) {
  out << YAML::Flow << YAML::BeginSeq << v.x() << v.y() << v.z() << YAML::EndSeq;
}

Eigen::Vector3d read_vec3(const YAML::Node& n, const std::string& where) {
  if (!n.IsSequence() || n.size() != 3) throw IoError(where + ": expected 3 numbers");
  return {n[0].as<double>(), n[1].as<double>(), n[2].as<double>()};
}

// ---- transitions.yaml ----

std::string transitions_to_yaml(const std::vector<TransitionRecord>& list) {
  YAML::Emitter out;
  setup_emitter(out);
  out << YAML::BeginMap << YAML::Key << "transitions" << YAML::Value << YAML::BeginSeq;
  for (const auto& t : list) {
    out << YAML::BeginMap;
    out << YAML::Key << "type" << YAML::Value << to_string(t.type);
    out << YAML::Key << "from" << YAML::Value << t.from;
    out << YAML::Key << "to" << YAML::Value << t.to;
    out << YAML::Key << "kf_before" << YAML::Value << t.kf_before;
    out << YAML::Key << "kf_after" << YAML::Value << t.kf_after;
    out << YAML::Key << "delta_z" << YAML::Value;
    if (t.delta_z) out << *t.delta_z; else out << YAML::Null;
    out << YAML::Key << "covariance" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) out << t.covariance(r, c);
    out << YAML::EndSeq;
    out << YAML::Key << "continuous_tracking" << YAML::Value << t.continuous_tracking;
    out << YAML::Key << "relative" << YAML::Value;
    if (t.relative) {
      out << YAML::Flow << YAML::BeginSeq;
      for (double v : *t.relative) out << v;
      out << YAML::EndSeq;
    } else {
      out << YAML::Null;
    }
    out << YAML::EndMap;
  }
  out << YAML::EndSeq << YAML::EndMap;
  return out.c_str();
}

std::vector<TransitionRecord> transitions_from_yaml(const std::string& text) {
  const YAML::Node root = YAML::Load(text);
  std::vector<TransitionRecord> list;
  for (const auto& n : root["transitions"]) {
    TransitionRecord t;
    if (!parse_transition_type(n["type"].as<std::string>(), &t.type)) {
      throw IoError("transitions.yaml: unknown transition type");
    }
    t.from = n["from"].as<FloorId>();
    t.to = n["to"].as<FloorId>();
    t.kf_before = n["kf_before"].as<KeyframeId>();
    t.kf_after = n["kf_after"].as<KeyframeId>();
    if (!n["delta_z"].IsNull()) t.delta_z = n["delta_z"].as<double>();
    const YAML::Node cov = n["covariance"];
    if (!cov.IsSequence() || cov.size() != 16) throw IoError("transitions.yaml: covariance needs 16 values");
    for (int i = 0; i < 16; ++i) t.covariance(i / 4, i % 4) = cov[i].as<double>();
    t.continuous_tracking = n["continuous_tracking"].as<bool>();
    if (n["relative"].IsDefined() && !n["relative"].IsNull()) {
      const YAML::Node r = n["relative"];
      if (!r.IsSequence() || r.size() != 4) throw IoError("transitions.yaml: relative needs 4 values");
      t.relative = std::array<double, 4>{r[0].as<double>(), r[1].as<double>(), r[2].as<double>(), r[3].as<double>()};
    }
    list.push_back(t);
  }
  return list;
}

// ---- floor metadata ----

std::string metadata_to_yaml(const FloorMetadata& m) {
  YAML::Emitter out;
  setup_emitter(out);
  out << YAML::BeginMap;
  out << YAML::Key << "floor_id" << YAML::Value << m.id;
  out << YAML::Key << "gravity" << YAML::Value;
  emit_vec3(out, m.gravity);
  out << YAML::Key << "ground_normal" << YAML::Value;
  emit_vec3(out, m.ground_normal);
  out << YAML::Key << "reference_pressure_pa" << YAML::Value;
  if (m.reference_pressure_pa) out << *m.reference_pressure_pa; else out << YAML::Null;
  out << YAML::Key << "sensors" << YAML::Value << YAML::Flow << YAML::BeginSeq;
  for (const auto& s : m.sensors) out << s;
  out << YAML::EndSeq;
  out << YAML::Key << "sessions" << YAML::Value << YAML::Flow << YAML::BeginSeq;
  for (auto s : m.sessions) out << s;
  out << YAML::EndSeq;
  out << YAML::Key << "alignment_residual_rad" << YAML::Value << m.alignment_residual_rad;
  out << YAML::Key << "altitude_m" << YAML::Value;
  if (m.altitude_m) out << *m.altitude_m; else out << YAML::Null;
  out << YAML::EndMap;
  return out.c_str();
}

FloorMetadata metadata_from_yaml(const std::string& text) {
  const YAML::Node n = YAML::Load(text);
  FloorMetadata m;
  m.id = n["floor_id"].as<FloorId>();
  m.gravity = read_vec3(n["gravity"], "metadata.gravity");
  m.ground_normal = read_vec3(n["ground_normal"], "metadata.ground_normal");
  if (!n["reference_pressure_pa"].IsNull()) {
    m.reference_pressure_pa = n["reference_pressure_pa"].as<double>();
  }
  for (const auto& s : n["sensors"]) m.sensors.push_back(s.as<std::string>());
  for (const auto& s : n["sessions"]) m.sessions.push_back(s.as<SessionId>());
  m.alignment_residual_rad = n["alignment_residual_rad"].as<double>();
  if (n["altitude_m"].IsDefined() && !n["altitude_m"].IsNull()) m.altitude_m = n["altitude_m"].as<double>();
  return m;
}

// ---- CSV ----

constexpr const char* kKeyframesHeader = "id,stamp_ns,session,x,y,z,qx,qy,qz,qw";
constexpr const char* kEdgesHeader = "type,from,to,measurement,info_dim,information";

std::string keyframes_to_csv(const std::vector<KeyframeRecord>& kfs) {
  std::string s = std::string(kKeyframesHeader) + "\n";
  for (const auto& k : kfs) {
    s += std::to_string(k.id) + "," + std::to_string(k.stamp_ns) + "," +
         std::to_string(k.session);
    for (double v : {k.t.x(), k.t.y(), k.t.z(), k.q.x(), k.q.y(), k.q.z(), k.q.w()}) {
      s += "," + fmt_double(v);
    }
    s += "\n";
  }
  return s;
}

std::vector<KeyframeRecord> keyframes_from_csv(const std::string& text, const std::string& where) {
  std::istringstream in(text);
  std::string line;
  std::getline(in, line);
  if (line != kKeyframesHeader) throw IoError(where + ": unexpected header");
  std::vector<KeyframeRecord> out;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto f = split(line, ',');
    if (f.size() != 10) throw IoError(where + ": expected 10 columns");
    KeyframeRecord k;
    k.id = parse_u64(f[0], where);
    k.stamp_ns = parse_i64(f[1], where);
    k.session = static_cast<SessionId>(parse_u64(f[2], where));
    k.t = {parse_double(f[3], where), parse_double(f[4], where), parse_double(f[5], where)};
    // Parsed before assigning: a parse error thrown inside Eigen's comma initializer would abort.
    const double qx = parse_double(f[6], where), qy = parse_double(f[7], where), qz = parse_double(f[8], where),
                 qw = parse_double(f[9], where);
    k.q = Eigen::Quaterniond(qw, qx, qy, qz);
    out.push_back(k);
  }
  return out;
}

std::string edges_to_csv(const std::vector<EdgeRecord>& edges) {
  std::string s = std::string(kEdgesHeader) + "\n";
  for (const auto& e : edges) {
    s += e.type + "," + std::to_string(e.from) + "," + std::to_string(e.to) + "," +
         join_doubles(e.measurement) + "," + std::to_string(e.info_dim) + "," +
         join_doubles(e.information) + "\n";
  }
  return s;
}

std::vector<EdgeRecord> edges_from_csv(const std::string& text, const std::string& where) {
  std::istringstream in(text);
  std::string line;
  std::getline(in, line);
  if (line != kEdgesHeader) throw IoError(where + ": unexpected header");
  std::vector<EdgeRecord> out;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto f = split(line, ',');
    if (f.size() != 6) throw IoError(where + ": expected 6 columns");
    EdgeRecord e;
    e.type = f[0];
    e.from = parse_u64(f[1], where);
    e.to = parse_u64(f[2], where);
    for (const auto& v : split(f[3], ';')) e.measurement.push_back(parse_double(v, where));
    e.info_dim = static_cast<uint32_t>(parse_u64(f[4], where));
    for (const auto& v : split(f[5], ';')) e.information.push_back(parse_double(v, where));
    if (e.information.size() != size_t(e.info_dim) * (e.info_dim + 1) / 2) {
      throw IoError(where + ": information size does not match info_dim");
    }
    out.push_back(e);
  }
  return out;
}

// ---- floors ----

void write_floor(const fs::path& dir, const FloorData& f) {
  fs::create_directories(dir / "clouds");
  fs::create_directories(dir / "visual");
  fs::create_directories(dir / "lidar");
  write_text(dir / "metadata.yaml", metadata_to_yaml(f.metadata));
  write_text(dir / "keyframes.csv", keyframes_to_csv(f.keyframes));
  write_text(dir / "edges.csv", edges_to_csv(f.edges));
  for (const auto& [id, cloud] : f.clouds) {
    check(write_pcd((dir / "clouds" / (std::to_string(id) + ".pcd")).string(), cloud));
  }
  for (const auto& [id, p] : f.visual) {
    check(write_visual_payload((dir / "visual" / (std::to_string(id) + ".bin")).string(), p));
  }
  for (const auto& [id, p] : f.lidar) {
    check(write_lidar_payload((dir / "lidar" / (std::to_string(id) + ".bin")).string(), p));
  }
  if (f.floor_cloud) check(write_pcd((dir / "floor_cloud.pcd").string(), *f.floor_cloud));
}

KeyframeId id_from_stem(const fs::path& p) {
  return parse_u64(p.stem().string(), p.string());
}

FloorData read_floor(const fs::path& dir) {
  FloorData f;
  f.metadata = metadata_from_yaml(read_text(dir / "metadata.yaml"));
  f.keyframes = keyframes_from_csv(read_text(dir / "keyframes.csv"), (dir / "keyframes.csv").string());
  f.edges = edges_from_csv(read_text(dir / "edges.csv"), (dir / "edges.csv").string());
  if (fs::exists(dir / "clouds")) {
    for (const auto& e : fs::directory_iterator(dir / "clouds")) {
      if (e.path().extension() == ".pcd") f.clouds[id_from_stem(e.path())] = unwrap(read_pcd(e.path().string()));
    }
  }
  if (fs::exists(dir / "visual")) {
    for (const auto& e : fs::directory_iterator(dir / "visual")) {
      if (e.path().extension() == ".bin") f.visual[id_from_stem(e.path())] = unwrap(read_visual_payload(e.path().string()));
    }
  }
  if (fs::exists(dir / "lidar")) {
    for (const auto& e : fs::directory_iterator(dir / "lidar")) {
      if (e.path().extension() == ".bin") f.lidar[id_from_stem(e.path())] = unwrap(read_lidar_payload(e.path().string()));
    }
  }
  if (fs::exists(dir / "floor_cloud.pcd")) {
    f.floor_cloud = unwrap(read_pcd((dir / "floor_cloud.pcd").string()));
  }
  return f;
}

}  // namespace

Result<void> write_map(const std::string& dir, const MapData& map) {
  try {
    if (map.format_version != kMapFormatVersion) {
      return Error{"cannot write format version " + std::to_string(map.format_version)};
    }
    std::set<FloorId> seen;
    for (const auto& f : map.floors) {
      if (f.metadata.id < 0) return Error{"floor ids must be >= 0"};
      if (!seen.insert(f.metadata.id).second) return Error{"duplicate floor id"};
    }
    const fs::path root(dir);
    fs::create_directories(root);
    write_text(root / "profile.yaml", map.profile_yaml);
    write_text(root / "transitions.yaml", transitions_to_yaml(map.transitions));
    for (const auto& f : map.floors) write_floor(root / floor_dir_name(f.metadata.id), f);

    YAML::Emitter out;
  setup_emitter(out);
    out << YAML::BeginMap;
    out << YAML::Key << "strataslam_version" << YAML::Value << map.strataslam_version;
    out << YAML::Key << "format_version" << YAML::Value << map.format_version;
    out << YAML::Key << "profile" << YAML::Value << "profile.yaml";
    out << YAML::Key << "transitions" << YAML::Value << "transitions.yaml";
    out << YAML::Key << "floors" << YAML::Value << YAML::BeginSeq;
    for (const auto& f : map.floors) {
      out << YAML::BeginMap << YAML::Key << "id" << YAML::Value << f.metadata.id << YAML::Key
          << "dir" << YAML::Value << floor_dir_name(f.metadata.id) << YAML::Key << "keyframes"
          << YAML::Value << f.keyframes.size() << YAML::EndMap;
    }
    out << YAML::EndSeq << YAML::EndMap;
    write_text(root / "manifest.yaml", out.c_str());  // written last: marks a complete map
    return {};
  } catch (const std::exception& e) {
    return Error{e.what()};
  }
}

/// More floors than any building has: a manifest listing more is corrupt or hostile.
constexpr size_t kMaxFloors = 1000;

Result<MapData> read_map(const std::string& dir) {
  try {
    const fs::path root(dir);
    const YAML::Node manifest = YAML::Load(read_text(root / "manifest.yaml"));
    if (!manifest.IsMap()) return Error{"manifest: not a YAML map"};
    const YAML::Node v = manifest["format_version"];
    if (!v.IsDefined() || !v.IsScalar()) return Error{"manifest: format_version is missing"};
    uint32_t version = 0;
    try {
      const long long n = std::stoll(v.Scalar());
      if (n < 0 || std::to_string(n) != v.Scalar()) throw std::invalid_argument("not a version");
      version = static_cast<uint32_t>(n);
    } catch (const std::exception&) {
      return Error{"manifest: format_version \"" + v.Scalar() + "\" is not a version number"};
    }
    if (version != kMapFormatVersion) {
      return Error{"unsupported map format version " + std::to_string(version) +
                   " (supported: " + std::to_string(kMapFormatVersion) + ")"};
    }
    MapData map;
    map.format_version = version;
    // Maps saved before the rename (0.2.x) carry `easyslam_version`; the format is otherwise the same.
    const YAML::Node writer = manifest["strataslam_version"].IsDefined() ? manifest["strataslam_version"] : manifest["easyslam_version"];
    map.strataslam_version = writer.as<std::string>();
    map.profile_yaml = read_text(root / "profile.yaml");
    map.transitions = transitions_from_yaml(read_text(root / "transitions.yaml"));
    // A manifest is checked before any floor is read: a hostile or corrupt one listing the same floor 100 000 times
    // made the reader load it that many times (11.9 GB, then bad_alloc; audit AUD-035).
    const YAML::Node floors = manifest["floors"];
    if (floors.IsDefined() && !floors.IsSequence()) return Error{"manifest: floors must be a list"};
    if (floors.size() > kMaxFloors) return Error{"manifest: " + std::to_string(floors.size()) + " floors (at most " + std::to_string(kMaxFloors) + ")"};
    std::set<std::string> dirs;
    for (const auto& f : floors) {
      const std::string name = f["dir"].as<std::string>();
      if (name.empty() || name.find('/') != std::string::npos || name.find("..") != std::string::npos) {
        return Error{"manifest: invalid floor directory \"" + name + "\""};
      }
      if (!dirs.insert(name).second) return Error{"manifest: floor directory \"" + name + "\" listed twice"};
    }
    std::set<FloorId> ids;
    for (const auto& f : floors) {
      map.floors.push_back(read_floor(root / f["dir"].as<std::string>()));
      const FloorId id = map.floors.back().metadata.id;
      if (id < 0) return Error{"floor " + f["dir"].as<std::string>() + ": negative floor id " + std::to_string(id)};
      if (!ids.insert(id).second) return Error{"floor " + f["dir"].as<std::string>() + ": floor id " + std::to_string(id) + " used twice"};
    }
    return map;
  } catch (const std::exception& e) {
    return Error{e.what()};
  }
}

}  // namespace strataslam
