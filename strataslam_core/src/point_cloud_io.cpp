#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "strataslam/io/point_cloud.hpp"

namespace strataslam {
namespace {

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "little-endian host required");

uint32_t pack_rgb(const std::array<uint8_t, 3>& c) {
  return (uint32_t(c[0]) << 16) | (uint32_t(c[1]) << 8) | uint32_t(c[2]);
}

Error bad(const std::string& path, const std::string& what) { return Error{path + ": " + what}; }

}  // namespace

Result<void> write_pcd(const std::string& path, const PointCloud& cloud) {
  if (!cloud.colors.empty() && cloud.colors.size() != cloud.points.size()) {
    return bad(path, "colors and points differ in size");
  }
  const bool rgb = !cloud.colors.empty();
  std::ofstream out(path, std::ios::binary);
  if (!out) return bad(path, "cannot open for writing");
  const size_t n = cloud.points.size();
  out << "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
      << (rgb ? "FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n"
              : "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
      << "WIDTH " << n << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS " << n << "\nDATA binary\n";
  for (size_t i = 0; i < n; ++i) {
    out.write(reinterpret_cast<const char*>(cloud.points[i].data()), 12);
    if (rgb) {
      const uint32_t v = pack_rgb(cloud.colors[i]);
      out.write(reinterpret_cast<const char*>(&v), 4);
    }
  }
  if (!out) return bad(path, "write failed");
  return {};
}

Result<PointCloud> read_pcd(const std::string& path) {
  // Reads the binary layout write_pcd produces (x y z [rgb], 4-byte fields, one value each). The
  // declared point count is checked against the bytes in the file before anything is allocated.
  std::ifstream in(path, std::ios::binary);
  if (!in) return bad(path, "cannot open");
  std::string line;
  bool rgb = false, binary = false, have_points = false;
  int nfields = 0;
  uint64_t n = 0;
  std::string size, type, count;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key == "FIELDS") {
      std::string f;
      std::vector<std::string> fields;
      while (ls >> f) fields.push_back(f);
      nfields = static_cast<int>(fields.size());
      if (nfields == 3 && fields[0] == "x" && fields[1] == "y" && fields[2] == "z") {
      } else if (nfields == 4 && fields[0] == "x" && fields[1] == "y" && fields[2] == "z" &&
                 fields[3] == "rgb") {
        rgb = true;
      } else {
        return bad(path, "unsupported PCD fields");
      }
    } else if (key == "SIZE" || key == "TYPE" || key == "COUNT") {
      std::string rest;
      std::getline(ls >> std::ws, rest);
      (key == "SIZE" ? size : key == "TYPE" ? type : count) = rest;
    } else if (key == "POINTS") {
      if (!(ls >> n)) return bad(path, "invalid POINTS");
      have_points = true;
    } else if (key == "DATA") {
      std::string kind;
      ls >> kind;
      if (kind != "binary") return bad(path, "only binary PCD is supported");
      binary = true;
      break;
    }
  }
  if (!binary || nfields == 0 || !have_points) return bad(path, "invalid PCD header");
  if (size != (rgb ? "4 4 4 4" : "4 4 4") || type != (rgb ? "F F F U" : "F F F") ||
      count != (rgb ? "1 1 1 1" : "1 1 1")) {
    return bad(path, "unsupported PCD field layout (SIZE/TYPE/COUNT)");
  }
  const uint64_t stride = rgb ? 16 : 12;
  const std::streamoff start = in.tellg();
  in.seekg(0, std::ios::end);
  const uint64_t available = static_cast<uint64_t>(in.tellg() - start);
  in.seekg(start);
  if (n > available / stride)
    return bad(path, "truncated PCD data (" + std::to_string(n) + " points declared)");
  PointCloud cloud;
  cloud.points.resize(n);
  if (rgb) cloud.colors.resize(n);
  for (size_t i = 0; i < n; ++i) {
    in.read(reinterpret_cast<char*>(cloud.points[i].data()), 12);
    if (rgb) {
      uint32_t v = 0;
      in.read(reinterpret_cast<char*>(&v), 4);
      cloud.colors[i] = {uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)};
    }
  }
  if (!in) return bad(path, "truncated PCD data");
  return cloud;
}

Result<void> write_ply(const std::string& path, const PointCloud& cloud) {
  if (!cloud.colors.empty() && cloud.colors.size() != cloud.points.size()) {
    return bad(path, "colors and points differ in size");
  }
  const bool rgb = !cloud.colors.empty();
  std::ofstream out(path, std::ios::binary);
  if (!out) return bad(path, "cannot open for writing");
  out << "ply\nformat binary_little_endian 1.0\nelement vertex " << cloud.points.size()
      << "\nproperty float x\nproperty float y\nproperty float z\n";
  if (rgb) out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  out << "end_header\n";
  for (size_t i = 0; i < cloud.points.size(); ++i) {
    out.write(reinterpret_cast<const char*>(cloud.points[i].data()), 12);
    if (rgb) out.write(reinterpret_cast<const char*>(cloud.colors[i].data()), 3);
  }
  if (!out) return bad(path, "write failed");
  return {};
}

Result<PointCloud> read_ply(const std::string& path) {
  // Reads the binary little-endian layout write_ply produces: float x y z, optionally uchar red
  // green blue. Anything else is rejected rather than misread.
  std::ifstream in(path, std::ios::binary);
  if (!in) return bad(path, "cannot open");
  std::string line;
  uint64_t n = 0;
  bool binary = false, have_vertex = false;
  std::vector<std::string> props;
  std::getline(in, line);
  if (line != "ply") return bad(path, "not a PLY file");
  while (std::getline(in, line)) {
    if (line == "end_header") {
      binary = true;
      break;
    }
    std::istringstream ls(line);
    std::string a, b, c;
    ls >> a >> b >> c;
    if (a == "format" && b != "binary_little_endian") return bad(path, "unsupported PLY format");
    if (a == "element") {
      if (b != "vertex" || have_vertex) return bad(path, "unsupported PLY element \"" + b + "\"");
      std::istringstream ns(c);
      if (!(ns >> n) || !ns.eof()) return bad(path, "invalid vertex count");
      have_vertex = true;
    }
    if (a == "property") props.push_back(b.append(" ").append(c));
  }
  if (!binary || !have_vertex) return bad(path, "invalid PLY header");
  const std::vector<std::string> xyz = {"float x", "float y", "float z"};
  std::vector<std::string> xyz_rgb = xyz;
  xyz_rgb.insert(xyz_rgb.end(), {"uchar red", "uchar green", "uchar blue"});
  if (props != xyz && props != xyz_rgb) return bad(path, "unsupported PLY vertex properties");
  const bool rgb = props == xyz_rgb;
  const uint64_t stride = rgb ? 15 : 12;
  const std::streamoff start = in.tellg();
  in.seekg(0, std::ios::end);
  const uint64_t available = static_cast<uint64_t>(in.tellg() - start);
  in.seekg(start);
  if (n > available / stride)
    return bad(path, "truncated PLY data (" + std::to_string(n) + " vertices declared)");
  PointCloud cloud;
  cloud.points.resize(n);
  if (rgb) cloud.colors.resize(n);
  for (size_t i = 0; i < n; ++i) {
    in.read(reinterpret_cast<char*>(cloud.points[i].data()), 12);
    if (rgb) in.read(reinterpret_cast<char*>(cloud.colors[i].data()), 3);
  }
  if (!in) return bad(path, "truncated PLY data");
  return cloud;
}

}  // namespace strataslam
