#include <cstring>
#include <fstream>
#include <iterator>

#include "strataslam/io/payloads.hpp"

namespace strataslam {
namespace {

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "little-endian host required");

// "ESLM" dates from the name easyslam; kept so that maps saved before the rename still load.
constexpr char kMagic[4] = {'E', 'S', 'L', 'M'};
constexpr uint32_t kKindVisual = 1;
constexpr uint32_t kKindLidar = 2;

class ByteWriter {
 public:
  template <typename T>
  void put(const T& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    buf_.insert(buf_.end(), p, p + sizeof(T));
  }
  template <typename T>
  void put_array(const std::vector<T>& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(v.data());
    buf_.insert(buf_.end(), p, p + v.size() * sizeof(T));
  }
  const std::vector<uint8_t>& bytes() const { return buf_; }

 private:
  std::vector<uint8_t> buf_;
};

class ByteReader {
 public:
  explicit ByteReader(const std::vector<uint8_t>& b) : b_(b) {}
  template <typename T>
  bool get(T& v) {
    if (pos_ + sizeof(T) > b_.size()) return false;
    std::memcpy(&v, b_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return true;
  }
  template <typename T>
  bool get_array(std::vector<T>& v, size_t n) {
    if (n > (b_.size() - pos_) / sizeof(T)) return false;
    v.resize(n);
    if (n == 0)
      return true;  // memcpy from/to a null pointer is undefined even for 0 bytes (audit A6-03)
    std::memcpy(v.data(), b_.data() + pos_, n * sizeof(T));
    pos_ += n * sizeof(T);
    return true;
  }
  bool at_end() const { return pos_ == b_.size(); }

 private:
  const std::vector<uint8_t>& b_;
  size_t pos_ = 0;
};

Result<void> write_container(const std::string& path, uint32_t kind, const ByteWriter& payload) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return Error{path + ": cannot open for writing"};
  const uint32_t version = kMapFormatVersion;
  const uint64_t size = payload.bytes().size();
  out.write(kMagic, 4);
  out.write(reinterpret_cast<const char*>(&version), 4);
  out.write(reinterpret_cast<const char*>(&kind), 4);
  out.write(reinterpret_cast<const char*>(&size), 8);
  out.write(reinterpret_cast<const char*>(payload.bytes().data()),
            static_cast<std::streamsize>(size));
  if (!out) return Error{path + ": write failed"};
  return {};
}

Result<std::vector<uint8_t>> read_container(const std::string& path, uint32_t kind) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return Error{path + ": cannot open"};
  char magic[4];
  uint32_t version = 0, k = 0;
  uint64_t size = 0;
  in.read(magic, 4);
  in.read(reinterpret_cast<char*>(&version), 4);
  in.read(reinterpret_cast<char*>(&k), 4);
  in.read(reinterpret_cast<char*>(&size), 8);
  if (!in || std::memcmp(magic, kMagic, 4) != 0) return Error{path + ": bad magic number"};
  if (version != kMapFormatVersion) {
    return Error{path + ": unsupported format version " + std::to_string(version) +
                 " (supported: " + std::to_string(kMapFormatVersion) + ")"};
  }
  if (k != kind) return Error{path + ": unexpected payload kind " + std::to_string(k)};
  std::vector<uint8_t> data(std::istreambuf_iterator<char>(in), {});
  if (data.size() != size) return Error{path + ": payload size mismatch"};
  return data;
}

}  // namespace

Result<void> write_visual_payload(const std::string& path, const VisualPayload& p) {
  if (p.descriptors.size() != p.keypoints.size() * p.descriptor_cols) {
    return Error{path + ": descriptor buffer does not match keypoints"};
  }
  ByteWriter w;
  w.put<uint32_t>(static_cast<uint32_t>(p.keypoints.size()));
  w.put<uint32_t>(p.descriptor_cols);
  for (const auto& k : p.keypoints) {
    w.put(k.x);
    w.put(k.y);
    w.put(k.response);
    w.put(k.depth);
  }
  w.put_array(p.descriptors);
  return write_container(path, kKindVisual, w);
}

Result<VisualPayload> read_visual_payload(const std::string& path) {
  auto raw = read_container(path, kKindVisual);
  if (!raw) return raw.error();
  ByteReader r(raw.value());
  VisualPayload p;
  uint32_t n = 0;
  if (!r.get(n) || !r.get(p.descriptor_cols)) return Error{path + ": truncated payload"};
  p.keypoints.resize(0);
  for (uint32_t i = 0; i < n; ++i) {
    VisualKeypoint k;
    if (!r.get(k.x) || !r.get(k.y) || !r.get(k.response) || !r.get(k.depth)) {
      return Error{path + ": truncated payload"};
    }
    p.keypoints.push_back(k);
  }
  if (!r.get_array(p.descriptors, static_cast<size_t>(n) * p.descriptor_cols) || !r.at_end()) {
    return Error{path + ": malformed payload"};
  }
  return p;
}

Result<void> write_lidar_payload(const std::string& path, const LidarPayload& p) {
  if (p.scan_context.size() != static_cast<size_t>(p.rings) * p.sectors ||
      p.ring_key.size() != p.rings) {
    return Error{path + ": scan context dimensions inconsistent"};
  }
  ByteWriter w;
  w.put(p.rings);
  w.put(p.sectors);
  w.put(p.num_points);
  w.put(p.max_range);
  w.put_array(p.scan_context);
  w.put_array(p.ring_key);
  return write_container(path, kKindLidar, w);
}

Result<LidarPayload> read_lidar_payload(const std::string& path) {
  auto raw = read_container(path, kKindLidar);
  if (!raw) return raw.error();
  ByteReader r(raw.value());
  LidarPayload p;
  if (!r.get(p.rings) || !r.get(p.sectors) || !r.get(p.num_points) || !r.get(p.max_range)) {
    return Error{path + ": truncated payload"};
  }
  if (!r.get_array(p.scan_context, static_cast<size_t>(p.rings) * p.sectors) ||
      !r.get_array(p.ring_key, p.rings) || !r.at_end()) {
    return Error{path + ": malformed payload"};
  }
  return p;
}

}  // namespace strataslam
