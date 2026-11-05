#include "strataslam/io/image_io.hpp"

#include <jpeglib.h>
#include <png.h>

#include <cctype>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>

namespace strataslam {

namespace {

using FilePtr = std::unique_ptr<FILE, int (*)(FILE*)>;

FilePtr open_file(const std::string& path) {
  return FilePtr(std::fopen(path.c_str(), "rb"), &std::fclose);
}

std::string lower_extension(const std::string& path) {
  const auto dot = path.find_last_of('.');
  std::string ext = dot == std::string::npos ? "" : path.substr(dot);
  for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return ext;
}

// ---- PNG ----

struct PngRead {
  png_structp png = nullptr;
  png_infop info = nullptr;
  ~PngRead() { png_destroy_read_struct(&png, info ? &info : nullptr, nullptr); }
};

void png_error_fn(png_structp png, png_const_charp msg) {
  auto* out = static_cast<std::string*>(png_get_error_ptr(png));
  if (out) *out = msg;
  png_longjmp(png, 1);
}
void png_warning_fn(png_structp, png_const_charp) {}

// The two phases that libpng may abort with longjmp run in functions holding no C++ object that
// changes after setjmp (only raw pointers and PODs), so a jump back never leaves a vector or string
// in an indeterminate state.

/// Reads the header and sets up the conversion to one channel of `bit_depth` bits. Colour is
/// converted to gray by libpng with the BT.601 weights (what OpenCV asks libpng for when reading
/// with IMREAD_GRAYSCALE); alpha is dropped.
bool png_header(png_structp png, png_infop info, FILE* f, int bit_depth, size_t sample_bytes,
                png_uint_32* w, png_uint_32* h, const char** problem) {
  if (setjmp(png_jmpbuf(png))) return false;
  png_init_io(png, f);
  png_set_sig_bytes(png, 8);
  png_read_info(png, info);
  *w = png_get_image_width(png, info);
  *h = png_get_image_height(png, info);
  const int color = png_get_color_type(png, info);
  const int depth = png_get_bit_depth(png, info);
  if (*w == 0 || *h == 0 || *w > 65535 || *h > 65535) {
    *problem = "unsupported PNG size";
    return false;
  }
  const bool trns = png_get_valid(png, info, PNG_INFO_tRNS) != 0;
  if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
  if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
  if (trns) png_set_tRNS_to_alpha(png);
  if ((color & PNG_COLOR_MASK_ALPHA) || trns) png_set_strip_alpha(png);
  if ((color & PNG_COLOR_MASK_COLOR) || color == PNG_COLOR_TYPE_PALETTE)
    png_set_rgb_to_gray(png, 1, 0.299, 0.587);
  if (bit_depth == 8 && depth == 16) png_set_strip_16(png);
  if (bit_depth == 16 && depth < 16) png_set_expand_16(png);
  if (bit_depth == 16) png_set_swap(png);  // PNG stores 16-bit samples big-endian
  png_read_update_info(png, info);
  if (png_get_channels(png, info) != 1 || png_get_rowbytes(png, info) != *w * sample_bytes) {
    *problem = "unexpected PNG layout after conversion";
    return false;
  }
  return true;
}

bool png_rows(png_structp png, png_bytep* rows) {
  if (setjmp(png_jmpbuf(png))) return false;
  png_read_image(png, rows);
  png_read_end(png, nullptr);
  return true;
}

template <class T>
Result<std::vector<T>> decode_png(const std::string& path, int bit_depth, int* width, int* height) {
  FilePtr f = open_file(path);
  if (!f) return Error{path + ": cannot open"};
  unsigned char sig[8];
  if (std::fread(sig, 1, 8, f.get()) != 8 || png_sig_cmp(sig, 0, 8) != 0)
    return Error{path + ": not a PNG file"};
  std::string err;
  PngRead r;
  r.png = png_create_read_struct(PNG_LIBPNG_VER_STRING, &err, png_error_fn, png_warning_fn);
  if (!r.png) return Error{path + ": libpng could not start"};
  r.info = png_create_info_struct(r.png);
  if (!r.info) return Error{path + ": libpng could not start"};
  png_uint_32 w = 0, h = 0;
  const char* problem = nullptr;
  if (!png_header(r.png, r.info, f.get(), bit_depth, sizeof(T), &w, &h, &problem))
    return Error{path + ": " +
                 (problem       ? std::string(problem)
                  : err.empty() ? std::string("invalid PNG")
                                : err)};
  std::vector<T> data(static_cast<size_t>(w) * h);
  std::vector<png_bytep> rows(h);
  for (png_uint_32 y = 0; y < h; ++y)
    rows[y] = reinterpret_cast<png_bytep>(data.data() + static_cast<size_t>(y) * w);
  if (!png_rows(r.png, rows.data()))
    return Error{path + ": " + (err.empty() ? std::string("invalid PNG") : err)};
  *width = static_cast<int>(w);
  *height = static_cast<int>(h);
  return data;
}

// ---- JPEG ----

struct JpegError {
  jpeg_error_mgr mgr;
  std::jmp_buf jump;
  char message[JMSG_LENGTH_MAX];
};

void jpeg_error_exit(j_common_ptr cinfo) {
  auto* e = reinterpret_cast<JpegError*>(cinfo->err);
  (*cinfo->err->format_message)(cinfo, e->message);
  std::longjmp(e->jump, 1);
}

bool jpeg_header(jpeg_decompress_struct* cinfo, JpegError* jerr, FILE* f) {
  if (setjmp(jerr->jump)) return false;
  jpeg_stdio_src(cinfo, f);
  jpeg_read_header(cinfo, TRUE);
  cinfo->out_color_space =
      JCS_GRAYSCALE;  // libjpeg converts (as OpenCV's IMREAD_GRAYSCALE asks it to)
  jpeg_start_decompress(cinfo);
  return cinfo->output_components == 1;
}

bool jpeg_rows(jpeg_decompress_struct* cinfo, JpegError* jerr, unsigned char* out, size_t stride) {
  if (setjmp(jerr->jump)) return false;
  while (cinfo->output_scanline < cinfo->output_height) {
    JSAMPROW row = out + static_cast<size_t>(cinfo->output_scanline) * stride;
    jpeg_read_scanlines(cinfo, &row, 1);
  }
  jpeg_finish_decompress(cinfo);
  return true;
}

Result<GrayImage> read_jpeg_gray(const std::string& path) {
  FilePtr f = open_file(path);
  if (!f) return Error{path + ": cannot open"};
  jpeg_decompress_struct cinfo;
  JpegError jerr;
  cinfo.err = jpeg_std_error(&jerr.mgr);
  jerr.mgr.error_exit = jpeg_error_exit;
  jerr.message[0] = '\0';
  jpeg_create_decompress(&cinfo);
  struct Destroy {
    jpeg_decompress_struct* c;
    ~Destroy() { jpeg_destroy_decompress(c); }
  } destroy{&cinfo};
  if (!jpeg_header(&cinfo, &jerr, f.get()))
    return Error{
        path + ": " +
        std::string(jerr.message[0] ? jerr.message : "the JPEG could not be decoded to grayscale")};
  GrayImage img;
  img.width = static_cast<int>(cinfo.output_width);
  img.height = static_cast<int>(cinfo.output_height);
  img.data.resize(static_cast<size_t>(img.width) * img.height);
  if (!jpeg_rows(&cinfo, &jerr, img.data.data(), static_cast<size_t>(img.width)))
    return Error{path + ": " + std::string(jerr.message[0] ? jerr.message : "invalid JPEG")};
  return img;
}

// ---- PGM (P5, 8 bit) ----

Result<GrayImage> read_pgm(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return Error{path + ": cannot open"};
  std::string magic;
  in >> magic;
  if (magic != "P5") return Error{path + ": only binary PGM (P5) is supported"};
  int vals[3], got = 0;
  while (got < 3 && in) {
    in >> std::ws;
    if (in.peek() == '#') {
      std::string comment;
      std::getline(in, comment);
      continue;
    }
    in >> vals[got++];
  }
  if (got < 3 || vals[0] <= 0 || vals[1] <= 0 || vals[2] != 255)
    return Error{path + ": unsupported PGM header (8-bit P5 only)"};
  in.get();  // the single whitespace after maxval
  GrayImage img;
  img.width = vals[0];
  img.height = vals[1];
  img.data.resize(static_cast<size_t>(img.width) * img.height);
  in.read(reinterpret_cast<char*>(img.data.data()), static_cast<std::streamsize>(img.data.size()));
  if (in.gcount() != static_cast<std::streamsize>(img.data.size()))
    return Error{path + ": truncated PGM"};
  return img;
}

}  // namespace

Result<GrayImage> read_gray_image(const std::string& path) {
  const std::string ext = lower_extension(path);
  if (ext == ".jpg" || ext == ".jpeg") return read_jpeg_gray(path);
  if (ext == ".pgm") return read_pgm(path);
  GrayImage img;
  auto d = decode_png<uint8_t>(path, 8, &img.width, &img.height);
  if (!d) return d.error();
  img.data = std::move(d.value());
  return img;
}

Result<Image16> read_png16(const std::string& path) {
  Image16 img;
  auto d = decode_png<uint16_t>(path, 16, &img.width, &img.height);
  if (!d) return d.error();
  img.data = std::move(d.value());
  return img;
}

}  // namespace strataslam
