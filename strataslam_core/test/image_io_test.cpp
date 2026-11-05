#include <gtest/gtest.h>

#include <png.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <unistd.h>

#include "strataslam/io/image_io.hpp"

using namespace strataslam;

namespace {

std::filesystem::path tmp(const std::string& name) {
  const auto dir = std::filesystem::temp_directory_path() / ("strataslam_image_io_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  return dir / name;
}

/// Writes a PNG with libpng's writer (independent of the reader under test).
void write_png(const std::filesystem::path& p, int w, int h, int color_type, int bit_depth, const std::vector<uint8_t>& raw) {
  FILE* f = std::fopen(p.c_str(), "wb");
  ASSERT_NE(f, nullptr);
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png_create_info_struct(png);
  png_init_io(png, f);
  png_set_IHDR(png, info, w, h, bit_depth, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  const size_t row = raw.size() / h;
  for (int y = 0; y < h; ++y) png_write_row(png, const_cast<uint8_t*>(raw.data() + y * row));
  png_write_end(png, nullptr);
  png_destroy_write_struct(&png, &info);
  std::fclose(f);
}

}  // namespace

TEST(ImageIo, ColourPngIsConvertedToGrayWithTheBt601Weights) {
  const auto p = tmp("rgb.png");
  // Pure red, green, blue, a gray pixel (kept as is) and white.
  write_png(p, 5, 1, PNG_COLOR_TYPE_RGB, 8, {255, 0, 0, 0, 255, 0, 0, 0, 255, 77, 77, 77, 255, 255, 255});
  const auto g = read_gray_image(p.string());
  ASSERT_TRUE(g) << g.error().message;
  ASSERT_EQ(g.value().width, 5);
  ASSERT_EQ(g.value().height, 1);
  EXPECT_NEAR(g.value().data[0], 0.299 * 255, 1.0);
  EXPECT_NEAR(g.value().data[1], 0.587 * 255, 1.0);
  EXPECT_NEAR(g.value().data[2], 0.114 * 255, 1.0);
  EXPECT_EQ(g.value().data[3], 77);
  EXPECT_EQ(g.value().data[4], 255);
}

TEST(ImageIo, SixteenBitPngKeepsItsValuesAndEightBitIsNotWidenedWrongly) {
  const auto p = tmp("depth.png");
  // 16-bit gray, big-endian in the file: 0x1388 = 5000 (1 m in TUM units) and 0xFFFF.
  write_png(p, 2, 1, PNG_COLOR_TYPE_GRAY, 16, {0x13, 0x88, 0xFF, 0xFF});
  const auto d = read_png16(p.string());
  ASSERT_TRUE(d) << d.error().message;
  ASSERT_EQ(d.value().data.size(), 2u);
  EXPECT_EQ(d.value().data[0], 5000);
  EXPECT_EQ(d.value().data[1], 65535);
  // The same file read as 8-bit gray keeps the high byte.
  const auto g = read_gray_image(p.string());
  ASSERT_TRUE(g);
  EXPECT_EQ(g.value().data[0], 0x13);
}

TEST(ImageIo, BinaryPgmIsRead) {
  const auto p = tmp("img.pgm");
  {
    std::ofstream out(p, std::ios::binary);
    out << "P5\n# a comment\n3 2\n255\n";
    const char px[6] = {1, 2, 3, 4, 5, 6};
    out.write(px, 6);
  }
  const auto g = read_gray_image(p.string());
  ASSERT_TRUE(g) << g.error().message;
  EXPECT_EQ(g.value().width, 3);
  EXPECT_EQ(g.value().height, 2);
  EXPECT_EQ(g.value().data, (std::vector<uint8_t>{1, 2, 3, 4, 5, 6}));
}

TEST(ImageIo, MissingTruncatedOrForeignFilesAreErrorsNotCrashes) {
  EXPECT_FALSE(read_gray_image(tmp("missing.png").string()));
  EXPECT_FALSE(read_png16(tmp("missing.png").string()));
  // A PNG cut in the middle of its image data: libpng aborts with longjmp, the reader returns an error.
  const auto good = tmp("big.png");
  write_png(good, 64, 64, PNG_COLOR_TYPE_RGB, 8, std::vector<uint8_t>(64 * 64 * 3, 128));
  std::string bytes;
  {
    std::ifstream in(good, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(in), {});
  }
  const auto cut = tmp("cut.png");
  std::ofstream(cut, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
  const auto r = read_gray_image(cut.string());
  EXPECT_FALSE(r);
  const auto junk = tmp("junk.jpg");
  std::ofstream(junk, std::ios::binary) << "not a jpeg at all";
  EXPECT_FALSE(read_gray_image(junk.string()));
  const auto pgm = tmp("short.pgm");
  std::ofstream(pgm, std::ios::binary) << "P5\n10 10\n255\nabc";
  EXPECT_FALSE(read_gray_image(pgm.string()));
  // A colour PNG cannot be read as a 16-bit single-channel image silently: it is converted to one channel.
  const auto d = read_png16(good.string());
  ASSERT_TRUE(d);
  EXPECT_EQ(d.value().data.size(), 64u * 64u);
}
