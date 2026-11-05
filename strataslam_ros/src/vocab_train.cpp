// strataslam_vocab_train <image_folder> <out.bin>: trains the visual vocabulary from ORB features of the
// images of a folder (use images with a license compatible with your project).
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "strataslam/frontend/feature_extractor.hpp"
#include "strataslam/io/image_io.hpp"
#include "strataslam/place_recognition/bow.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <image_folder> <out.bin>\n";
    return 2;
  }
  auto orb = strataslam::make_orb_extractor(1000);
  std::vector<strataslam::Descriptor> all;
  size_t images = 0;
  // In name order: the directory's own order depends on the file system, and the k-means starts from the descriptors in
  // the order they come, so the same folder would give another vocabulary elsewhere.
  std::vector<std::filesystem::path> files;
  for (const auto& e : std::filesystem::recursive_directory_iterator(argv[1])) {
    std::string ext = e.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".pgm") files.push_back(e.path());
  }
  std::sort(files.begin(), files.end());
  for (const auto& file : files) {
    const auto g = strataslam::read_gray_image(file.string());
    if (!g) {
      std::cerr << "skipped: " << g.error().message << "\n";
      continue;
    }
    for (const auto& k : orb->extract(g.value())) all.push_back(k.descriptor);
    ++images;
  }
  if (all.size() < 2000) {
    std::cerr << "only " << all.size() << " descriptors from " << images << " image(s): need at least 2000 (use more or richer images)\n";
    return 1;
  }
  const strataslam::BowVocabulary vocab = strataslam::BowVocabulary::train(all, strataslam::BowConfig{});
  const auto r = vocab.save(argv[2]);
  if (!r) {
    std::cerr << r.error().message << "\n";
    return 1;
  }
  std::cout << "trained a vocabulary of " << vocab.words() << " words from " << all.size() << " descriptors (" << images << " images)\n";
  return 0;
}
