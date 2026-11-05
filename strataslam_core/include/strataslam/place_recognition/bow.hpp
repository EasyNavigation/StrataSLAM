#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "strataslam/common/result.hpp"
#include "strataslam/common/types.hpp"
#include "strataslam/sensors/measurements.hpp"

namespace strataslam {

struct BowConfig {
  int branching = 10;
  int levels = 4;            ///< up to branching^levels words (10^4: far more words than descriptors per image)
  int kmeans_iterations = 6;
  uint64_t seed = 42;
  int docs_chunk = 300;      ///< training descriptors per pseudo-document when computing IDF
};

/// Sparse bag-of-words vector: (word id, weight) sorted by id, L1-normalised.
using BowVector = std::vector<std::pair<uint32_t, float>>;

/// Hierarchical vocabulary over 256-bit binary descriptors (k-majority clustering), with TF-IDF
/// weights. Implemented from the description of vocabulary trees (Nister & Stewenius, CVPR 2006) and
/// binary bag-of-words (Galvez-Lopez & Tardos, T-RO 2012); no third-party vocabulary code or data.
class BowVocabulary {
 public:
  static BowVocabulary train(const std::vector<Descriptor>& training, const BowConfig& config);

  size_t words() const { return n_words_; }
  BowVector transform(const std::vector<Descriptor>& descriptors) const;
  /// Similarity in [0, 1]: sum_i min(a_i, b_i) = 1 - L1(a, b)/2 for normalised vectors.
  static double score(const BowVector& a, const BowVector& b);

  Result<void> save(const std::string& path) const;
  static Result<BowVocabulary> load(const std::string& path);

  bool operator==(const BowVocabulary& o) const;

 private:
  struct Node {
    Descriptor centroid;
    int32_t first_child = -1;  ///< children are contiguous: [first_child, first_child + n_children)
    int32_t n_children = 0;
    int32_t word = -1;         ///< leaf word id
  };
  uint32_t lookup(const Descriptor& d) const;

  std::vector<Node> nodes_;     ///< node 0 is the root
  std::vector<float> idf_;      ///< per word
  size_t n_words_ = 0;
};

/// Inverted-index database. The caller keeps one per floor (gating layer 1).
class BowDatabase {
 public:
  explicit BowDatabase(std::shared_ptr<const BowVocabulary> vocabulary) : vocab_(std::move(vocabulary)) {}

  void add(KeyframeId id, SessionId session, Timestamp stamp, const BowVector& vec);

  struct Candidate {
    KeyframeId id;
    double score;
  };
  /// Best candidates by score. Entries of the same session within `exclude_recent_s` seconds of the
  /// query are skipped; entries of other sessions are always eligible.
  std::vector<Candidate> query(const BowVector& q, SessionId session, Timestamp stamp, double exclude_recent_s,
                               double min_score, size_t max_results) const;
  size_t size() const { return entries_.size(); }

 private:
  struct Entry { KeyframeId id; SessionId session; Timestamp stamp; };
  std::shared_ptr<const BowVocabulary> vocab_;
  std::vector<Entry> entries_;
  std::vector<std::vector<std::pair<uint32_t, float>>> inverted_;  // word -> (entry index, weight)
};

}  // namespace strataslam
