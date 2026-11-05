#include "strataslam/place_recognition/bow.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "strataslam/frontend/visual_tracker.hpp"  // hamming_distance

namespace strataslam {

namespace {

// "ESVB" dates from the name easyslam; kept so that vocabularies trained before the rename still
// load.
constexpr char kMagic[4] = {'E', 'S', 'V', 'B'};

Descriptor majority(const std::vector<Descriptor>& all, const std::vector<int>& members) {
  int counts[256] = {0};
  for (int m : members) {
    for (size_t b = 0; b < 256; ++b) counts[b] += (all[m][b / 8] >> (b % 8)) & 1;
  }
  Descriptor d{};
  for (size_t b = 0; b < 256; ++b) {
    if (2 * counts[b] > static_cast<int>(members.size()))
      d[b / 8] |= static_cast<uint8_t>(1u << (b % 8));
  }
  return d;
}

}  // namespace

BowVocabulary BowVocabulary::train(const std::vector<Descriptor>& data, const BowConfig& cfg) {
  BowVocabulary v;
  v.nodes_.push_back(Node{});
  std::mt19937_64 rng(cfg.seed);

  struct Job {
    int node;
    std::vector<int> members;
    int level;
  };
  std::vector<Job> stack;
  std::vector<int> all(data.size());
  std::iota(all.begin(), all.end(), 0);
  stack.push_back({0, std::move(all), 0});
  std::vector<std::vector<int>> leaf_members;  // word -> training descriptors (for IDF)

  while (!stack.empty()) {
    Job job = std::move(stack.back());
    stack.pop_back();
    const bool leaf =
        job.level >= cfg.levels || static_cast<int>(job.members.size()) < 2 * cfg.branching;
    if (leaf) {
      v.nodes_[job.node].word = static_cast<int32_t>(leaf_members.size());
      leaf_members.push_back(std::move(job.members));
      continue;
    }
    // k-majority clustering with farthest-point initialisation.
    const int K = cfg.branching;
    std::vector<Descriptor> cent;
    cent.push_back(data[job.members[rng() % job.members.size()]]);
    while (static_cast<int>(cent.size()) < K) {
      int best = job.members[0], best_d = -1;
      for (int m : job.members) {
        int dmin = 1 << 20;
        for (const auto& c : cent) dmin = std::min(dmin, hamming_distance(data[m], c));
        if (dmin > best_d) {
          best_d = dmin;
          best = m;
        }
      }
      cent.push_back(data[best]);
    }
    std::vector<std::vector<int>> groups(K);
    for (int it = 0; it < cfg.kmeans_iterations; ++it) {
      for (auto& g : groups) g.clear();
      for (int m : job.members) {
        int bc = 0, bd = 1 << 20;
        for (int c = 0; c < K; ++c) {
          const int d = hamming_distance(data[m], cent[c]);
          if (d < bd) {
            bd = d;
            bc = c;
          }
        }
        groups[bc].push_back(m);
      }
      for (int c = 0; c < K; ++c) {
        if (!groups[c].empty()) cent[c] = majority(data, groups[c]);
      }
    }
    const int first = static_cast<int>(v.nodes_.size());
    v.nodes_[job.node].first_child = first;
    v.nodes_[job.node].n_children = K;
    for (int c = 0; c < K; ++c) {
      Node n;
      n.centroid = cent[c];
      v.nodes_.push_back(n);
    }
    for (int c = 0; c < K; ++c) stack.push_back({first + c, std::move(groups[c]), job.level + 1});
  }
  v.n_words_ = leaf_members.size();

  // IDF over pseudo-documents (consecutive chunks of the training set).
  const int n_docs = std::max<int>(1, static_cast<int>(data.size()) / cfg.docs_chunk);
  std::vector<int> doc_freq(v.n_words_, 0);
  std::vector<int> word_of(data.size(), 0);
  for (size_t w = 0; w < leaf_members.size(); ++w)
    for (int m : leaf_members[w]) word_of[m] = static_cast<int>(w);
  for (int d = 0; d < n_docs; ++d) {
    std::unordered_set<int> seen;
    for (int i = d * cfg.docs_chunk;
         i < std::min(static_cast<int>(data.size()), (d + 1) * cfg.docs_chunk); ++i)
      seen.insert(word_of[i]);
    for (int w : seen) ++doc_freq[w];
  }
  v.idf_.resize(v.n_words_);
  for (size_t w = 0; w < v.n_words_; ++w) {
    v.idf_[w] =
        std::max(0.05f, static_cast<float>(std::log((n_docs + 1.0) / (doc_freq[w] + 1.0))) + 0.1f);
  }
  return v;
}

uint32_t BowVocabulary::lookup(const Descriptor& d) const {
  int32_t n = 0;
  while (nodes_[n].word < 0) {
    int32_t best = nodes_[n].first_child;
    int best_d = 1 << 20;
    for (int32_t c = nodes_[n].first_child; c < nodes_[n].first_child + nodes_[n].n_children; ++c) {
      const int dist = hamming_distance(d, nodes_[c].centroid);
      if (dist < best_d) {
        best_d = dist;
        best = c;
      }
    }
    n = best;
  }
  return static_cast<uint32_t>(nodes_[n].word);
}

BowVector BowVocabulary::transform(const std::vector<Descriptor>& descriptors) const {
  std::unordered_map<uint32_t, float> acc;
  for (const auto& d : descriptors) {
    const uint32_t w = lookup(d);
    acc[w] += idf_[w];
  }
  BowVector out(acc.begin(), acc.end());
  std::sort(out.begin(), out.end());
  float sum = 0;
  for (const auto& p : out) sum += p.second;
  if (sum > 0)
    for (auto& p : out) p.second /= sum;
  return out;
}

double BowVocabulary::score(const BowVector& a, const BowVector& b) {
  double s = 0;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i].first == b[j].first) {
      s += std::min(a[i].second, b[j].second);
      ++i;
      ++j;
    } else if (a[i].first < b[j].first)
      ++i;
    else
      ++j;
  }
  return s;
}

Result<void> BowVocabulary::save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) return Error{path + ": cannot open for writing"};
  const uint32_t version = 1, n_nodes = static_cast<uint32_t>(nodes_.size()),
                 n_words = static_cast<uint32_t>(n_words_);
  out.write(kMagic, 4);
  out.write(reinterpret_cast<const char*>(&version), 4);
  out.write(reinterpret_cast<const char*>(&n_nodes), 4);
  out.write(reinterpret_cast<const char*>(&n_words), 4);
  for (const auto& n : nodes_) {
    out.write(reinterpret_cast<const char*>(n.centroid.data()), 32);
    out.write(reinterpret_cast<const char*>(&n.first_child), 4);
    out.write(reinterpret_cast<const char*>(&n.n_children), 4);
    out.write(reinterpret_cast<const char*>(&n.word), 4);
  }
  out.write(reinterpret_cast<const char*>(idf_.data()),
            static_cast<std::streamsize>(idf_.size() * 4));
  if (!out) return Error{path + ": write failed"};
  return {};
}

Result<BowVocabulary> BowVocabulary::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return Error{path + ": cannot open"};
  char magic[4];
  uint32_t version = 0, n_nodes = 0, n_words = 0;
  in.read(magic, 4);
  in.read(reinterpret_cast<char*>(&version), 4);
  in.read(reinterpret_cast<char*>(&n_nodes), 4);
  in.read(reinterpret_cast<char*>(&n_words), 4);
  if (!in || std::memcmp(magic, kMagic, 4) != 0)
    return Error{path + ": bad vocabulary magic number"};
  if (version != 1)
    return Error{path + ": unsupported vocabulary version " + std::to_string(version)};
  if (n_nodes > (1u << 26)) return Error{path + ": implausible vocabulary size"};
  BowVocabulary v;
  v.nodes_.resize(n_nodes);
  v.n_words_ = n_words;
  for (auto& n : v.nodes_) {
    in.read(reinterpret_cast<char*>(n.centroid.data()), 32);
    in.read(reinterpret_cast<char*>(&n.first_child), 4);
    in.read(reinterpret_cast<char*>(&n.n_children), 4);
    in.read(reinterpret_cast<char*>(&n.word), 4);
  }
  v.idf_.resize(n_words);
  in.read(reinterpret_cast<char*>(v.idf_.data()), static_cast<std::streamsize>(n_words) * 4);
  if (!in) return Error{path + ": truncated vocabulary"};
  return v;
}

bool BowVocabulary::operator==(const BowVocabulary& o) const {
  if (nodes_.size() != o.nodes_.size() || idf_ != o.idf_) return false;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].centroid != o.nodes_[i].centroid ||
        nodes_[i].first_child != o.nodes_[i].first_child ||
        nodes_[i].n_children != o.nodes_[i].n_children || nodes_[i].word != o.nodes_[i].word)
      return false;
  }
  return true;
}

void BowDatabase::add(KeyframeId id, SessionId session, Timestamp stamp, const BowVector& vec) {
  const uint32_t idx = static_cast<uint32_t>(entries_.size());
  entries_.push_back({id, session, stamp});
  if (inverted_.size() < vocab_->words()) inverted_.resize(vocab_->words());
  for (const auto& [w, weight] : vec) inverted_[w].push_back({idx, weight});
}

std::vector<BowDatabase::Candidate> BowDatabase::query(const BowVector& q, SessionId session,
                                                       Timestamp stamp, double exclude_recent_s,
                                                       double min_score, size_t max_results) const {
  std::unordered_map<uint32_t, double> acc;
  for (const auto& [w, weight] : q) {
    if (w >= inverted_.size()) continue;
    for (const auto& [idx, ew] : inverted_[w]) acc[idx] += std::min(weight, ew);
  }
  std::vector<Candidate> out;
  for (const auto& [idx, s] : acc) {
    const Entry& e = entries_[idx];
    if (e.session == session && (stamp - e.stamp) * 1e-9 < exclude_recent_s) continue;
    if (s >= min_score) out.push_back({e.id, s});
  }
  std::sort(out.begin(), out.end(),
            [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
  if (out.size() > max_results) out.resize(max_results);
  return out;
}

}  // namespace strataslam
