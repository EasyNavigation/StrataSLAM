#pragma once

#include <string>
#include <utility>
#include <variant>

namespace strataslam {

/// Expected, recoverable failure (configuration, I/O). Programming errors use exceptions.
struct Error {
  std::string message;
};

/// Value-or-error return type for expected failures.
template <typename T>
class Result {
 public:
  Result(T value) : v_(std::move(value)) {}      // NOLINT: implicit by design
  Result(Error error) : v_(std::move(error)) {}  // NOLINT: implicit by design

  bool ok() const { return std::holds_alternative<T>(v_); }
  explicit operator bool() const { return ok(); }
  T& value() { return std::get<T>(v_); }
  const T& value() const { return std::get<T>(v_); }
  const Error& error() const { return std::get<Error>(v_); }

 private:
  std::variant<T, Error> v_;
};

template <>
class Result<void> {
 public:
  Result() = default;
  Result(Error error) : error_(std::move(error)), ok_(false) {}  // NOLINT

  bool ok() const { return ok_; }
  explicit operator bool() const { return ok_; }
  const Error& error() const { return error_; }

 private:
  Error error_;
  bool ok_ = true;
};

}  // namespace strataslam
