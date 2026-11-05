#pragma once

#include <cstdint>

namespace strataslam {

using Timestamp = int64_t;  ///< nanoseconds
using KeyframeId = uint64_t;
using FloorId = int32_t;  ///< -1 = unassigned
using SessionId = uint32_t;

constexpr FloorId kUnassignedFloor = -1;

}  // namespace strataslam
