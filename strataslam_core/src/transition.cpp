#include "strataslam/floors/transition.hpp"

namespace strataslam {

const char* to_string(TransitionType type) {
  switch (type) {
    case TransitionType::Stairs:
      return "stairs";
    case TransitionType::Ramp:
      return "ramp";
    case TransitionType::Elevator:
      return "elevator";
    case TransitionType::Aerial:
      return "aerial";
    case TransitionType::Unknown:
      return "unknown";
  }
  return "unknown";
}

bool parse_transition_type(const std::string& name, TransitionType* out) {
  for (TransitionType t : {TransitionType::Stairs, TransitionType::Ramp, TransitionType::Elevator,
                           TransitionType::Aerial, TransitionType::Unknown}) {
    if (name == to_string(t)) {
      *out = t;
      return true;
    }
  }
  return false;
}

}  // namespace strataslam
