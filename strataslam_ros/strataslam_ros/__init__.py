"""strataslam ROS 2 helpers: profile validation for launch files (backed by the C++ core)."""

from __future__ import annotations


def validate_profile(path: str) -> tuple[list[str], list[str]]:
    """Returns ``(errors, warnings)`` for a profile, using the same code as the node."""
    from strataslam_ros import _strataslam_py  # built by CMake (pybind11)

    errors, warnings = _strataslam_py.validate_profile(path)
    return list(errors), list(warnings)
