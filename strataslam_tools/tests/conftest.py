"""Shared helpers: optional test dependencies (Open3D, the C++ core executables).

Under `colcon test` the executables are found in the installed workspace; elsewhere set
STRATASLAM_SAMPLE_BIN / STRATASLAM_EVAL_BIN. A missing dependency skips the test, unless
STRATASLAM_TEST_STRICT=1 (set in CI), which turns it into a failure so nothing is skipped silently.
"""
import os
from pathlib import Path
from types import ModuleType

import pytest

STRICT = os.environ.get("STRATASLAM_TEST_STRICT") == "1"


def _missing(reason: str) -> None:
    """Skip the test asking for it. Call from a fixture or a test, never at import time: a module-level skip
    under the launch_testing plugin (loaded by `colcon test` with ROS) skips the whole session."""
    if STRICT:
        pytest.fail(reason + " (STRATASLAM_TEST_STRICT=1)")
    pytest.skip(reason)


def require_open3d() -> ModuleType:
    try:
        import open3d
    except ImportError:
        _missing("Open3D is not installed (pip install open3d)")
    return open3d


def require_module(name: str, hint: str) -> ModuleType:
    """An optional module, skipped like the others (and a failure under STRATASLAM_TEST_STRICT=1)."""
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError:
        _missing(f"{name} is not installed ({hint})")
        raise


def core_executable(env_var: str, name: str) -> str:
    """Path of a core executable: from `env_var`, else from the colcon install (lib/strataslam_core).

    strataslam_core is a plain CMake package: colcon puts it in CMAKE_PREFIX_PATH, not AMENT_PREFIX_PATH."""
    exe = os.environ.get(env_var)
    if exe:
        return exe
    prefixes = os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep) + os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
    for prefix in filter(None, prefixes):
        candidate = Path(prefix) / "lib" / "strataslam_core" / name
        if candidate.is_file():
            return str(candidate)
    _missing(f"set {env_var} to the {name} executable (or source a workspace that installs it)")
    return ""
