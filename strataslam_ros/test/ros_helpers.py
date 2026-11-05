"""Shared helpers of the ROS 2 tests: locate the installed tools, generate simulator bags."""

import os
import subprocess
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory

SHARE = Path(get_package_share_directory("strataslam_ros"))
PROFILES = SHARE / "profiles"
EXPORTER = Path(get_package_prefix("strataslam_ros")) / "lib" / "strataslam_ros" / "strataslam_sim_export_bag"


def export_bag(out: Path, profile: Path | None = None, corrupt: str = "", seconds: float = 20.0, scenario: str = "S01") -> Path:
    """Runs the simulator and writes a rosbag2 (plus vocab.bin and the ground truth, trajectory.tum) under `out`."""
    cmd = [str(EXPORTER), scenario, str(out), "--seconds", str(seconds)]
    if profile is not None:
        cmd += ["--profile", str(profile)]
    if corrupt:
        cmd += ["--corrupt", corrupt]
    subprocess.run(cmd, check=True, capture_output=True, text=True, env=os.environ)
    return out / "bag"


def profile_with_map_dir(profile: Path, map_dir: Path, dest: Path) -> Path:
    """Copy of a profile that saves its map to `map_dir` ."""
    text = profile.read_text()
    lines = [ln for ln in text.splitlines() if not ln.startswith("  map_dir:")]
    out = []
    for ln in lines:
        out.append(ln)
        if ln.strip() == "output:":
            out.append(f"  map_dir: {map_dir}")
    if "output:" not in text:
        out += ["", "output:", f"  map_dir: {map_dir}"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n")
    return dest
