"""An invalid profile aborts the launch with the validation messages, before any node starts."""

import subprocess
from pathlib import Path


def test_invalid_profile_aborts_before_any_node_starts(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("strataslam: {version: 1}\nsensors:\n  camra: {enabled: true}\n  lidar: {enabled: true}\n")
    proc = subprocess.run(["ros2", "launch", "strataslam_ros", "strataslam.launch.py", f"profile:={bad}"], capture_output=True, text=True, timeout=60, check=False)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert 'config error: sensors.camra: unknown key; did you mean "camera"?' in out
    assert "process started" not in out and "strataslam_container" not in out.replace("strataslam_container error", "")


def test_unknown_mode_is_rejected(tmp_path: Path) -> None:
    from ros_helpers import PROFILES

    proc = subprocess.run(["ros2", "launch", "strataslam_ros", "strataslam.launch.py", f"profile:={PROFILES / 'lidar_only.yaml'}", "mode:=fly"],
                          capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode != 0
    assert "unknown mode" in proc.stdout + proc.stderr
