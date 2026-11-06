"""Every shipped profile validates with the core's code; mistakes produce the plan's messages."""

from pathlib import Path

import pytest
from ros_helpers import PROFILES

from strataslam_ros import validate_profile


@pytest.mark.parametrize("name", ["ground_camera", "ground_lidar", "ground_lidar_camera_imu", "handheld_camera_imu",
                                  "aerial_lidar_camera_imu"])
def test_shipped_profiles_are_valid(name: str) -> None:
    errors, _ = validate_profile(str(PROFILES / f"{name}.yaml"))
    assert errors == []


def test_all_errors_are_reported_with_paths_and_suggestions(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("strataslam: {version: 1}\nsensors:\n  camra: {enabled: true}\n  external_odometry: {enabled: true}\noutput: {voxel_size: 5}\n")
    errors, _ = validate_profile(str(bad))
    assert 'config error: sensors.camra: unknown key; did you mean "camera"?' in errors
    assert "config error: sensors: at least one sensor must be enabled (camera or lidar)" in errors
    assert ("config error: sensors.external_odometry: needs either a topic (nav_msgs/Odometry) or a frame "
            "(TF frame -> robot.base_frame)") in errors
    assert "config error: output.voxel_size: must be in [0.005, 1.0]" in errors
    assert len(errors) >= 4
