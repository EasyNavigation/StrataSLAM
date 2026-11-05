"""`strataslam doctor` on simulator bags: exit 0 for a correct one, the expected failure for each
corruption of the plan (no TF, wrong depth encoding, 50 ms RGB-depth offset, inverted IMU z, no camera_info)."""

import json
from pathlib import Path

import pytest
from ros_helpers import PROFILES, export_bag

from strataslam_tools import doctor

PROFILE = PROFILES / "handheld_camera_imu.yaml"


def run_doctor(bag: Path) -> tuple[int, dict[str, str]]:
    code, text = doctor.run(str(PROFILE), bag=str(bag), as_json=True)
    checks = {c["name"]: c["status"] for c in json.loads(text)["checks"]}
    return code, checks


def test_a_correct_bag_returns_zero(tmp_path: Path) -> None:
    bag = export_bag(tmp_path / "good", PROFILE, seconds=8.0)
    code, checks = run_doctor(bag)
    assert code == 0, checks


@pytest.mark.parametrize(
    "corruption,failing_check",
    [
        ("no_tf", "tf_chain"),
        ("depth_encoding", "depth_encoding"),
        ("rgb_depth_offset", "rgb_depth_sync"),
        ("imu_z_flip", "imu_rest"),
        ("no_camera_info", "camera_info"),
    ],
)
def test_each_corrupted_bag_fails_where_expected(tmp_path: Path, corruption: str, failing_check: str) -> None:
    bag = export_bag(tmp_path / corruption, PROFILE, corrupt=corruption, seconds=8.0)
    code, checks = run_doctor(bag)
    assert code == 2
    assert checks[failing_check] == doctor.FAIL, checks
