"""`strataslam doctor` on simulator bags: exit 0 for a correct one, the expected failure for each
corruption of the plan (no TF, wrong depth encoding, 50 ms RGB-depth offset, inverted IMU z, no camera_info)."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from ros_helpers import PROFILES, export_bag

from strataslam_tools import doctor

PROFILE = PROFILES / "handheld_camera_imu.yaml"


@pytest.fixture(autouse=True)
def roomy_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The disk check grades the machine the tests run on, not the bag: a CI runner with under 10 GB free warned."""
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: SimpleNamespace(total=500e9, used=100e9, free=400e9))


def run_doctor(bag: Path) -> tuple[int, dict[str, str]]:
    code, text = doctor.run(str(PROFILE), bag=str(bag), as_json=True)
    checks = {c["name"]: c["status"] for c in json.loads(text)["checks"]}
    return code, checks


def test_a_correct_bag_returns_zero(tmp_path: Path) -> None:
    bag = export_bag(tmp_path / "good", PROFILE, seconds=8.0)
    code, checks = run_doctor(bag)
    assert code == 0, {name: status for name, status in checks.items() if status != doctor.OK}


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
