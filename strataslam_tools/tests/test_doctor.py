"""strataslam doctor: every check on synthetic data, including the five corruptions of the plan."""

import json
import types
from pathlib import Path

import numpy as np
import pytest

from strataslam_tools import doctor
from strataslam_tools.profile import load_profile

REPO = Path(__file__).resolve().parents[2]
PROFILES = REPO / "strataslam_ros" / "profiles"


def ns(**kw):  # type: ignore[no-untyped-def]
    return types.SimpleNamespace(**kw)


def hdr(frame: str, t: float):  # type: ignore[no-untyped-def]
    return ns(frame_id=frame, stamp=ns(sec=int(t), nanosec=round((t % 1) * 1e9)))


def image(frame: str, t: float, enc: str, fill: float = 2.0, w: int = 32, h: int = 24):  # type: ignore[no-untyped-def]
    if enc == "32FC1":
        data = np.full(w * h, fill, dtype=np.float32).tobytes()
    elif enc == "16UC1":
        data = np.full(w * h, int(fill * 1000), dtype=np.uint16).tobytes()
    else:
        data = bytes(w * h)
    return ns(header=hdr(frame, t), encoding=enc, width=w, height=h, data=data)


def camera_info(t: float = 0.0, w: int = 32, h: int = 24, fx: float = 30.0, model: str = "plumb_bob"):  # type: ignore[no-untyped-def]
    return ns(header=hdr("cam", t), width=w, height=h, distortion_model=model, k=[fx, 0, w / 2, 0, fx, h / 2, 0, 0, 1])


def imu(t: float, accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0)):  # type: ignore[no-untyped-def]
    return ns(header=hdr("imu_link", t), linear_acceleration=ns(x=accel[0], y=accel[1], z=accel[2]),
              angular_velocity=ns(x=gyro[0], y=gyro[1], z=gyro[2]))


def tf(parent: str, child: str, t=(0, 0, 0), q=(0, 0, 0, 1)):  # type: ignore[no-untyped-def]
    return ns(header=ns(frame_id=parent), child_frame_id=child,
              transform=ns(translation=ns(x=t[0], y=t[1], z=t[2]), rotation=ns(x=q[0], y=q[1], z=q[2], w=q[3])))


def topic(stamps, samples):  # type: ignore[no-untyped-def]
    return doctor.TopicData(type="x", header_stamps=list(stamps), recv_times=list(stamps), samples=samples)


def good_data(prof, depth_enc="32FC1", rgb_offset=0.0, with_info=True, imu_accel=(0, 0, 9.81), with_tf=True):  # type: ignore[no-untyped-def]
    cam = prof["sensors"]["camera"]
    d = doctor.DoctorData()
    cam_t = [i / 20 + 1000.0 for i in range(60)]
    d.topics[cam["depth_topic"]] = topic(cam_t, [image("cam", t, depth_enc) for t in cam_t[:12]])
    d.topics[cam["rgb_topic"]] = topic([t + rgb_offset for t in cam_t], [image("cam", t + rgb_offset, "mono8") for t in cam_t[:12]])
    if with_info:
        d.topics[cam["info_topic"]] = topic(cam_t, [camera_info(t) for t in cam_t[:5]])
    imu_t = [i / 200 + 1000.0 for i in range(1000)]
    d.topics[prof["sensors"]["imu"]["topic"]] = topic(imu_t, [imu(t, imu_accel) for t in imu_t])
    if with_tf:
        d.tf_static = [tf("base_link", "cam"), tf("base_link", "imu_link"), tf("base_link", "lidar")]
    return d


def status(checks, name):  # type: ignore[no-untyped-def]
    return next(c for c in checks if c.name == name).status


@pytest.fixture(autouse=True)
def lots_of_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda _p: ns(free=100e9))


@pytest.fixture
def prof():  # type: ignore[no-untyped-def]
    return load_profile(PROFILES / "handheld_camera_imu.yaml")


def test_a_correct_setup_passes_everything(prof) -> None:  # type: ignore[no-untyped-def]
    checks = doctor.run_checks(prof, good_data(prof))
    assert doctor.exit_code(checks) == 0, doctor.format_table(checks)
    assert all(c.status == doctor.OK for c in checks)
    assert "✔" in doctor.format_table(checks)


@pytest.mark.parametrize(
    "kwargs,check",
    [
        ({"with_tf": False}, "tf_chain"),
        ({"depth_enc": "8UC1"}, "depth_encoding"),
        ({"rgb_offset": 0.050}, "rgb_depth_sync"),
        ({"imu_accel": (0, 0, -9.81)}, "imu_rest"),
        ({"with_info": False}, "camera_info"),
    ],
)
def test_each_corruption_of_the_plan_is_a_failure(prof, kwargs, check) -> None:  # type: ignore[no-untyped-def]
    checks = doctor.run_checks(prof, good_data(prof, **kwargs))
    assert status(checks, check) == doctor.FAIL
    assert doctor.exit_code(checks) == 2
    assert "✘" in doctor.format_table(checks)
    assert "->" in doctor.format_table(checks)  # a hint on how to fix it


def test_missing_topics_are_reported(prof) -> None:  # type: ignore[no-untyped-def]
    data = good_data(prof)
    del data.topics[prof["sensors"]["camera"]["depth_topic"]]
    checks = doctor.run_checks(prof, data)
    assert status(checks, "topics") == doctor.FAIL


@pytest.mark.parametrize("hz,expected", [(20, doctor.OK), (15, doctor.OK), (14, doctor.WARN), (10, doctor.WARN), (3, doctor.FAIL)])
def test_camera_rate_thresholds(prof, hz, expected) -> None:  # type: ignore[no-untyped-def]
    data = good_data(prof)
    t = [i / hz + 1000.0 for i in range(int(hz * 3) + 2)]
    data.topics[prof["sensors"]["camera"]["rgb_topic"]] = topic(t, [image("cam", x, "mono8") for x in t[:5]])
    assert status(doctor.run_checks(prof, data), "camera_rate") == expected


@pytest.mark.parametrize("hz,expected", [(200, doctor.OK), (80, doctor.WARN), (20, doctor.FAIL)])
def test_imu_rate_thresholds(prof, hz, expected) -> None:  # type: ignore[no-untyped-def]
    data = good_data(prof)
    t = [i / hz + 1000.0 for i in range(hz * 4)]
    data.topics[prof["sensors"]["imu"]["topic"]] = topic(t, [imu(x) for x in t])
    assert status(doctor.run_checks(prof, data), "imu_rate") == expected


def test_sync_thresholds(prof) -> None:  # type: ignore[no-untyped-def]
    assert status(doctor.run_checks(prof, good_data(prof, rgb_offset=0.020)), "rgb_depth_sync") == doctor.WARN
    assert status(doctor.run_checks(prof, good_data(prof, rgb_offset=0.005)), "rgb_depth_sync") == doctor.OK


def test_valid_depth_pixels_thresholds(prof) -> None:  # type: ignore[no-untyped-def]
    cam = prof["sensors"]["camera"]
    for fill, expected in ((2.0, doctor.OK), (50.0, doctor.FAIL)):  # 50 m is outside [depth_min, depth_max]
        data = good_data(prof)
        data.topics[cam["depth_topic"]].samples = [image("cam", 1000.0, "32FC1", fill)]
        assert status(doctor.run_checks(prof, data), "valid_depth") == expected
    data = good_data(prof)
    half = np.concatenate([np.full(384, 2.0, np.float32), np.zeros(384, np.float32)])
    sample = image("cam", 1000.0, "32FC1")
    sample.data = half.tobytes()
    data.topics[cam["depth_topic"]].samples = [sample]
    assert status(doctor.run_checks(prof, data), "valid_depth") == doctor.OK  # 50 %
    sample.data = np.concatenate([np.full(200, 2.0, np.float32), np.zeros(568, np.float32)]).tobytes()  # 26 %
    assert status(doctor.run_checks(prof, data), "valid_depth") == doctor.WARN


def test_depth_must_be_registered_to_color(prof) -> None:  # type: ignore[no-untyped-def]
    data = good_data(prof)
    cam = prof["sensors"]["camera"]
    data.topics[cam["depth_topic"]].samples = [image("depth_optical_frame", 1000.0, "32FC1")]
    data.topics[cam["depth_topic"]].header_stamps = data.topics[cam["depth_topic"]].header_stamps
    prof2 = {**prof, "sensors": {**prof["sensors"], "camera": {**cam, "depth_topic": cam["depth_topic"].replace("aligned_", "")}}}
    data.topics[prof2["sensors"]["camera"]["depth_topic"]] = data.topics[cam["depth_topic"]]
    assert status(doctor.run_checks(prof2, data), "depth_registered") == doctor.FAIL


def test_camera_info_size_and_distortion(prof) -> None:  # type: ignore[no-untyped-def]
    cam = prof["sensors"]["camera"]
    data = good_data(prof)
    data.topics[cam["info_topic"]].samples = [camera_info(w=64)]
    assert status(doctor.run_checks(prof, data), "camera_info") == doctor.FAIL  # size mismatch
    data.topics[cam["info_topic"]].samples = [camera_info(model="fisheye_weird")]
    assert status(doctor.run_checks(prof, data), "camera_info") == doctor.WARN
    data.topics[cam["info_topic"]].samples = [camera_info(fx=0.0)]
    assert status(doctor.run_checks(prof, data), "camera_info") == doctor.FAIL  # null K


def test_imu_units_and_mounting(prof) -> None:  # type: ignore[no-untyped-def]
    data = good_data(prof, imu_accel=(0, 0, 1.0))  # in g, not m/s^2
    assert status(doctor.run_checks(prof, data), "imu_rest") == doctor.FAIL
    # Mounted rotated 90 degrees about x: the TF tells the doctor that "up" in the IMU frame is -y in base.
    data = good_data(prof, imu_accel=(0, 9.81, 0))
    s = np.sqrt(0.5)
    data.tf_static = [tf("base_link", "cam"), tf("base_link", "imu_link", q=(s, 0, 0, s)), tf("base_link", "lidar")]
    assert status(doctor.run_checks(prof, data), "imu_rest") == doctor.OK
    # Slightly off magnitude: a warning, not a failure.
    data = good_data(prof, imu_accel=(0, 0, 10.4))
    assert status(doctor.run_checks(prof, data), "imu_rest") == doctor.WARN


def test_a_chain_through_moving_joints_is_a_warning_not_a_failure(prof) -> None:  # type: ignore[no-untyped-def]
    # A mount on /tf (a robot's pan-tilt head, a torso lift): the node can map with it as long as the joints do not
    # move (it reads the mounting once). Audit A8-07: the doctor failed the PAL Tiago, which maps fine.
    data = good_data(prof)
    data.tf_dynamic = list(data.tf_static)
    data.tf_static = []
    checks = doctor.run_checks(prof, data)
    c = next(c for c in checks if c.name == "tf_chain")
    assert c.status == doctor.WARN
    assert "moving joints" in c.detail and "keep those joints still" in c.hint


def test_time_checks(prof) -> None:  # type: ignore[no-untyped-def]
    data = good_data(prof)
    data.topics["/camera/imu"].header_stamps[10] = 5.0  # a stamp that goes backwards
    assert status(doctor.run_checks(prof, data), "time") == doctor.FAIL
    data = good_data(prof)
    data.topics["/camera/imu"].header_stamps = [0.0] * len(data.topics["/camera/imu"].header_stamps)
    assert status(doctor.run_checks(prof, data), "time") == doctor.FAIL  # zero stamps
    data = good_data(prof)
    data.use_sim_time, data.wall_clock_now = False, 1.7e9  # stamps near 1000 s but the wall clock is far away
    assert status(doctor.run_checks(prof, data), "time") == doctor.FAIL


@pytest.mark.parametrize("free,expected", [(50e9, doctor.OK), (5e9, doctor.WARN), (1e9, doctor.FAIL)])
def test_disk_thresholds(prof, monkeypatch, free, expected) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda _p: ns(free=free))
    assert status(doctor.run_checks(prof, good_data(prof)), "disk") == expected


def test_combined_profile_checks_lidar_camera_sync_and_extrinsic() -> None:
    prof = load_profile(PROFILES / "aerial_lidar_camera_imu.yaml")
    data = good_data(prof)
    lid_t = [i / 10 + 1000.0 for i in range(40)]
    data.topics[prof["sensors"]["lidar"]["topic"]] = topic(lid_t, [ns(header=hdr("lidar", t)) for t in lid_t[:3]])
    checks = doctor.run_checks(prof, data)
    names = [c.name for c in checks]
    assert "lidar_camera_sync" in names and "lidar_camera_extrinsic" in names
    assert status(checks, "lidar_camera_extrinsic") == doctor.OK
    # No transform between the camera and the lidar frames.
    data.tf_static = [tf("base_link", "cam"), tf("base_link", "imu_link")]
    assert status(doctor.run_checks(prof, data), "lidar_camera_extrinsic") == doctor.FAIL
    # 80 ms clock offset between the sensors.
    data = good_data(prof)
    data.topics[prof["sensors"]["lidar"]["topic"]] = topic([t + 0.08 for t in lid_t], [ns(header=hdr("lidar", 1000.0))])
    data.topics[prof["sensors"]["camera"]["rgb_topic"]] = topic([i / 10 + 1000.0 for i in range(40)], [image("cam", 1000.0, "mono8")])
    assert status(doctor.run_checks(prof, data), "lidar_camera_sync") == doctor.FAIL


def test_barometer_checks() -> None:
    prof = load_profile(PROFILES / "ground_lidar.yaml")
    prof["sensors"]["barometer"] = {"enabled": True, "topic": "/pressure"}

    def data_with(pressure: np.ndarray):  # type: ignore[no-untyped-def]
        d = doctor.DoctorData()
        t = [i / 20 + 1000.0 for i in range(len(pressure))]
        d.topics["/pressure"] = topic(t, [ns(header=hdr("b", x), fluid_pressure=float(p)) for x, p in zip(t, pressure)])
        return d

    rng = np.random.default_rng(1)
    assert doctor.check_barometer(prof, data_with(101325 + rng.normal(0, 1, 100))).status == doctor.OK
    assert doctor.check_barometer(prof, data_with(101325 + rng.normal(0, 40, 100))).status == doctor.WARN
    assert doctor.check_barometer(prof, data_with(np.full(100, 1013.25))).status == doctor.FAIL  # hPa, not Pa


def test_json_output_and_exit_codes(prof) -> None:  # type: ignore[no-untyped-def]
    checks = doctor.run_checks(prof, good_data(prof, with_tf=False))
    doc = json.loads(doctor.to_json(checks))
    assert doc["exit_code"] == 2
    assert {"name", "status", "detail", "hint"} <= set(doc["checks"][0])
    assert doctor.exit_code([doctor.Check("a", doctor.OK, ""), doctor.Check("b", doctor.WARN, "")]) == 1
    assert doctor.exit_code([doctor.Check("a", doctor.OK, "")]) == 0


# ---- audit 2026-11-04 (A8-01..A8-06) ----

def test_camera_info_with_nan_k_is_a_failure(prof) -> None:  # type: ignore[no-untyped-def]
    data = good_data(prof)
    info = camera_info()
    info.k = [float("nan")] * 9
    data.topics[prof["sensors"]["camera"]["info_topic"]] = topic([1000.0], [info])
    assert status(doctor.run_checks(prof, data), "camera_info") == doctor.FAIL


def test_rgb_depth_offset_survives_a_dropped_frame_and_still_sees_a_whole_period(prof) -> None:  # type: ignore[no-untyped-def]
    cam = prof["sensors"]["camera"]
    # The first depth frame was not recorded: pairing by index would see 50 ms everywhere.
    data = good_data(prof)
    d = data.topics[cam["depth_topic"]]
    data.topics[cam["depth_topic"]] = topic(d.header_stamps[1:], d.samples[1:])
    assert status(doctor.run_checks(prof, data), "rgb_depth_sync") == doctor.OK
    # A frame dropped in the middle: the median keeps the aligned majority.
    data = good_data(prof)
    d = data.topics[cam["depth_topic"]]
    data.topics[cam["depth_topic"]] = topic(d.header_stamps[:20] + d.header_stamps[21:], d.samples)
    assert status(doctor.run_checks(prof, data), "rgb_depth_sync") == doctor.OK
    # A real offset of one whole period, same number of frames: still a failure.
    assert status(doctor.run_checks(prof, good_data(prof, rgb_offset=0.050)), "rgb_depth_sync") == doctor.FAIL


def test_lidar_camera_offset_at_different_rates_says_what_it_cannot_see() -> None:
    prof = load_profile(PROFILES / "aerial_lidar_camera_imu.yaml")
    data = good_data(prof)
    lid_t = [i / 10 + 1000.0 for i in range(40)]
    data.topics[prof["sensors"]["lidar"]["topic"]] = topic(lid_t, [ns(header=hdr("lidar", 1000.0))])
    c = next(c for c in doctor.run_checks(prof, data) if c.name == "lidar_camera_sync")
    assert c.status == doctor.OK
    assert "modulo the slower period" in c.detail and "100 ms" in c.detail


def test_a_planar_scanner_is_listened_to_as_a_laser_scan(prof) -> None:  # type: ignore[no-untyped-def]
    # lidar.type: 2d reads sensor_msgs/LaserScan (ADR-0066); 3d, the default, a PointCloud2.
    prof = dict(prof)
    prof["sensors"] = dict(prof["sensors"], lidar={"enabled": True, "topic": "/points"})
    assert doctor.topics_for_profile(prof)["/points"] == "sensor_msgs/msg/PointCloud2"
    prof["sensors"] = dict(prof["sensors"], lidar={"enabled": True, "topic": "/scan", "type": "2d"})
    assert doctor.topics_for_profile(prof)["/scan"] == "sensor_msgs/msg/LaserScan"


def test_external_odometry_topic_is_listened_to_and_its_tf_frame_checked(prof, tmp_path) -> None:  # type: ignore[no-untyped-def]
    prof = dict(prof)
    prof["sensors"] = dict(prof["sensors"], external_odometry={"topic": "/wheel_odom"})
    assert doctor.topics_for_profile(prof)["/wheel_odom"] == "nav_msgs/msg/Odometry"
    data = good_data(prof)
    assert status(doctor.run_checks(prof, data), "topics") == doctor.FAIL  # no messages on /wheel_odom
    data.topics["/wheel_odom"] = topic([1000.0, 1000.1, 1000.2], [])
    assert status(doctor.run_checks(prof, data), "topics") == doctor.OK
    prof["sensors"] = dict(prof["sensors"], external_odometry={"frame": "odom"})
    assert "/wheel_odom" not in doctor.topics_for_profile(prof)
    c = next(c for c in doctor.run_checks(prof, good_data(prof)) if c.name == "topics")
    assert c.status == doctor.FAIL and "TF odom -> base_link" in c.detail
    data = good_data(prof)
    data.tf_dynamic = [tf("odom", "base_link")]
    assert status(doctor.run_checks(prof, data), "topics") == doctor.OK


def test_no_imu_section_means_no_imu(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # As the core: no `imu` section = no IMU; an empty one is an IMU with its defaults (audit A8-05, ADR-0053).
    from strataslam_tools.profile import enabled

    p = tmp_path / "p.yaml"
    p.write_text("sensors:\n  camera: {rgb_topic: /c, depth_topic: /d, info_topic: /i}\n")
    assert not enabled(load_profile(p), "imu")
    p.write_text("sensors:\n  camera: {rgb_topic: /c, depth_topic: /d, info_topic: /i}\n  imu:\n")
    assert enabled(load_profile(p), "imu")


def test_profile_row_reports_an_aerial_platform_without_imu(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import builtins
    real_import = builtins.__import__

    def no_ros(name, *a, **k):  # type: ignore[no-untyped-def]
        if name == "strataslam_ros":
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_ros)
    c = doctor.check_profile("p.yaml", {"robot": {"platform": "aerial"}, "sensors": {}})
    assert c.status == doctor.WARN and "aerial" in c.detail
    c = doctor.check_profile("p.yaml", {"robot": {"platform": "ground"}, "sensors": {}})
    assert c.status == doctor.WARN and "not validated" in c.detail


def test_imu_labels_inverted_and_sideways(prof) -> None:  # type: ignore[no-untyped-def]
    # Audit A8-10: the two labels were swapped.
    c = next(c for c in doctor.run_checks(prof, good_data(prof, imu_accel=(0, 0, -9.81))) if c.name == "imu_rest")
    assert c.status == doctor.FAIL and "points down" in c.detail
    c = next(c for c in doctor.run_checks(prof, good_data(prof, imu_accel=(9.81, 0, 0))) if c.name == "imu_rest")
    assert c.status == doctor.FAIL and "sideways" in c.detail


def test_doctor_cannot_run_is_exit_code_3(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import argparse

    from strataslam_tools import cli
    args = argparse.Namespace(profile=str(PROFILES / "ground_lidar.yaml"), bag=str(tmp_path / "missing"), json=False, seconds=1.0)
    assert cli.cmd_doctor(args) == 3
    args = argparse.Namespace(profile=str(tmp_path / "missing.yaml"), bag=str(tmp_path), json=False, seconds=1.0)
    assert cli.cmd_doctor(args) == 3


def test_sections_with_every_key_commented_out_read_as_the_core_reads_them(tmp_path: Path) -> None:
    # The templates comment out whole sections (`output:` with nothing under it): YAML null. The core reads a null
    # section as present and empty; the doctor crashed on it (`None.get`) and called an empty sensor disabled.
    from strataslam_tools.profile import enabled

    p = tmp_path / "p.yaml"
    p.write_text("strataslam: {version: 1}\nsensors:\n  lidar:\n  camera: {enabled: false}\noutput:\nruntime:\n")
    prof = load_profile(p)
    assert prof["output"] == {} and prof["runtime"] == {}
    assert enabled(prof, "lidar") and not enabled(prof, "camera") and not enabled(prof, "imu")
    assert doctor.check_disk(prof).name == "disk"
    for name in ("ground_camera", "ground_lidar", "ground_lidar_camera_imu", "handheld_camera_imu",
                 "aerial_lidar_camera_imu"):
        prof = load_profile(PROFILES / f"{name}.yaml")
        assert doctor.check_disk(prof).name == "disk", name
