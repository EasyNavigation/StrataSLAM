"""`strataslam doctor`: checks sensors, TF and synchronization before mapping.

The checks are pure functions over simple message-like objects (the ROS messages themselves, or any
object with the same attributes), so they are tested without ROS. Adapters read a rosbag2 or a live
robot. Each check returns OK, WARN or FAIL with a hint; the exit code is 0 (all OK), 1 (some warning)
or 2 (some failure).
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from strataslam_tools.profile import enabled, load_profile

Mat = npt.NDArray[np.float64]

OK, WARN, FAIL = "ok", "warn", "fail"
SYMBOL = {OK: "✔", WARN: "⚠", FAIL: "✘"}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    hint: str = ""


@dataclass
class TopicData:
    """What was heard on one topic: header stamps (s), receive times (s) and a sample of messages."""

    type: str = ""
    header_stamps: list[float] = field(default_factory=list)
    recv_times: list[float] = field(default_factory=list)
    samples: list[Any] = field(default_factory=list)


@dataclass
class DoctorData:
    topics: dict[str, TopicData] = field(default_factory=dict)
    tf_static: list[Any] = field(default_factory=list)   # geometry_msgs/TransformStamped from /tf_static
    tf_dynamic: list[Any] = field(default_factory=list)  # from /tf
    use_sim_time: bool | None = None                      # live only
    wall_clock_now: float | None = None                   # live only

    def topic(self, name: str) -> TopicData | None:
        t = self.topics.get(name)
        return t if t and (t.header_stamps or t.recv_times) else None


# ---------------------------------------------------------------- helpers

def rate_hz(stamps: list[float]) -> float:
    if len(stamps) < 2:
        return 0.0
    span = stamps[-1] - stamps[0]
    return (len(stamps) - 1) / span if span > 0 else 0.0


def _grade(value: float, ok: float, warn: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return OK if value >= ok else WARN if value >= warn else FAIL
    return OK if value <= ok else WARN if value <= warn else FAIL


def mean_offset(a: list[float], b: list[float]) -> float:
    """Typical absolute clock offset between two streams (seconds).

    Streams of the same rate are paired by index: matching each stamp with the nearest of the other stream would hide
    an offset of one whole period (50 ms at 20 Hz). When one stream has a few messages more (a frame dropped by the
    driver or not recorded), the pairing is tried with each shift up to that difference and the shift whose pairs
    agree best wins; the median of the pair differences ignores the pairs a drop in the middle misaligns. Streams at
    different rates are matched by nearest stamp, which cannot see more than half the slower period.
    """
    if not a or not b:
        return math.inf
    aa, bb = np.asarray(sorted(a)), np.asarray(sorted(b))
    if len(aa) > 2 and abs(len(aa) - len(bb)) <= 2:
        best = math.inf
        for k in range(-abs(len(aa) - len(bb)), abs(len(aa) - len(bb)) + 1):
            x, y = (aa[k:], bb) if k >= 0 else (aa, bb[-k:])
            n = min(len(x), len(y))
            if n > 2:
                best = min(best, float(np.median(np.abs(x[:n] - y[:n]))))
        return best
    idx = np.searchsorted(bb, aa)
    nearest = []
    for t, i in zip(aa, idx):
        cand = [bb[j] for j in (i - 1, i) if 0 <= j < len(bb)]
        nearest.append(min(abs(t - c) for c in cand))
    return float(np.median(nearest))


def same_rate(a: list[float], b: list[float]) -> bool:
    return len(a) > 2 and abs(len(a) - len(b)) <= 2

def _quat_to_rot(q: Any) -> Mat:
    x, y, z, w = q.x, q.y, q.z, q.w
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class TfGraph:
    """Static/dynamic transform tree built from TF messages, to answer 'is there a path and is it static'."""

    def __init__(self, static: Iterable[Any], dynamic: Iterable[Any]) -> None:
        self.edges: dict[tuple[str, str], tuple[Mat, bool]] = {}
        for is_static, items in ((True, static), (False, dynamic)):
            for t in items:
                R = _quat_to_rot(t.transform.rotation)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
                self.edges[(t.header.frame_id, t.child_frame_id)] = (T, is_static)

    def path(self, a: str, b: str) -> tuple[Mat, bool] | None:
        """Transform of frame b expressed in frame a, and whether every edge is static."""
        if a == b:
            return np.eye(4), True
        frontier: list[tuple[str, Mat, bool]] = [(a, np.eye(4), True)]
        seen = {a}
        while frontier:
            cur, T, st = frontier.pop()
            for (p, c), (E, es) in self.edges.items():
                for nxt, M in ((c, E), (p, np.linalg.inv(E))):
                    src = p if nxt == c else c
                    if src != cur or nxt in seen:
                        continue
                    if nxt == b:
                        return T @ M, st and es
                    seen.add(nxt)
                    frontier.append((nxt, T @ M, st and es))
        return None


# ---------------------------------------------------------------- the checks

def check_topics(prof: dict[str, Any], data: DoctorData) -> Check:
    s = prof["sensors"]
    wanted: list[str] = []
    if enabled(prof, "camera"):
        wanted += [s["camera"].get(k, "") for k in ("rgb_topic", "depth_topic", "info_topic")]
    if enabled(prof, "lidar"):
        wanted.append(s["lidar"].get("topic", "/points"))
    if enabled(prof, "imu"):
        wanted.append(s["imu"].get("topic", "/imu"))
    if enabled(prof, "barometer"):
        wanted.append(s["barometer"].get("topic", "/pressure"))
        wanted.append(s["barometer"].get("reference_topic", ""))
    ext = s.get("external_odometry") or {}
    if enabled(prof, "external_odometry") and ext.get("topic"):
        wanted.append(ext["topic"])
    missing = [t for t in wanted if t and data.topic(t) is None]
    if enabled(prof, "external_odometry") and ext.get("frame"):
        # `frame:` = the wheel odometry is the TF odom frame -> base frame, published on /tf.
        base = prof.get("robot", {}).get("base_frame", "base_link")
        if TfGraph(data.tf_static, data.tf_dynamic).path(ext["frame"], base) is None:
            missing.append(f"TF {ext['frame']} -> {base} (external_odometry.frame)")
    if missing:
        return Check("topics", FAIL, f"no data on: {', '.join(missing)}", "check the driver and the topic names in the profile")
    return Check("topics", OK, f"{len(wanted)} topic(s) publishing")


def _freq_check(name: str, topic: str, data: DoctorData, ok: float, warn: float) -> Check:
    t = data.topic(topic)
    if t is None:
        return Check(name, FAIL, f"{topic}: no data", "start the driver")
    stamps = t.header_stamps or t.recv_times
    f = rate_hz(sorted(stamps))
    # 2 % of tolerance: a sensor running exactly at the threshold measures a hair below it (a 15 Hz
    # camera read 14.99 Hz on the simulator's bag and was graded a warning).
    status = _grade(f, 0.98 * ok, warn)
    hint = "" if status == OK else f"expected >= {ok:g} Hz; lower the resolution or fix the transport"
    return Check(name, status, f"{topic}: {f:.1f} Hz", hint)


def check_camera_info(prof: dict[str, Any], data: DoctorData) -> Check:
    topic = prof["sensors"]["camera"].get("info_topic", "")
    info = data.topic(topic)
    img = data.topic(prof["sensors"]["camera"].get("rgb_topic", ""))
    if info is None or not info.samples:
        return Check("camera_info", FAIL, "camera_info is missing", "the driver must publish camera_info with a valid K")
    ci = info.samples[0]
    K = list(ci.k)
    # `not (x > 0)` rather than `x <= 0`: NaN fails every comparison (some drivers publish K full of NaN).
    if len(K) != 9 or not all(math.isfinite(k) for k in K) or not (K[0] > 0 and K[4] > 0):
        return Check("camera_info", FAIL, "K is null or invalid", "calibrate the camera or use the driver's factory calibration")
    if img and img.samples and (img.samples[0].width != ci.width or img.samples[0].height != ci.height):
        return Check("camera_info", FAIL, f"camera_info is {ci.width}x{ci.height} but the image is {img.samples[0].width}x{img.samples[0].height}",
                     "publish camera_info for the resolution actually streamed")
    if ci.distortion_model not in ("plumb_bob", "rational_polynomial", "equidistant"):
        return Check("camera_info", WARN, f"non-standard distortion model '{ci.distortion_model}'", "rectify the images or use plumb_bob")
    return Check("camera_info", OK, f"fx={K[0]:.1f}, {ci.width}x{ci.height}, {ci.distortion_model}")


def check_depth_encoding(prof: dict[str, Any], data: DoctorData) -> Check:
    t = data.topic(prof["sensors"]["camera"].get("depth_topic", ""))
    if t is None or not t.samples:
        return Check("depth_encoding", FAIL, "no depth image", "start the depth stream")
    enc = t.samples[0].encoding
    if enc not in ("16UC1", "32FC1"):
        return Check("depth_encoding", FAIL, f"encoding {enc}", "publish depth as 16UC1 (mm) or 32FC1 (m)")
    return Check("depth_encoding", OK, enc)


def check_depth_registered(prof: dict[str, Any], data: DoctorData) -> Check:
    cam = prof["sensors"]["camera"]
    d, c = data.topic(cam.get("depth_topic", "")), data.topic(cam.get("rgb_topic", ""))
    if d is None or c is None or not d.samples or not c.samples:
        return Check("depth_registered", FAIL, "cannot compare depth and color frames", "start both streams")
    same_frame = d.samples[0].header.frame_id == c.samples[0].header.frame_id
    aligned = "aligned" in cam.get("depth_topic", "")
    if same_frame or aligned:
        return Check("depth_registered", OK, f"depth in frame {d.samples[0].header.frame_id}")
    return Check("depth_registered", FAIL, f"depth frame {d.samples[0].header.frame_id} != color frame {c.samples[0].header.frame_id}",
                 "enable depth alignment to color in the driver (align_depth)")


def check_valid_depth(prof: dict[str, Any], data: DoctorData) -> Check:
    cam = prof["sensors"]["camera"]
    t = data.topic(cam.get("depth_topic", ""))
    if t is None or not t.samples:
        return Check("valid_depth", FAIL, "no depth image", "start the depth stream")
    lo, hi = float(cam.get("depth_min", 0.3)), float(cam.get("depth_max", 6.0))
    ratios = []
    for m in t.samples[:10]:
        if m.encoding == "32FC1":
            v = np.frombuffer(bytes(m.data), dtype=np.float32)
        elif m.encoding == "16UC1":
            v = np.frombuffer(bytes(m.data), dtype=np.uint16).astype(np.float32) * 1e-3
        else:
            return Check("valid_depth", FAIL, f"cannot read encoding {m.encoding}", "publish 16UC1 or 32FC1")
        ratios.append(float(np.mean(np.isfinite(v) & (v >= lo) & (v <= hi))))
    pct = 100 * float(np.mean(ratios))
    status = _grade(pct, 50, 20)
    return Check("valid_depth", status, f"{pct:.0f} % of pixels in [{lo:g}, {hi:g}] m",
                 "" if status == OK else "point the camera at a textured scene within range, or check the projector")


def check_rgb_depth_sync(prof: dict[str, Any], data: DoctorData) -> Check:
    cam = prof["sensors"]["camera"]
    d, c = data.topic(cam.get("depth_topic", "")), data.topic(cam.get("rgb_topic", ""))
    if d is None or c is None:
        return Check("rgb_depth_sync", FAIL, "missing stream", "start both streams")
    off_ms = 1e3 * mean_offset(sorted(d.header_stamps), sorted(c.header_stamps))
    status = _grade(off_ms, 10, 30, higher_is_better=False)
    return Check("rgb_depth_sync", status, f"{off_ms:.1f} ms", "" if status == OK else "enable hardware sync or a common clock for color and depth")


def check_lidar_camera_sync(prof: dict[str, Any], data: DoctorData) -> Check:
    lid, cam = data.topic(prof["sensors"]["lidar"].get("topic", "/points")), data.topic(prof["sensors"]["camera"].get("rgb_topic", ""))
    if lid is None or cam is None:
        return Check("lidar_camera_sync", FAIL, "missing stream", "start both sensors")
    off_ms = 1e3 * mean_offset(sorted(lid.header_stamps), sorted(cam.header_stamps))
    status = _grade(off_ms, 20, 50, higher_is_better=False)
    detail = f"{off_ms:.1f} ms"
    if not same_rate(lid.header_stamps, cam.header_stamps):
        # At different rates the stamps only tell the offset modulo the slower period (100 ms for a 10 Hz LiDAR).
        slow = 1e3 / max(1e-9, min(rate_hz(sorted(lid.header_stamps)), rate_hz(sorted(cam.header_stamps))))
        detail += f" (modulo the slower period, {slow:.0f} ms: a whole-period offset cannot be seen from the stamps)"
    return Check("lidar_camera_sync", status, detail, "" if status == OK else "synchronize the sensors' clocks (PTP / hardware trigger)")


def frames_of(prof: dict[str, Any], data: DoctorData) -> dict[str, str]:
    """Frame id of each sensor, read from its messages."""
    out: dict[str, str] = {}
    for key, topic in (
        ("camera", prof["sensors"].get("camera", {}).get("depth_topic", "")),
        ("lidar", prof["sensors"].get("lidar", {}).get("topic", "")),
        ("imu", prof["sensors"].get("imu", {}).get("topic", "")),
    ):
        t = data.topic(topic) if topic else None
        if t and t.samples:
            out[key] = t.samples[0].header.frame_id
    return out


def check_tf_chain(prof: dict[str, Any], data: DoctorData) -> Check:
    graph = TfGraph(data.tf_static, data.tf_dynamic)
    base = prof.get("robot", {}).get("base_frame", "base_link")
    problems: list[str] = []
    moving: list[str] = []
    ok: list[str] = []
    for sensor, frame in frames_of(prof, data).items():
        res = graph.path(base, frame)
        if res is None:
            problems.append(f"{base} -> {frame} ({sensor}) is missing")
        elif not res[1]:
            moving.append(f"{base} -> {frame} ({sensor})")
        else:
            ok.append(sensor)
    if problems:
        return Check("tf_chain", FAIL, "; ".join(problems), "publish the sensor mounting transforms (robot_state_publisher / static_transform_publisher)")
    if moving:
        # A chain through joints (a pan-tilt head, a torso lift) exists but is published on /tf. The node reads each
        # sensor's mounting once, when the pipeline starts: it is right as long as those joints do not move.
        return Check("tf_chain", WARN, f"through moving joints (/tf): {'; '.join(moving)}",
                     "the node reads the mounting once at start: keep those joints still while mapping")
    if not ok:
        return Check("tf_chain", WARN, "not checked: no sensor message to read the frames from", "start the drivers (or record them in the bag)")
    return Check("tf_chain", OK, f"static transforms for {', '.join(ok)}")


def check_lidar_camera_extrinsic(prof: dict[str, Any], data: DoctorData) -> Check:
    f = frames_of(prof, data)
    res = TfGraph(data.tf_static, data.tf_dynamic).path(f.get("camera", ""), f.get("lidar", "")) if "camera" in f and "lidar" in f else None
    if res is None:
        return Check("lidar_camera_extrinsic", FAIL, "no transform between the camera and the LiDAR", "calibrate and publish the LiDAR-camera extrinsic")
    return Check("lidar_camera_extrinsic", OK, "transform present")


def check_imu_rest(prof: dict[str, Any], data: DoctorData, seconds: float = 3.0) -> Check:
    t = data.topic(prof["sensors"]["imu"].get("topic", "/imu"))
    if t is None or not t.samples:
        return Check("imu_rest", FAIL, "no IMU data", "start the IMU")
    stamps = t.header_stamps
    t0 = stamps[0] if stamps else 0.0
    window = [m for m, s in zip(t.samples, stamps) if s - t0 <= seconds] or t.samples
    a = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for m in window])
    w = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for m in window])
    mean_a = a.mean(0)
    norm = float(np.linalg.norm(mean_a))
    base = prof.get("robot", {}).get("base_frame", "base_link")
    f = frames_of(prof, data).get("imu", "")
    res = TfGraph(data.tf_static, data.tf_dynamic).path(base, f) if f else None
    R = res[0][:3, :3] if res is not None else np.eye(3)
    up_base = R @ (mean_a / (norm or 1))
    if abs(norm - 9.81) > 1.5:
        return Check("imu_rest", FAIL, f"|a| = {norm:.2f} m/s^2", "the IMU must publish m/s^2 (not g) and include gravity")
    if up_base[2] < 0.7:
        # The labels were swapped (audit A8-10): a reading pointing down is an inverted axis, one in between is sideways.
        return Check("imu_rest", FAIL, f"gravity points {'down' if up_base[2] <= -0.7 else 'sideways'} in {base} (up = {np.round(up_base, 2).tolist()})",
                     "an axis is inverted or the mounting TF is wrong: at rest the accelerometer must read +z in the base frame")
    moving = float(np.linalg.norm(w.mean(0))) > 0.3
    status = OK if abs(norm - 9.81) <= 0.3 else WARN
    detail = f"|a| = {norm:.2f} m/s^2, up = {np.round(up_base, 2).tolist()}" + (" (the robot was moving)" if moving else "")
    return Check("imu_rest", status, detail, "" if status == OK else "check the accelerometer scale / calibration")


def check_barometer(prof: dict[str, Any], data: DoctorData) -> Check:
    t = data.topic(prof["sensors"]["barometer"].get("topic", "/pressure"))
    if t is None or not t.samples:
        return Check("barometer", FAIL, "no pressure data", "start the barometer")
    p = np.array([m.fluid_pressure for m in t.samples[:200]])
    mean, sd = float(p.mean()), float(p.std())
    if not 80e3 <= mean <= 110e3:
        return Check("barometer", FAIL, f"{mean / 1e3:.1f} kPa", "the barometer must publish Pa (80-110 kPa at sea level)")
    if sd > 20.0:
        return Check("barometer", WARN, f"{mean / 1e3:.1f} kPa, noise {sd:.1f} Pa", "high noise: filter or shield the sensor from airflow")
    return Check("barometer", OK, f"{mean / 1e3:.1f} kPa, noise {sd:.2f} Pa")


def check_time(prof: dict[str, Any], data: DoctorData) -> Check:
    problems = []
    for name, t in data.topics.items():
        s = t.header_stamps
        if not s:
            continue
        if min(s) <= 0:
            problems.append(f"{name}: zero stamps")
        elif any(b < a for a, b in itertools.pairwise(s)):
            problems.append(f"{name}: stamps go backwards")
    if data.use_sim_time is not None and data.wall_clock_now is not None:
        recent = [s[-1] for t in data.topics.values() if (s := t.header_stamps)]
        if recent:
            skew = abs(recent[0] - data.wall_clock_now)
            if not data.use_sim_time and skew > 60:
                problems.append("use_sim_time is false but the stamps are far from the wall clock")
            if data.use_sim_time and skew < 5:
                problems.append("use_sim_time is true but the stamps look like wall-clock time")
    if problems:
        return Check("time", FAIL, "; ".join(problems), "fix the drivers' clocks; use_sim_time must match the data source")
    return Check("time", OK, "stamps monotonic")


def check_disk(prof: dict[str, Any], map_dir: str | None = None) -> Check:
    path = Path(map_dir or prof.get("output", {}).get("map_dir", "~/strataslam_maps")).expanduser()
    while not path.exists() and path != path.parent:
        path = path.parent
    free_gb = shutil.disk_usage(path).free / 1e9
    status = _grade(free_gb, 10, 2)
    return Check("disk", status, f"{free_gb:.1f} GB free at {path}", "" if status == OK else "free some space or change output.map_dir")


# ---------------------------------------------------------------- orchestration

def check_profile(profile_path: str | None, prof: dict[str, Any]) -> Check:
    """The profile, validated by the core's own code when it is installed (strataslam_ros), as the launch does."""
    platform = prof.get("robot", {}).get("platform", "handheld")
    try:
        from strataslam_ros import validate_profile  # type: ignore[import-not-found, import-untyped, unused-ignore]  # ROS package
    except ImportError:
        validate_profile = None
    errors: list[str] = []
    warnings: list[str] = []
    if validate_profile is not None and profile_path:
        errors, warnings = validate_profile(str(profile_path))
    elif platform == "aerial" and not enabled(prof, "imu"):
        warnings = ["robot.platform: aerial platform without an IMU: the flight estimate is not reliable (strong warning)"]
    if errors:
        return Check("profile", FAIL, "; ".join(errors), "fix the profile: the launch would refuse it")
    if warnings:
        return Check("profile", WARN, "; ".join(warnings), "")
    if validate_profile is None:
        return Check("profile", WARN, "not validated: strataslam_ros (the core's validator) is not installed",
                     "source the ROS workspace to validate the profile here")
    return Check("profile", OK, "valid")


def run_checks(prof: dict[str, Any], data: DoctorData, map_dir: str | None = None, profile_path: str | None = None) -> list[Check]:
    checks = [check_profile(profile_path, prof)] if profile_path else []
    checks.append(check_topics(prof, data))
    if enabled(prof, "camera"):
        cam = prof["sensors"]["camera"]
        checks += [
            _freq_check("camera_rate", cam.get("rgb_topic", ""), data, 15, 5),
            check_camera_info(prof, data),
            check_depth_encoding(prof, data),
            check_depth_registered(prof, data),
            check_valid_depth(prof, data),
            check_rgb_depth_sync(prof, data),
        ]
    if enabled(prof, "lidar"):
        checks.append(_freq_check("lidar_rate", prof["sensors"]["lidar"].get("topic", "/points"), data, 5, 2))
    if enabled(prof, "imu"):
        checks.append(_freq_check("imu_rate", prof["sensors"]["imu"].get("topic", "/imu"), data, 100, 50))
        checks.append(check_imu_rest(prof, data))
    if enabled(prof, "barometer"):
        checks.append(_freq_check("barometer_rate", prof["sensors"]["barometer"].get("topic", "/pressure"), data, 5, 1))
        checks.append(check_barometer(prof, data))
    checks.append(check_tf_chain(prof, data))
    if enabled(prof, "camera") and enabled(prof, "lidar"):
        checks += [check_lidar_camera_sync(prof, data), check_lidar_camera_extrinsic(prof, data)]
    checks += [check_time(prof, data), check_disk(prof, map_dir)]
    return checks


def exit_code(checks: list[Check]) -> int:
    return 2 if any(c.status == FAIL for c in checks) else 1 if any(c.status == WARN for c in checks) else 0


def format_table(checks: list[Check]) -> str:
    w = max(len(c.name) for c in checks)
    lines = []
    for c in checks:
        lines.append(f"{SYMBOL[c.status]} {c.name.ljust(w)}  {c.detail}")
        if c.hint and c.status != OK:
            lines.append(f"  {' ' * w}  -> {c.hint}")
    return "\n".join(lines)


def to_json(checks: list[Check]) -> str:
    return json.dumps({"exit_code": exit_code(checks), "checks": [asdict(c) for c in checks]}, indent=2)


# ---------------------------------------------------------------- adapters (need ROS 2)

def _stamp_s(msg: Any) -> float | None:
    h = getattr(msg, "header", None)
    return None if h is None else h.stamp.sec + h.stamp.nanosec * 1e-9


def collect_from_bag(bag_path: str, topics: set[str], max_samples: int = 400) -> DoctorData:
    """Reads a rosbag2 (the first `max_samples` messages of each topic are kept as samples)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage = rosbag2_py.StorageOptions(uri=bag_path)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    data = DoctorData()
    wanted = topics | {"/tf_static", "/tf"}
    msg_cls = {n: get_message(types[n]) for n in wanted if n in types}
    while reader.has_next():
        name, raw, t_ns = reader.read_next()
        if name not in msg_cls:
            continue
        msg = deserialize_message(raw, msg_cls[name])
        if name == "/tf_static":
            data.tf_static += list(msg.transforms)
        elif name == "/tf":
            data.tf_dynamic += list(msg.transforms)
        else:
            td = data.topics.setdefault(name, TopicData(type=types[name]))
            s = _stamp_s(msg)
            if s is not None:
                td.header_stamps.append(s)
            td.recv_times.append(t_ns * 1e-9)
            if len(td.samples) < max_samples:
                td.samples.append(msg)
    return data


def collect_live(topics: dict[str, str], seconds: float = 6.0) -> DoctorData:
    """Listens to a live robot for `seconds` (rclpy). `topics` maps topic name -> message type string."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rosidl_runtime_py.utilities import get_message
    from tf2_msgs.msg import TFMessage

    rclpy.init()
    node = rclpy.create_node("strataslam_doctor")
    data = DoctorData()

    def make_cb(name: str) -> Any:
        def cb(msg: Any) -> None:
            td = data.topics.setdefault(name, TopicData(type=topics[name]))
            s = _stamp_s(msg)
            if s is not None:
                td.header_stamps.append(s)
            td.recv_times.append(time.time())
            if len(td.samples) < 400:
                td.samples.append(msg)
        return cb

    for name, typ in topics.items():
        node.create_subscription(get_message(typ), name, make_cb(name), qos_profile_sensor_data)
    from rclpy.qos import DurabilityPolicy, QoSProfile

    node.create_subscription(TFMessage, "/tf_static", lambda m: data.tf_static.extend(cast(Any, m).transforms),
                             QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node.create_subscription(TFMessage, "/tf", lambda m: data.tf_dynamic.extend(cast(Any, m).transforms), 100)
    end = time.time() + seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    data.use_sim_time = bool(node.get_parameter("use_sim_time").value) if node.has_parameter("use_sim_time") else False
    data.wall_clock_now = time.time()
    node.destroy_node()
    rclpy.shutdown()
    return data


def topics_for_profile(prof: dict[str, Any]) -> dict[str, str]:
    s = prof["sensors"]
    out: dict[str, str] = {}
    if enabled(prof, "camera"):
        out[s["camera"]["rgb_topic"]] = "sensor_msgs/msg/Image"
        out[s["camera"]["depth_topic"]] = "sensor_msgs/msg/Image"
        out[s["camera"]["info_topic"]] = "sensor_msgs/msg/CameraInfo"
    if enabled(prof, "lidar"):
        planar = str(s["lidar"].get("type", "3d")) == "2d"  # a planar scanner (ADR-0066)
        out[s["lidar"].get("topic", "/points")] = "sensor_msgs/msg/LaserScan" if planar else "sensor_msgs/msg/PointCloud2"
    if enabled(prof, "imu"):
        out[s["imu"].get("topic", "/imu")] = "sensor_msgs/msg/Imu"
    if enabled(prof, "barometer"):
        out[s["barometer"].get("topic", "/pressure")] = "sensor_msgs/msg/FluidPressure"
        if s["barometer"].get("reference_topic"):
            out[s["barometer"]["reference_topic"]] = "sensor_msgs/msg/FluidPressure"
    ext = s.get("external_odometry") or {}
    if enabled(prof, "external_odometry") and ext.get("topic"):
        out[ext["topic"]] = "nav_msgs/msg/Odometry"
    return out


def run(profile_path: str, bag: str | None = None, as_json: bool = False, seconds: float = 6.0) -> tuple[int, str]:
    prof = load_profile(profile_path)
    topics = topics_for_profile(prof)
    data = collect_from_bag(bag, set(topics)) if bag else collect_live(topics, seconds)
    checks = run_checks(prof, data, profile_path=profile_path)
    return exit_code(checks), to_json(checks) if as_json else format_table(checks)
