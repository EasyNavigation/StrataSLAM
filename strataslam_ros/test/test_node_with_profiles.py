"""The node runs with every example profile on a rosbag from the simulator: odometry at >= 10 Hz,
`~/save_map` writes a valid map, `~/mark_floor_change` opens a floor."""

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from ros_helpers import PROFILES, export_bag, profile_with_map_dir
from std_srvs.srv import Trigger

from strataslam_msgs.msg import FloorState
from strataslam_msgs.srv import MarkFloorChange
from strataslam_tools import mapio


class Probe(Node):
    def __init__(self) -> None:
        super().__init__("strataslam_test_probe")
        self.odom_stamps: list[float] = []
        self.states: list[FloorState] = []
        self.create_subscription(Odometry, "/strataslam/odometry", lambda m: self.odom_stamps.append(time.monotonic()), 100)
        self.create_subscription(FloorState, "/strataslam/floor_state", self.states.append, 100)
        self.save = self.create_client(Trigger, "/strataslam_node/save_map")
        self.mark = self.create_client(MarkFloorChange, "/strataslam_node/mark_floor_change")
        self.session = self.create_client(Trigger, "/strataslam_node/new_session")

    def call(self, client, request, timeout=180.0):  # type: ignore[no-untyped-def]
        # Generous: saving a map on a loaded machine took more than 60 s once (audit AUD-017).
        assert client.wait_for_service(timeout_sec=timeout)
        fut = client.call_async(request)
        end = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < end:
            time.sleep(0.05)
        assert fut.done(), "service call timed out"
        return fut.result()


@pytest.fixture(scope="module", autouse=True)
def ros_context():  # type: ignore[no-untyped-def]
    # A ROS domain of its own (audit AUD-046): another test, a running robot or a parallel run cannot answer here.
    os.environ["ROS_DOMAIN_ID"] = str(10 + (os.getpid() + 71) % 80)
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.mark.parametrize("name", ["ground_camera", "ground_lidar", "ground_lidar_camera_imu", "handheld_camera_imu",
                                  "aerial_lidar_camera_imu"])
def test_node_publishes_odometry_and_saves_a_valid_map(name: str, tmp_path: Path) -> None:
    bag_dir = tmp_path / "sim"
    export_bag(bag_dir, PROFILES / f"{name}.yaml", seconds=25.0)
    if name == "ground_camera":  # the bag carries the wheel odometry the profile asks for: TF odom -> base_link
        assert re.search(r"name: /tf$", (bag_dir / "bag" / "metadata.yaml").read_text(), re.MULTILINE)
    map_dir = tmp_path / "map"
    profile = profile_with_map_dir(PROFILES / f"{name}.yaml", map_dir, tmp_path / "profiles" / f"{name}.yaml")
    # Where the vocabulary comes from: the profile, as a path relative to it (handheld_camera_imu) or as `default`, the
    # one shipped (ground_camera, ground_lidar_camera_imu); nowhere (aerial_lidar_camera_imu: the
    # node must say at the end that no visual loop could close); or the launch (ground_lidar). ground_camera also runs
    # with the wheel odometry the bag carries as TF (odom -> base_link).
    vocabulary = []
    text = profile.read_text()
    line = next(ln for ln in text.splitlines() if ln.lstrip().startswith(("vocabulary:", "# vocabulary:")))
    if name == "handheld_camera_imu":
        shutil.copy(bag_dir / "vocab.bin", profile.parent / "vocab.bin")
        profile.write_text(text.replace(line, "  vocabulary: vocab.bin", 1))
    elif name == "aerial_lidar_camera_imu":
        profile.write_text(text.replace(line, "", 1))
    elif name == "ground_lidar":
        vocabulary = [f"vocabulary:={bag_dir / 'vocab.bin'}"]

    log_path = tmp_path / "launch.log"
    with open(log_path, "w") as log:  # a file, not a pipe nobody reads: a full pipe would block the node
        launch = subprocess.Popen(
            ["ros2", "launch", "strataslam_ros", "strataslam.launch.py", f"profile:={profile}", *vocabulary,
             "features_topic:=/strataslam/features"],
            stdout=log, stderr=subprocess.STDOUT, text=True, env={**os.environ, "RCUTILS_COLORIZED_OUTPUT": "0"},
            start_new_session=True)
    node = Probe()
    stop = threading.Event()
    spinner = threading.Thread(target=lambda: [rclpy.spin_once(node, timeout_sec=0.05) for _ in iter(stop.is_set, True)], daemon=True)
    spinner.start()
    try:
        time.sleep(3.0)  # let the container come up
        player = subprocess.Popen(["ros2", "bag", "play", str(bag_dir / "bag")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        player.wait(timeout=120)
        time.sleep(2.0)
        stamps = node.odom_stamps
        assert len(stamps) > 100, f"only {len(stamps)} odometry messages"
        steady = stamps[len(stamps) // 4:]
        rate = (len(steady) - 1) / (steady[-1] - steady[0])
        assert rate >= 10.0, f"odometry at {rate:.1f} Hz"
        assert node.states, "no floor_state published"
        assert node.states[-1].num_floors >= 1
        # QoS of section 11.1: odometry, path and floor_state reliable, floor_cloud transient local.
        for topic in ("/strataslam/odometry", "/strataslam/path", "/strataslam/floor_state"):
            infos = node.get_publishers_info_by_topic(topic)
            assert infos and all(i.qos_profile.reliability == ReliabilityPolicy.RELIABLE for i in infos), topic
        infos = node.get_publishers_info_by_topic("/strataslam/floor_cloud")
        assert infos and all(i.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for i in infos)

        res = node.call(node.save, Trigger.Request())
        assert res.success, res.message
        assert mapio.validate_map(map_dir) == []
        manifest = mapio.read_manifest(map_dir)
        assert len(manifest["floors"]) >= 1

        if name == "handheld_camera_imu":  # manual floor change
            r = node.call(node.mark, MarkFloorChange.Request(transition_type="elevator", target_floor=-1))
            assert r.success and r.floor == 1
            end = time.monotonic() + 15.0  # the next floor_state comes with the next frame
            while node.states[-1].floor not in (1, -1) and time.monotonic() < end:
                time.sleep(0.2)
            assert node.states[-1].floor in (1, -1)
            bad = node.call(node.mark, MarkFloorChange.Request(transition_type="teleport", target_floor=-1))
            assert not bad.success
        r = node.call(node.session, Trigger.Request())  # ~/new_session (section 11.1)
        assert r.success, r.message
    finally:
        stop.set()
        node.destroy_node()
        os.killpg(os.getpgid(launch.pid), signal.SIGINT)
        try:
            launch.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(launch.pid), signal.SIGKILL)
    text = log_path.read_text()
    assert "loop closures accepted" in text, text[-3000:]  # the summary of save_map
    off = "visual loop closure was disabled for the whole run: no vocabulary"
    if name == "handheld_camera_imu":
        assert f"visual vocabulary: {profile.parent / 'vocab.bin'} (" in text and "from the profile)" in text, text[-3000:]
    if name in ("ground_camera", "ground_lidar_camera_imu"):
        assert "vocabularies/tum_rgbd.bin (9964 words, from the profile)" in text, text[-3000:]
    if name == "ground_lidar":
        assert "words, from the launch)" in text, text[-3000:]
    if name == "aerial_lidar_camera_imu":
        assert off in text, text[-3000:]
    else:
        assert off not in text, text[-3000:]
