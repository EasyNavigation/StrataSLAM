"""The TF tree the node leaves: whole without a robot odometry, a correction on top of it when the robot has one.

Audit AUD-044: the node only published floor_<k> -> odom. Without wheel odometry the robot was not in the tree; with it
(Summit, Tiago) `floor_k -> odom -> base` composed the SLAM correction with the wheels' odometry, a different `odom`.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest
import rclpy
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from test_malformed_clouds import PROFILE, ROOM, ROOM_N, cloud
from tf2_msgs.msg import TFMessage


@pytest.fixture()
def ros_domain(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ROS_DOMAIN_ID", str(10 + (os.getpid() + 53) % 80))
    rclpy.init()
    yield
    rclpy.shutdown()


def run_node(tmp_path: Path, wheel_odometry: bool, localizer: bool = False, publish_tf: bool = True) -> set[tuple[str, str]]:
    """Runs the node on 30 identical scans; returns the (parent, child) pairs the node published on /tf. `localizer`:
    another node also publishes map -> odom, as AMCL does."""
    profile = tmp_path / "profile.yaml"
    text = PROFILE.format(map_dir=tmp_path / "map")
    if not publish_tf:
        text = text.replace("output:\n", "output:\n  publish_tf: false\n", 1)
    profile.write_text(text)
    exe = Path(get_package_prefix("strataslam_ros")) / "lib" / "strataslam_ros" / "strataslam_node_exe"
    node = Node("strataslam_tf_probe")
    pub = node.create_publisher(PointCloud2, "/points", qos_profile_sensor_data)
    tf_pub = node.create_publisher(TFMessage, "/tf", 100)
    seen: set[tuple[str, str]] = set()
    node.create_subscription(TFMessage, "/tf", lambda m: seen.update((t.header.frame_id, t.child_frame_id) for t in m.transforms), 100)
    with open(tmp_path / "node.log", "w") as log:  # the child keeps its own descriptor
        proc = subprocess.Popen([str(exe), "--ros-args", "-p", f"profile:={profile}"], stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, RCUTILS_COLORIZED_OUTPUT="0"))
    try:
        end = time.monotonic() + 30.0
        while pub.get_subscription_count() == 0 and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        t = 1000.0
        for i in range(30):
            t += 0.1
            stamp = node.get_clock().now().to_msg()
            stamp.sec, stamp.nanosec = int(t), int((t % 1) * 1e9)
            if wheel_odometry:  # the robot's wheels: odom -> base_link, published before every scan
                w = TransformStamped()
                w.header.stamp = stamp
                w.header.frame_id, w.child_frame_id = "odom", "base_link"
                w.transform.rotation.w = 1.0
                transforms = [w]
                if localizer:  # map -> odom, as a localizer such as AMCL publishes it
                    m = TransformStamped()
                    m.header.stamp = stamp
                    m.header.frame_id, m.child_frame_id = "map", "odom"
                    m.transform.rotation.w = 1.0
                    transforms.append(m)
                tf_pub.publish(TFMessage(transforms=transforms))
            pub.publish(cloud(stamp, ROOM_N, 16, ROOM))
            stop = time.monotonic() + 0.15
            while time.monotonic() < stop:
                rclpy.spin_once(node, timeout_sec=0.02)
        stop = time.monotonic() + 1.0
        while time.monotonic() < stop:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        proc.send_signal(2)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        node.destroy_node()
    if wheel_odometry:
        seen.discard(("odom", "base_link"))  # ours; whether the node also sent it is checked from its log below
    if localizer:
        seen.discard(("map", "odom"))
    return seen


def test_without_a_robot_odometry_the_node_completes_the_tree(ros_domain, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    seen = run_node(tmp_path, wheel_odometry=False)
    assert ("floor_0", "odom") in seen, seen
    assert ("odom", "base_link") in seen, seen
    assert ("map", "floor_0") in seen, seen  # map -> floor_<k> -> odom -> base (REP-105)
    text = (tmp_path / "node.log").read_text()
    assert "publishing it (and floor_<k> -> odom) from the SLAM odometry" in text
    # Ctrl+C on a live run saves the map, as with a bag (audit AUD-092).
    assert "not saved: saving the map" in text, text[-2000:]
    assert (tmp_path / "map" / "manifest.yaml").exists()


def test_with_a_robot_odometry_the_node_publishes_the_correction_on_top_of_it(ros_domain, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    seen = run_node(tmp_path, wheel_odometry=True)
    text = (tmp_path / "node.log").read_text()
    assert "comes from the robot" in text, text[-2000:]
    assert ("floor_0", "odom") in seen, seen
    assert ("floor_0", "strataslam_odom") in seen, seen
    assert ("map", "floor_0") in seen, seen
    assert "also has the parent" not in text


def test_a_localizer_that_also_publishes_map_to_odom_is_reported(ros_domain, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    # AMCL's map -> odom and the SLAM's floor_0 -> odom give odom two parents: TF keeps one and the floors hang loose
    # (seen with EasyNav in the Summit simulation). The node says so, and what to set.
    run_node(tmp_path, wheel_odometry=True, localizer=True)
    text = (tmp_path / "node.log").read_text()
    assert "TF: odom also has the parent 'map'" in text, text[-2000:]
    assert "output.publish_tf: false" in text


def test_without_publish_tf_the_node_leaves_map_and_odom_alone(ros_domain, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    seen = run_node(tmp_path, wheel_odometry=True, localizer=True, publish_tf=False)
    text = (tmp_path / "node.log").read_text()
    assert "output.publish_tf is false" in text, text[-2000:]
    assert ("floor_0", "strataslam_odom") in seen, seen
    assert not {p for p in seen if p[1] in ("odom", "base_link", "floor_0")}, seen  # nothing on the robot's frames or below map
    assert "also has the parent" not in text
