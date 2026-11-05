"""A PointCloud2 whose layout does not match its buffer is dropped with a warning; the node keeps running.

Regression test for a crash found by the 2026-11-04 audit: a cloud with `data` shorter than
`width x height x point_step` made the node read past the buffer and die with SIGSEGV, and a bag with a
corrupt message took the same path.
"""

import math
import os
import struct
import subprocess
import time
from pathlib import Path

import pytest
import rclpy
from ament_index_python.packages import get_package_prefix
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField

PROFILE = """\
strataslam:
  version: 1
robot:
  platform: ground
  base_frame: base_link
sensors:
  lidar:
    enabled: true
    topic: /points
    type: 3d
    min_range: 0.2
    max_range: 50.0
    deskew: false
multifloor:
  mode: single
fusion:
  colorize_cloud: false
output:
  map_dir: {map_dir}
runtime:
  threads: 1
"""


def room_points() -> bytes:
    """A static 8 x 6 x 3 m room (floor and four walls), the same in every scan: the odometry stays put."""
    pts = []
    for i in range(40):
        for j in range(30):
            x, y = -4.0 + 8.0 * i / 39, -3.0 + 6.0 * j / 29
            pts.append((x, y, -0.5))
    for k in range(15):
        z = -0.5 + 3.0 * k / 14
        for i in range(40):
            x = -4.0 + 8.0 * i / 39
            pts += [(x, -3.0, z), (x, 3.0, z)]
        for j in range(30):
            y = -3.0 + 6.0 * j / 29
            pts += [(-4.0, y, z), (4.0, y, z)]
    # A pillar breaks the symmetry of the room.
    for k in range(15):
        for a in range(12):
            ang = 2 * math.pi * a / 12
            pts.append((1.5 + 0.2 * math.cos(ang), 1.0 + 0.2 * math.sin(ang), -0.5 + 3.0 * k / 14))
    return b"".join(struct.pack("<ffff", x, y, z, 0.0) for x, y, z in pts)


ROOM = room_points()
ROOM_N = len(ROOM) // 16


def cloud(stamp, width: int, point_step: int, data: bytes, fields=None, height: int = 1, row_step: int | None = None) -> PointCloud2:  # type: ignore[no-untyped-def]
    m = PointCloud2()
    m.header.frame_id = "base_link"
    m.header.stamp = stamp
    m.height = height
    m.width = width
    m.fields = fields if fields is not None else [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    m.point_step = point_step
    m.row_step = row_step if row_step is not None else point_step * width
    m.is_dense = True
    m.data = data
    return m


MALFORMED = {
    "data shorter than width x point_step": lambda s: cloud(s, 200000, 16, ROOM[:64]),
    "point_step 0": lambda s: cloud(s, 1000, 0, ROOM[:12]),
    "field offset beyond point_step": lambda s: cloud(
        s, 10, 4, ROOM[:40], fields=[PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                                     PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                                     PointField(name="z", offset=1000, datatype=PointField.FLOAT32, count=1)]),
    "row_step too small for the rows": lambda s: cloud(s, ROOM_N // 2, 16, ROOM, height=2, row_step=16),
}


@pytest.fixture()
def ros_domain(monkeypatch):  # type: ignore[no-untyped-def]
    """A ROS domain of its own, so a parallel test or a running robot cannot talk to this node."""
    monkeypatch.setenv("ROS_DOMAIN_ID", str(10 + os.getpid() % 80))
    rclpy.init()
    yield
    rclpy.shutdown()


def test_malformed_clouds_are_dropped_and_the_node_keeps_running(ros_domain, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    profile = tmp_path / "profile.yaml"
    profile.write_text(PROFILE.format(map_dir=tmp_path / "map"))
    log_path = tmp_path / "node.log"
    with open(log_path, "w") as log:
        # The executable itself, not `ros2 run`, which does not pass the final SIGINT on to the node.
        exe = Path(get_package_prefix("strataslam_ros")) / "lib" / "strataslam_ros" / "strataslam_node_exe"
        proc = subprocess.Popen([str(exe), "--ros-args", "-p", f"profile:={profile}"],
                                stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, RCUTILS_COLORIZED_OUTPUT="0"))
    node = Node("strataslam_malformed_probe")
    odom: list[float] = []
    node.create_subscription(Odometry, "/strataslam/odometry", lambda m: odom.append(time.monotonic()), 100)
    pub = node.create_publisher(PointCloud2, "/points", qos_profile_sensor_data)
    t = [1000.0]

    def stamp():  # type: ignore[no-untyped-def]
        s = node.get_clock().now().to_msg()
        s.sec, s.nanosec = int(t[0]), int((t[0] % 1) * 1e9)
        return s

    def send_valid(n: int) -> None:
        for _ in range(n):
            t[0] += 0.1
            pub.publish(cloud(stamp(), ROOM_N, 16, ROOM))
            end = time.monotonic() + 0.15
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.02)

    try:
        end = time.monotonic() + 30.0
        while pub.get_subscription_count() == 0 and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert pub.get_subscription_count() > 0, "the node never subscribed to /points"
        send_valid(20)
        assert odom, "no odometry from valid clouds: the pipeline did not start, so the malformed clouds would not be read"
        for make in MALFORMED.values():
            t[0] += 0.1
            pub.publish(make(stamp()))
            send_valid(3)
            assert proc.poll() is None, f"node died (rc {proc.returncode}); log:\n{log_path.read_text()[-3000:]}"
        before = len(odom)
        send_valid(10)
        assert len(odom) > before, "odometry stopped after the malformed clouds"
        text = log_path.read_text()
        assert "dropping malformed PointCloud2" in text, text[-3000:]
    finally:
        proc.send_signal(2)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        node.destroy_node()
