"""The camera takes colour images in rgb8, bgr8, rgba8 and bgra8 (a ZED publishes bgra8), with padded rows.

Each image is two flat colours, left and right; the live camera view (strataslam/camera_view, rgb8) must show the
same two colours whatever the encoding. Until 2026-11-16 the node took mono8, rgb8 and bgr8 only and read the rows
packed: a ZED's frames were dropped with "unsupported color encoding".
"""

import os
import struct
import subprocess
import time
from pathlib import Path

import pytest
import rclpy
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

from strataslam_msgs.msg import CameraView

PROFILE = """\
strataslam:
  version: 1
robot:
  platform: handheld
  base_frame: base_link
sensors:
  camera:
    enabled: true
    rgb_topic: /rgb
    depth_topic: /depth
    info_topic: /info
    depth_min: 0.3
    depth_max: 6.0
multifloor:
  mode: single
fusion:
  colorize_cloud: true
output:
  map_dir: {map_dir}
runtime:
  threads: 1
"""

W, H, PAD = 320, 240, 8  # PAD bytes at the end of every row
LEFT, RIGHT = (200, 30, 10), (10, 60, 220)  # r, g, b
ORDER = {"rgb8": "rgb", "bgr8": "bgr", "rgba8": "rgba", "bgra8": "bgra"}


def color_image(stamp, encoding: str) -> Image:  # type: ignore[no-untyped-def]
    order = ORDER[encoding]
    px = {
        side: bytes({"r": c[0], "g": c[1], "b": c[2], "a": 255}[k] for k in order)
        for side, c in (("l", LEFT), ("r", RIGHT))
    }
    row = px["l"] * (W // 2) + px["r"] * (W // 2) + b"\x00" * PAD
    m = Image(height=H, width=W, encoding=encoding, step=W * len(order) + PAD, data=row * H)
    m.header.stamp, m.header.frame_id = stamp, "camera"
    return m


def depth_image(stamp) -> Image:  # type: ignore[no-untyped-def]
    m = Image(
        height=H, width=W, encoding="32FC1", step=W * 4, data=struct.pack("<f", 2.0) * (W * H)
    )
    m.header.stamp, m.header.frame_id = stamp, "camera"
    return m


def info(stamp) -> CameraInfo:  # type: ignore[no-untyped-def]
    m = CameraInfo(height=H, width=W)
    m.k = [250.0, 0.0, W / 2, 0.0, 250.0, H / 2, 0.0, 0.0, 1.0]
    m.header.stamp, m.header.frame_id = stamp, "camera"
    return m


def pixel(img: Image, row: int, col: int) -> tuple[int, ...]:
    i = row * img.step + col * 3
    return tuple(img.data[i : i + 3])


@pytest.fixture()
def ros_domain(monkeypatch):  # type: ignore[no-untyped-def]
    """A ROS domain of its own, so a parallel test or a running robot cannot talk to this node."""
    monkeypatch.setenv("ROS_DOMAIN_ID", str(10 + os.getpid() % 80))
    rclpy.init()
    yield
    rclpy.shutdown()


def test_every_colour_encoding_shows_its_colours_in_the_camera_view(
    ros_domain, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    profile = tmp_path / "profile.yaml"
    profile.write_text(PROFILE.format(map_dir=tmp_path / "map"))
    log_path = tmp_path / "node.log"
    with open(log_path, "w") as log:
        exe = (
            Path(get_package_prefix("strataslam_ros"))
            / "lib"
            / "strataslam_ros"
            / "strataslam_node_exe"
        )
        proc = subprocess.Popen(
            [str(exe), "--ros-args", "-p", f"profile:={profile}"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, RCUTILS_COLORIZED_OUTPUT="0"),
        )
    node = Node("strataslam_color_probe")
    views: list[CameraView] = []
    node.create_subscription(
        CameraView, "/strataslam/camera_view", views.append, qos_profile_sensor_data
    )
    static = node.create_publisher(
        TFMessage, "/tf_static", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    )
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = "base_link", "camera"
    tf.header.stamp = node.get_clock().now().to_msg()
    (
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z,
        tf.transform.rotation.w,
    ) = -0.5, 0.5, -0.5, 0.5
    static.publish(TFMessage(transforms=[tf]))
    pubs = {
        t: node.create_publisher(m, t, qos_profile_sensor_data)
        for t, m in (("/rgb", Image), ("/depth", Image), ("/info", CameraInfo))
    }
    t = [1000.0]

    def send(encoding: str) -> CameraView | None:
        """One frame in `encoding`; the view the node publishes for it (it publishes one every 200 ms at most)."""
        views.clear()
        end = time.monotonic() + 10.0
        while time.monotonic() < end:
            t[0] += 0.1
            s = node.get_clock().now().to_msg()
            s.sec, s.nanosec = int(t[0]), round((t[0] % 1) * 1e9) % 1_000_000_000
            pubs["/depth"].publish(depth_image(s))
            pubs["/info"].publish(info(s))
            pubs["/rgb"].publish(color_image(s, encoding))
            stop = time.monotonic() + 0.3
            while time.monotonic() < stop:
                rclpy.spin_once(node, timeout_sec=0.02)
            if views and views[-1].image.data:
                return views[-1]
        return None

    try:
        end = time.monotonic() + 30.0
        while pubs["/rgb"].get_subscription_count() == 0 and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert pubs["/rgb"].get_subscription_count() > 0, "the node never subscribed to /rgb"
        for encoding in ORDER:
            v = send(encoding)
            assert v is not None, (
                f"{encoding}: no camera view with an image; log:\n{log_path.read_text()[-3000:]}"
            )
            img = v.image
            assert img.encoding == "rgb8"
            assert pixel(img, img.height // 2, 5) == LEFT, encoding
            assert pixel(img, img.height // 2, img.width - 5) == RIGHT, encoding
        assert "unsupported color" not in log_path.read_text()
    finally:
        proc.send_signal(2)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        node.destroy_node()
