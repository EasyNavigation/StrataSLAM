"""The wheel odometry from TF when it moves a frame above the profile's base: odom -> base_footprint, with
base_footprint -> base_link fixed (the Kobuki's URDF), and `robot.base_frame: base_link`.

The node took only a transform odom -> base_frame itself, so the odometry of such a robot never reached it, silently:
every tracking loss of the camera opened a session and, with no relocalization, a floor (three floors in one storey of
the simulated Kobuki). Now the moving transform is composed with the fixed chain below it, and a profile whose base
the odometry does not reach is told after 2 s of frames.
"""

import os
import subprocess
from pathlib import Path

import rosbag2_py
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import deserialize_message, serialize_message
from ros_helpers import PROFILES, export_bag
from tf2_msgs.msg import TFMessage

SECONDS = 15.0


def profile(tmp: Path, name: str, base: str, map_dir: Path) -> Path:
    """ground_camera.yaml with this base frame, its own map folder and one thread (a run that repeats exactly)."""
    text = (PROFILES / "ground_camera.yaml").read_text()
    text = text.replace("  base_frame: base_link\n", f"  base_frame: {base}\n", 1)
    text = text.replace("output:\n", f"output:\n  map_dir: {map_dir}\n", 1)
    text = text.replace("runtime:\n", "runtime:\n  threads: 1\n", 1)
    assert f"base_frame: {base}" in text and str(map_dir) in text and "threads: 1" in text
    path = tmp / f"{name}.yaml"
    path.write_text(text)
    return path


def with_fixed_child(src: Path, dst: Path, parent: str, child: str) -> Path:
    """Copy of the bag `src` with a static transform parent -> child (identity) added at its start."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(src)), rosbag2_py.ConverterOptions("", ""))
    storage = reader.get_metadata().storage_identifier
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(dst), storage_id=storage), rosbag2_py.ConverterOptions("", ""))
    topics = reader.get_all_topics_and_types()
    for t in topics:
        writer.create_topic(t)
    if not any(t.name == "/tf_static" for t in topics):
        writer.create_topic(rosbag2_py.TopicMetadata(name="/tf_static", type="tf2_msgs/msg/TFMessage",
                                                     serialization_format="cdr"))
    added = False
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if not added:
            tr = TransformStamped()
            tr.header.stamp.sec, tr.header.stamp.nanosec = divmod(t_ns, 1_000_000_000)
            tr.header.frame_id, tr.child_frame_id = parent, child
            tr.transform.rotation.w = 1.0
            writer.write("/tf_static", serialize_message(TFMessage(transforms=[tr])), t_ns)
            added = True
        writer.write(topic, data, t_ns)
    del writer
    return dst


def odometry_children(bag: Path) -> set[tuple[str, str]]:
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag)), rosbag2_py.ConverterOptions("", ""))
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/tf"]))
    pairs = set()
    while reader.has_next():
        _, data, _ = reader.read_next()
        pairs |= {(t.header.frame_id, t.child_frame_id) for t in deserialize_message(data, TFMessage).transforms}
    return pairs


def run(profile_path: Path, bag: Path, log_path: Path) -> str:
    env = {**os.environ, "RCUTILS_COLORIZED_OUTPUT": "0", "ROS_DOMAIN_ID": str(10 + (os.getpid() + 37) % 80)}
    with open(log_path, "w") as log:
        subprocess.run(["ros2", "launch", "strataslam_ros", "strataslam.launch.py", f"profile:={profile_path}",
                        f"bag:={bag}", "features_topic:=/strataslam/features"], stdout=log, stderr=subprocess.STDOUT, env=env, timeout=600, check=False)
    return log_path.read_text()


def keyframes(map_dir: Path) -> list[str]:
    return sorted((p.relative_to(map_dir).as_posix(), p.read_text()) for p in map_dir.glob("floor_*/keyframes.csv"))


def test_the_odometry_of_a_frame_above_the_base_reaches_the_tracking(tmp_path: Path) -> None:
    sim = tmp_path / "sim"
    export_bag(sim, profile(tmp_path, "export", "base_footprint", tmp_path / "unused"), seconds=SECONDS)
    assert ("odom", "base_footprint") in odometry_children(sim / "bag")
    chained = with_fixed_child(sim / "bag", tmp_path / "chained", "base_footprint", "base_link")

    direct_map, chain_map = tmp_path / "direct", tmp_path / "chain"
    run(profile(tmp_path, "direct", "base_footprint", direct_map), sim / "bag", tmp_path / "direct.log")
    log = run(profile(tmp_path, "chain", "base_link", chain_map), chained, tmp_path / "chain.log")

    assert "wheel odometry: TF odom -> base_footprint composed with the fixed base_footprint -> base_link" in log, log[-3000:]
    assert "external_odometry: no TF" not in log, log[-3000:]
    # base_link is base_footprint (identity): the same odometry, the same map.
    direct, chain = keyframes(direct_map), keyframes(chain_map)
    assert direct and direct == chain


def test_an_odometry_that_does_not_reach_the_base_is_told(tmp_path: Path) -> None:
    sim = tmp_path / "sim"
    export_bag(sim, profile(tmp_path, "export", "base_footprint", tmp_path / "unused"), seconds=8.0)
    # The profile names an odometry frame the robot does not publish: the camera tracks on its own, and says so.
    path = profile(tmp_path, "other", "base_footprint", tmp_path / "map")
    path.write_text(path.read_text().replace("    frame: odom", "    frame: wheel_odom", 1))
    assert "frame: wheel_odom" in path.read_text()
    log = run(path, sim / "bag", tmp_path / "other.log")
    assert "external_odometry: no TF wheel_odom -> base_footprint after 2 s of camera frames" in log, log[-3000:]
    assert "TF has wheel_odom -> {nothing}" in log, log[-3000:]
