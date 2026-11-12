"""The live viewer node: frames, transforms and the events of the log panel (with a stand-in for Rerun)."""

import math
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import rclpy
import strataslam_ros.viewer as viewer_mod
from geometry_msgs.msg import Transform
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Log
from sensor_msgs.msg import Image
from strataslam_ros.viewer import (
    Viewer,
    depth_array,
    floor_of_frame,
    image_array,
    matrix,
)

from strataslam_msgs.msg import CameraView, FloorState


class FakeView:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.poses: list[tuple[float, tuple[float, float, float], int]] = []
        self.baros: list[tuple[float, float]] = []
        self.frames: list[tuple] = []
        self.clouds: dict[int, np.ndarray] = {}
        self.transitions: list[tuple] = []

    def pose(self, t, xyz, floor) -> None:
        self.poses.append((t, xyz, floor))

    def event(self, t, text, level="INFO") -> None:
        self.events.append((text, level))

    def floor_cloud(self, t, floor, points, colours=None) -> None:
        self.clouds[floor] = np.asarray(points)

    def sensor_cloud(self, t, sensor, points) -> None:
        pass

    def flush_paths(self) -> None:
        pass

    def transition(self, t, kind, from_floor, to_floor, dz) -> None:
        self.transitions.append((kind, from_floor, to_floor, dz))

    def camera(self, t, image, depth, uv, use) -> None:
        self.frames.append((t, image, depth, uv, use))

    def baro(self, t, z) -> None:
        self.baros.append((t, z))


@pytest.fixture
def viewer():
    rclpy.init()
    node = Viewer(FakeView())
    yield node
    node.destroy_node()
    rclpy.shutdown()


def odometry(t: float, z: float) -> Odometry:
    m = Odometry()
    m.header.stamp.sec = int(t)
    m.pose.pose.position.z = z
    return m


def state(floor: int, s: str = "on_floor", num_floors: int = 1) -> FloorState:
    m = FloorState()
    m.floor, m.state, m.num_floors = floor, s, num_floors
    return m


def test_frames_and_transforms() -> None:
    assert floor_of_frame("floor_3") == 3
    assert floor_of_frame("odom") is None and floor_of_frame("floor_x") is None
    tr = Transform()
    tr.translation.x, tr.translation.z = 1.0, 2.0
    tr.rotation.z, tr.rotation.w = math.sin(math.pi / 4), math.cos(math.pi / 4)  # 90 deg about z
    m = matrix(tr)
    np.testing.assert_allclose(m @ [1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 2.0, 1.0], atol=1e-12)


def tf(parent: str, child: str, yaw: float, xyz: tuple[float, float, float]):
    from geometry_msgs.msg import TransformStamped
    from tf2_msgs.msg import TFMessage

    ts = TransformStamped()
    ts.header.frame_id, ts.child_frame_id = parent, child
    ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = xyz
    ts.transform.rotation.z, ts.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return TFMessage(transforms=[ts])


def cloud(frame: str, points: list[tuple[float, float, float]]):
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header

    return point_cloud2.create_cloud_xyz32(Header(frame_id=frame), points)


def test_the_floors_keep_their_anchors_and_the_robot_takes_the_correction(viewer) -> None:
    # The Summit's elevator bag (ADR-0075): back on floor 0 the loop closures corrected the odometry by 11 deg and
    # 4.6 m. Floor 1 must stay where it was against floor 0, and the robot must stand on floor 0's optimized map.
    v = viewer
    v.on_floor_state(state(0))
    v.on_tf(tf("floor_0", "odom", 0.0, (0.0, 0.0, 0.0)))  # floor 0's anchor: odom is floor 0 at the start
    v.on_tf(tf("floor_1", "odom", math.pi, (-22.0, -6.0, -3.0)))  # floor 1 opened after the ride up
    v.on_cloud(cloud("floor_1", [(0.0, 0.0, 0.0)]))
    floor1_before = v.view.clouds[1].copy()
    v.on_floor_state(state(0))
    yaw = math.radians(11.0)
    v.on_tf(tf("floor_0", "odom", yaw, (-1.9, 4.2, 0.0)))  # the correction after the ride down
    v.on_cloud(cloud("floor_1", [(0.0, 0.0, 0.0)]))
    np.testing.assert_allclose(v.view.clouds[1], floor1_before, atol=1e-6)  # floor 1 has not turned with it
    v.on_cloud(cloud("floor_0", [(5.0, 0.0, 0.0)]))
    np.testing.assert_allclose(v.view.clouds[0], [[5.0, 0.0, 0.0]], atol=1e-6)  # nor floor 0: map is its anchor
    # The robot at odometry (5, 0, 0): through the correction it is at its optimized place on floor 0.
    m = Odometry()
    m.header.stamp.sec = 10
    m.pose.pose.position.x = 5.0
    v.on_odometry(m)
    expected = (5.0 * math.cos(yaw) - 1.9, 5.0 * math.sin(yaw) + 4.2, 0.0)
    np.testing.assert_allclose(v.view.poses[-1][1], expected, atol=1e-6)


def test_without_map_the_first_transform_of_each_floor_is_its_anchor(viewer) -> None:
    # output.publish_tf: false publishes no map: the viewer places floor 1 against floor 0 by their first transforms.
    v = viewer
    v.on_floor_state(state(1))
    v.on_tf(tf("floor_0", "strataslam_odom", 0.0, (0.0, 0.0, 0.0)))
    v.on_tf(tf("floor_1", "strataslam_odom", 0.0, (0.0, 0.0, -3.0)))
    v.on_tf(tf("floor_1", "strataslam_odom", 0.3, (1.0, 0.0, -3.0)))  # a correction on floor 1
    v.on_cloud(cloud("floor_1", [(0.0, 0.0, 0.0)]))
    np.testing.assert_allclose(v.view.clouds[1], [[0.0, 0.0, 3.0]], atol=1e-6)


def test_a_new_floor_continues_the_robot_where_the_floor_it_left_had_it(viewer) -> None:
    # The Summit's prueba_03: floor 0's loop closures had lifted the robot 1.4 m above its odometry when it took the
    # ramp down; the floor it reached is placed by that correction (as the node places map -> floor_<k>), so the robot
    # drawn through the new floor stands where it was drawn through the old one: no jump.
    v = viewer
    v.on_floor_state(state(0))
    v.on_tf(tf("floor_0", "strataslam_odom", 0.0, (0.0, 0.0, 0.0)))
    v.on_tf(tf("floor_0", "strataslam_odom", 0.0, (0.0, 0.0, 1.4)))  # the correction on floor 0
    m = Odometry()
    m.header.stamp.sec = 10
    m.pose.pose.position.x, m.pose.pose.position.z = 20.0, -1.3
    v.on_odometry(m)
    before = v.view.poses[-1][1]
    v.on_floor_state(state(1))  # the floor state comes before the new floor's transform
    v.on_tf(tf("floor_1", "strataslam_odom", 0.3, (-5.0, 2.0, 1.3)))  # floor 1's anchor
    v.on_odometry(m)
    np.testing.assert_allclose(v.view.poses[-1][1], before, atol=1e-9)
    np.testing.assert_allclose(before, (20.0, 0.0, 0.1), atol=1e-9)


def test_the_published_map_places_the_floors(viewer) -> None:
    v = viewer
    v.on_floor_state(state(1))
    v.on_tf(tf("floor_1", "odom", 0.0, (0.0, 0.0, -3.0)))
    v.on_tf(tf("map", "floor_1", 0.2, (1.0, 2.0, 3.0)))
    v.on_cloud(cloud("floor_1", [(0.0, 0.0, 0.0)]))
    np.testing.assert_allclose(v.view.clouds[1], [[1.0, 2.0, 3.0]], atol=1e-6)


def test_direction_new_floor_and_known_floor_are_reported(viewer) -> None:
    v = viewer
    v.on_odometry(odometry(1, 0.0))  # before any floor state: not drawn
    assert v.view.poses == []
    v.on_floor_state(state(0))
    v.on_odometry(odometry(2, 0.0))
    v.on_floor_state(state(-1, "in_transition"))
    v.on_odometry(odometry(3, -0.2))
    v.on_odometry(odometry(4, -0.5))  # 0.5 m below where the transition began
    v.on_floor_state(state(1, num_floors=2))
    v.on_floor_state(state(-1, "in_transition", 2))
    v.on_odometry(odometry(5, 0.4))
    v.on_floor_state(state(0, num_floors=2))
    texts = [t for t, _ in v.view.events]
    assert texts == ["starting on floor 0", "going down (-0.5 m so far)", "new floor 1", "going up (+0.9 m so far)",
                     "on floor 0 again: recognised as an already mapped floor (same level)"]


def test_lost_tracking_is_told_apart_from_a_transition(viewer) -> None:
    from strataslam_tools.live_view import LOST

    v = viewer
    v.on_floor_state(state(0))
    v.on_odometry(odometry(1, 0.0))
    v.on_floor_state(state(-1, "lost"))
    v.on_odometry(odometry(2, 0.0))
    v.on_floor_state(state(-1, "pending"))
    v.on_odometry(odometry(3, 0.0))
    v.on_floor_state(state(-1, "in_transition"))
    v.on_odometry(odometry(4, 0.0))
    assert [p[2] for p in v.view.poses] == [0, LOST, LOST, -1]


def test_the_node_log_is_shown_with_its_level(viewer) -> None:
    v = viewer
    warn = Log(name="strataslam_node", msg="tracking lost: new session (floor pending)", level=30)
    v.on_log(Log(name="other_node", msg="not ours", level=20))
    v.on_log(warn)  # before the first pose: kept until there is a data time
    assert v.view.events == []
    v.on_floor_state(state(0))
    v.on_odometry(odometry(1, 0.0))
    assert ("tracking lost: new session (floor pending)", "WARN") in v.view.events
    assert all(text != "not ours" for text, _ in v.view.events)


def test_the_transitions_the_node_closes_are_passed_to_the_view(viewer) -> None:
    v = viewer
    v.on_floor_state(state(0))
    v.on_odometry(odometry(1, 0.0))
    for text in ("floor transition ended: elevator 0->1 dz=14.770732",
                 "floor transition ended: unknown 0->0 dz=0.000000",
                 "floor transition ended: stairs 1->0 dz=-15.663437", "loop closure accepted"):
        v.on_log(Log(name="strataslam_node", msg=text, level=Log.INFO))
    assert v.view.transitions == [("elevator", 0, 1, 14.770732), ("unknown", 0, 0, 0.0), ("stairs", 1, 0, -15.663437)]


def test_images_are_read_with_their_row_padding() -> None:
    img = Image(width=2, height=2, encoding="rgb8", step=8)  # 2 bytes of padding per row
    img.data = bytes([1, 2, 3, 4, 5, 6, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0])
    np.testing.assert_array_equal(image_array(img)[1, 1], [10, 11, 12])
    assert image_array(Image()) is None  # feature-frame input: no image
    depth = Image(width=2, height=1, encoding="32FC1", step=8)
    depth.data = np.array([1.5, 0.0], dtype=np.float32).tobytes()
    np.testing.assert_array_equal(depth_array(depth), [[1.5, 0.0]])


def test_camera_frames_and_the_barometer_reach_the_view(viewer) -> None:
    v = viewer
    v.on_log(Log(name="strataslam_node", msg="pipeline started", level=20))
    cam = CameraView(u=[10.0, 20.0], v=[5.0, 6.0], use=[CameraView.INLIER, CameraView.UNMATCHED])
    cam.header.stamp.sec = 1
    v.on_camera(cam)  # before any pose or floor: drawn, and it gives the log its time
    assert ("pipeline started", "INFO") in v.view.events
    baro = state(0)
    baro.has_baro, baro.baro_height = True, 0.3
    v.on_floor_state(baro)  # no robot height yet to align it with
    assert v.view.baros == []
    v.on_odometry(odometry(1, 2.0))
    _, image, depth, uv, use = v.view.frames[0]
    assert image is None and depth is None
    np.testing.assert_array_equal(uv, [[10.0, 5.0], [20.0, 6.0]])
    assert list(use) == [CameraView.INLIER, CameraView.UNMATCHED]
    v.on_floor_state(baro)  # aligned with the robot at its first reading
    baro.baro_height = 3.3
    v.on_floor_state(baro)
    assert [z for _, z in v.view.baros] == pytest.approx([2.0, 5.0])
    v.on_floor_state(state(0))  # without a barometer nothing is drawn
    assert len(v.view.baros) == 2


def test_children_and_exits_of_the_window() -> None:
    assert not viewer_mod.exited(None)  # no window of ours (an open one was reused): never seen closing
    before = viewer_mod.children()
    child = subprocess.Popen(["sleep", "0.3"])  # stands in for the Rerun window, a child of the viewer
    assert child.pid in viewer_mod.children() - before
    assert not viewer_mod.exited(child.pid)
    time.sleep(0.6)
    assert viewer_mod.exited(child.pid)


# ---- the viewer's life with its window (Rerun stand-in), the SLAM (a node of that name) and the dialog ----

class WindowView(FakeView):
    """LiveView stand-in whose window is a child process, as Rerun's is."""

    windows: ClassVar[list[subprocess.Popen]] = []
    reopened = 0
    seconds: ClassVar[list[float]] = [30.0]

    def __init__(self) -> None:
        super().__init__()
        self._open()

    def _open(self) -> None:
        WindowView.windows.append(subprocess.Popen(["sleep", str(WindowView.seconds.pop(0) if WindowView.seconds else 30.0)]))

    def reopen(self) -> None:
        WindowView.reopened += 1
        self._open()


@pytest.fixture
def life(monkeypatch, tmp_path):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ROS_DOMAIN_ID", str(10 + (os.getpid() + 71) % 80))
    monkeypatch.setattr(viewer_mod, "LiveView", WindowView)
    WindowView.windows, WindowView.reopened = [], 0
    slams: list[subprocess.Popen] = []

    def slam(discarded: Path | None = None) -> subprocess.Popen:
        """A node named as the SLAM's; with `discarded`, it answers ~/discard_map by writing that file."""
        code = "import rclpy; rclpy.init(); n = rclpy.create_node('strataslam_node')\n"
        if discarded is not None:
            code += ("from std_srvs.srv import Trigger\n"
                     "def on_discard(req, res):\n"
                     f"    open({str(discarded)!r}, 'w').write('discarded')\n"
                     "    res.success = True\n"
                     "    return res\n"
                     "n.create_service(Trigger, '~/discard_map', on_discard)\n")
        code += "rclpy.spin(n)\n"
        p = subprocess.Popen([sys.executable, "-c", code], env=os.environ.copy())
        slams.append(p)
        return p

    def answers(*codes: int) -> Path:
        """The dialog answers these, one per question; each question leaves a line in the returned file."""
        log = tmp_path / "asked"
        monkeypatch.setattr(viewer_mod, "DIALOG", f"import sys\nlog = open({str(log)!r}, 'a')\nlog.write('q\\n')\nlog.close()\n"
                                                 f"n = sum(1 for _ in open({str(log)!r}))\ncodes = {list(codes)!r}\n"
                                                 "sys.exit(codes[min(n, len(codes)) - 1])\n")
        return log

    def run(after: float | None = None) -> int:
        """The viewer's main(); `after` seconds: a Ctrl+C."""
        if after is not None:
            threading.Timer(after, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
        with pytest.raises(SystemExit) as e:
            viewer_mod.main([])
        return int(e.value.code or 0)

    yield types.SimpleNamespace(slam=slam, answers=answers, run=run)
    for p in slams + WindowView.windows:
        if p.poll() is None:
            p.kill()
            p.wait()


def test_ctrl_c_closes_the_window(life) -> None:  # type: ignore[no-untyped-def]
    life.slam()
    asked = life.answers(viewer_mod.STOP_SLAM)
    life.run(after=4.0)
    assert WindowView.windows[0].wait(timeout=5) == -signal.SIGTERM  # closed by the viewer, not left behind
    assert not asked.exists()


def test_closing_the_window_while_the_slam_runs_asks_first(life, tmp_path) -> None:  # type: ignore[no-untyped-def]
    discarded = tmp_path / "discarded"
    life.slam(discarded)
    WindowView.seconds = [4.0, 1.0]  # the user closes the window, reopens it, and closes it again
    asked = life.answers(viewer_mod.REOPEN, viewer_mod.STOP_SLAM)
    assert life.run() == 0  # the launch ends: the SLAM stops and saves its map
    assert WindowView.reopened == 1
    assert asked.read_text().count("q") == 2
    assert not discarded.exists()


def test_stopping_without_saving_tells_the_slam_to_discard_its_map(life, tmp_path) -> None:  # type: ignore[no-untyped-def]
    discarded = tmp_path / "discarded"
    life.slam(discarded)
    WindowView.seconds = [4.0]
    life.answers(viewer_mod.DISCARD)
    assert life.run() == 0  # the launch ends
    assert discarded.exists()  # and the SLAM was told before it stops


def test_the_question_is_a_valid_program_with_the_three_answers() -> None:
    compile(viewer_mod.DIALOG, "dialog", "exec")
    for code in (viewer_mod.REOPEN, viewer_mod.STOP_SLAM, viewer_mod.DISCARD):
        assert f", {code})" in viewer_mod.DIALOG
    assert "11)" not in viewer_mod.DIALOG  # "close only the viewer" is gone: launch without the viewer instead


def test_after_the_slam_the_window_stays_until_it_is_closed(life) -> None:  # type: ignore[no-untyped-def]
    slam = life.slam()
    WindowView.seconds = [8.0]
    asked = life.answers(viewer_mod.STOP_SLAM)
    threading.Timer(4.0, slam.terminate).start()  # the bag is processed: the SLAM node ends
    t0 = time.monotonic()
    assert life.run() == 0  # then closing the window ends the launch, without a question
    assert time.monotonic() - t0 > 7.0  # the viewer waited for the window, not for the SLAM
    assert not asked.exists()


def test_without_a_slam_closing_the_window_just_ends(life) -> None:  # type: ignore[no-untyped-def]
    WindowView.seconds = [2.0]
    asked = life.answers(viewer_mod.REOPEN)
    assert life.run() == 0
    assert not asked.exists()


def test_the_launch_waits_for_the_viewer_and_ends_with_it() -> None:
    import importlib.util

    from ament_index_python.packages import get_package_share_directory

    path = Path(get_package_share_directory("strataslam_ros")) / "launch" / "strataslam.launch.py"
    spec = importlib.util.spec_from_file_location("strataslam_launch", path)
    assert spec and spec.loader
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    assert launch.VIEWER_ONLY == viewer_mod.VIEWER_ONLY
    assert launch.ends_on_slam_exit(viewer_running=False)  # no window: the launch ends with the SLAM, as before
    assert not launch.ends_on_slam_exit(viewer_running=True)  # the window stays, to look at the result
    assert launch.ends_on_viewer_exit(0)  # closed after the SLAM, or "stop the SLAM": everything ends
    assert not launch.ends_on_viewer_exit(viewer_mod.VIEWER_ONLY)  # only the viewer was closed: the SLAM goes on
