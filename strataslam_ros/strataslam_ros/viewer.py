"""ROS 2 node behind `viewer:=true`: draws a StrataSLAM run live in Rerun (strataslam_tools.live_view).

Everything is drawn in the odometry frame, the one frame that is continuous across floors: the robot
from /strataslam/odometry, each floor's cloud (/strataslam/floor_cloud, in `floor_<k>`) moved with the
latest TF floor_<k> -> odom, and the floor of every pose from /strataslam/floor_state. The placement
of the floors is the live estimate; the fused building comes from `strataslam fuse`.

What the sensors see right now (/strataslam/scan, /strataslam/depth_cloud) is drawn in magenta under
`sensor`, with an eye in the entity tree to show or hide it. With a camera, /strataslam/camera_view adds
its image with the keypoints coloured by their use and the depth image used; with a barometer, its
height (in /strataslam/floor_state) is a magenta line in the level line, aligned with the robot's height
at its first reading.

The log panel shows the node's own log (/rosout: transitions with their type and height, loop closures,
relocalizations, losses) and two things read from the floor state: whether a transition goes up or down,
and whether the floor it ends on is new or one already mapped (recognised as the same level).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Log
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from strataslam_tools.live_view import LiveView
from tf2_msgs.msg import TFMessage

from strataslam_msgs.msg import CameraView, FloorState

NODE_NAME = "strataslam_node"
DIRECTION_M = 0.3  # height change after which a transition is reported as going up or down
LEVELS = {10: "DEBUG", 20: "INFO", 30: "WARN", 40: "ERROR", 50: "CRITICAL"}


def stamp_s(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def matrix(transform) -> np.ndarray:
    """4x4 matrix of a geometry_msgs/Transform."""
    q, t = transform.rotation, transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = [t.x, t.y, t.z]
    return m


def image_array(img) -> np.ndarray | None:
    """H x W x 3 (rgb8) or H x W (mono8) array of a sensor_msgs/Image; None if empty or another encoding."""
    channels = {"rgb8": 3, "mono8": 1}.get(img.encoding)
    if channels is None or img.width == 0 or img.height == 0:
        return None
    rows = np.frombuffer(bytes(img.data), dtype=np.uint8).reshape(img.height, img.step)
    a = rows[:, :img.width * channels]
    return a.reshape(img.height, img.width, 3) if channels == 3 else a


def depth_array(img) -> np.ndarray | None:
    """H x W metres of a 32FC1 sensor_msgs/Image; None if empty or another encoding."""
    if img.encoding != "32FC1" or img.width == 0 or img.height == 0:
        return None
    rows = np.frombuffer(bytes(img.data), dtype=np.uint8).reshape(img.height, img.step)
    return rows[:, :img.width * 4].copy().view(np.float32)


def floor_of_frame(frame: str) -> int | None:
    if frame.startswith("floor_") and frame[6:].isdigit():
        return int(frame[6:])
    return None


class Viewer(Node):
    def __init__(self, view: LiveView) -> None:
        super().__init__("strataslam_viewer")
        self.view = view
        self.floor: int | None = None  # unknown until the first /strataslam/floor_state
        self.state = ""
        self.t: float | None = None  # stamp of the latest pose or camera frame (clouds carry the wall clock)
        self.floor_from_odom: dict[int, np.ndarray] = {}  # T_floor_odom, latest per floor
        self.pending_clouds: dict[int, PointCloud2] = {}  # clouds that arrived before their floor's TF
        self.z: float | None = None
        self.transit_z0: float | None = None  # height when the current transition began
        self.direction_told = False
        self.floors_seen: set[int] = set()
        self.early_events: list[tuple[str, str]] = []  # log lines before the first data (no data time yet)
        self.baro_offset: float | None = None  # robot height minus barometric height at the first reading
        reliable = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, "strataslam/odometry", self.on_odometry, reliable)
        self.create_subscription(FloorState, "strataslam/floor_state", self.on_floor_state, reliable)
        self.create_subscription(PointCloud2, "strataslam/floor_cloud", self.on_cloud, latched)
        # What the sensors see right now, already in the odometry frame.
        self.create_subscription(PointCloud2, "strataslam/scan", lambda m: self.on_sensor("lidar", m), qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "strataslam/depth_cloud", lambda m: self.on_sensor("camera", m),
                                 qos_profile_sensor_data)
        self.create_subscription(CameraView, "strataslam/camera_view", self.on_camera, qos_profile_sensor_data)
        self.create_subscription(TFMessage, "/tf", self.on_tf, reliable)
        rosout = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Log, "/rosout", self.on_log, rosout)
        # The last stretch of path is sent even when poses stop arriving (end of a bag, robot stopped).
        self.create_timer(0.5, self.view.flush_paths)

    def slam_running(self) -> bool:
        """Whether the SLAM node of this launch (same namespace) is in the ROS graph."""
        ns = self.get_namespace()
        return any(n == NODE_NAME and space == ns for n, space in self.get_node_names_and_namespaces())

    def event(self, text: str, level: str = "INFO") -> None:
        if self.t is None:  # before the first data there is no time to put it at
            self.early_events.append((text, level))
        else:
            self.view.event(self.t, text, level)

    def at(self, t: float) -> None:
        """Data time of a pose or a camera frame; the log lines kept until the first one are sent at it (when the
        tracking never starts, the camera frames still give the log its time)."""
        self.t = t
        for text, level in self.early_events:
            self.view.event(t, text, level)
        self.early_events.clear()

    def on_odometry(self, m: Odometry) -> None:
        t = stamp_s(m.header.stamp)
        self.at(t)
        if self.floor is None:
            return
        p = m.pose.pose.position
        self.view.pose(t, (p.x, p.y, p.z), self.floor)
        self.z = p.z
        if self.transit_z0 is not None and not self.direction_told and abs(p.z - self.transit_z0) >= DIRECTION_M:
            dz = p.z - self.transit_z0
            self.event(f"going {'up' if dz > 0 else 'down'} ({dz:+.1f} m so far)")
            self.direction_told = True

    def on_floor_state(self, m: FloorState) -> None:
        if m.has_baro and self.z is not None:
            if self.baro_offset is None:
                self.baro_offset = self.z - m.baro_height
            self.view.baro(stamp_s(m.header.stamp), m.baro_height + self.baro_offset)
        if m.floor < 0 and self.transit_z0 is None and self.z is not None:  # a transition began
            self.transit_z0, self.direction_told = self.z, False
        if m.floor >= 0 and m.floor != self.floor:  # arrived on a floor (type and height: the node's own log line)
            if m.floor in self.floors_seen:
                self.event(f"on floor {m.floor} again: recognised as an already mapped floor (same level)")
            elif self.floors_seen:
                self.event(f"new floor {m.floor}")
            else:
                self.event(f"starting on floor {m.floor}")
            self.floors_seen.add(m.floor)
            self.transit_z0 = None
        self.floor, self.state = m.floor, m.state

    def on_log(self, m: Log) -> None:
        if m.name == NODE_NAME:
            self.event(m.msg, LEVELS.get(m.level, "INFO"))

    def on_tf(self, m: TFMessage) -> None:
        for tr in m.transforms:
            floor = floor_of_frame(tr.header.frame_id)
            if floor is None:
                continue
            self.floor_from_odom[floor] = matrix(tr.transform)
            if floor in self.pending_clouds:
                self.on_cloud(self.pending_clouds.pop(floor))

    def on_sensor(self, sensor: str, m: PointCloud2) -> None:
        if self.floor is None:
            return
        p = point_cloud2.read_points_numpy(m, field_names=("x", "y", "z"), skip_nans=True)
        self.view.sensor_cloud(stamp_s(m.header.stamp), sensor, p)

    def on_camera(self, m: CameraView) -> None:
        # Drawn from the first frame, floor or not: a camera that never starts tracking is what it shows best.
        t = stamp_s(m.header.stamp)
        self.at(t)
        uv = np.stack([np.asarray(m.u, dtype=np.float32), np.asarray(m.v, dtype=np.float32)], axis=1)
        use = np.frombuffer(bytes(m.use), dtype=np.uint8)
        self.view.camera(t, image_array(m.image), depth_array(m.depth), uv, use)

    def on_cloud(self, m: PointCloud2) -> None:
        floor = floor_of_frame(m.header.frame_id)
        if floor is None:
            return
        if floor not in self.floor_from_odom:
            self.pending_clouds[floor] = m
            return
        p = point_cloud2.read_points_numpy(m, field_names=("x", "y", "z"), skip_nans=True).astype(np.float64)
        odom_from_floor = np.linalg.inv(self.floor_from_odom[floor])
        p = p @ odom_from_floor[:3, :3].T + odom_from_floor[:3, 3]
        self.view.floor_cloud(self.t or 0.0, floor, p)


# Exit code of the viewer when the user closed only the viewer and the SLAM goes on: the launch does not end then.
VIEWER_ONLY = 3
REOPEN, CLOSE_VIEWER, STOP_SLAM = 10, 11, 12  # answers of the dialog

# The question when the window is closed while the SLAM runs, in a process of its own (Tk wants the main thread, and
# the viewer keeps spinning meanwhile). Closing the dialog itself is "no, keep it": the window is opened again.
DIALOG = r"""
import sys
import tkinter as tk
root = tk.Tk()
root.title("StrataSLAM")
root.attributes("-topmost", True)
answer = [10]
def pick(code):
    answer[0] = code
    root.destroy()
tk.Label(root, text="The viewer window was closed, but the SLAM is still running.", padx=20, pady=12).pack()
for text, code in (("Reopen the viewer", 10), ("Close the viewer (the SLAM goes on)", 11),
                   ("Stop the SLAM and save the map", 12)):
    tk.Button(root, text=text, width=34, command=lambda c=code: pick(c)).pack(padx=20, pady=3)
tk.Label(root, text="", pady=4).pack()
root.protocol("WM_DELETE_WINDOW", lambda: pick(10))
root.mainloop()
sys.exit(answer[0])
"""


def children() -> set[int]:
    """PIDs of this process's children (Linux)."""
    out: set[int] = set()
    for task in Path("/proc/self/task").iterdir():
        try:
            out |= {int(x) for x in (task / "children").read_text().split()}
        except OSError:
            continue
    return out


def exited(pid: int | None) -> bool:
    """Whether the child `pid` has exited (it is reaped). None, no child of ours (a window that was already open was
    reused), never does."""
    if pid is None:
        return False
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return done != 0


def open_window(view: LiveView | None) -> tuple[LiveView, int | None]:
    """A new Rerun window (or `view` reopened in one); returns it with the PID of the window this viewer started."""
    before = children()
    if view is None:
        view = LiveView()
    else:
        view.reopen()
    new = children() - before
    return view, (min(new) if new else None)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    view, window = open_window(None)
    node = Viewer(view)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    code = 0
    dialog: subprocess.Popen[bytes] | None = None
    seen_slam = finished = False
    checked = 0.0
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.25)
            if time.monotonic() - checked > 1.0:  # the SLAM node in the ROS graph: started, then gone
                checked = time.monotonic()
                running = node.slam_running()
                seen_slam |= running
                if seen_slam and not running and not finished:
                    finished = True
                    text = "The SLAM has finished: close the viewer window, or press Ctrl+C, to end"
                    node.get_logger().info(text)
                    view.event(node.t or 0.0, text)
                    view.flush_paths()
            if dialog is not None:
                answer = dialog.poll()
                if answer is None:
                    continue
                dialog = None
                if answer == REOPEN:
                    view, window = open_window(view)
                    continue
                if answer == CLOSE_VIEWER:
                    node.get_logger().info("the viewer is closed; the SLAM goes on")
                    code = VIEWER_ONLY
                elif answer == STOP_SLAM:
                    node.get_logger().info("stopping the SLAM: it saves its map")
                else:  # no Tk (python3-tk is not installed): the viewer closes, as before
                    node.get_logger().warn("no dialog (python3-tk is missing): the viewer is closed, the SLAM goes on")
                    code = VIEWER_ONLY
                break
            if exited(window):
                window = None
                if finished or not node.slam_running():
                    break  # nothing runs any more: the launch ends
                dialog = subprocess.Popen([sys.executable, "-c", DIALOG])
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C (or the launch ending) closes the window this viewer opened, and the question if it is up.
        for pid in (window, dialog.pid if dialog is not None else None):
            if pid is not None and not exited(pid):
                os.kill(pid, signal.SIGTERM)
        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()  # also after Ctrl+C, whose handler already shut the context down
    sys.exit(code)


if __name__ == "__main__":
    main(sys.argv)
