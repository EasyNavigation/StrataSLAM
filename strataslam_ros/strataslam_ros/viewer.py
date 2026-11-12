"""ROS 2 node behind `viewer:=true`: draws a StrataSLAM run live in Rerun (strataslam_tools.live_view).

Everything is drawn in the map frame, as the node's TF tree has it (ADR-0075): each floor's cloud and
path (/strataslam/floor_cloud and /strataslam/path, in `floor_<k>`) placed with map -> floor_<k>, the
floor's fixed anchor, so that the floors keep their placement against each other; the robot (from
/strataslam/odometry) and what the sensors see through map -> floor_<k> -> odom, whose last link carries
the loop closures' correction, so that the robot stands on the optimized map. Without `map` (profile
`output.publish_tf: false`) each floor is placed as the node places it: when it appears, through the
floor the robot came from and that floor's correction then (with the new floor's first floor_<k> ->
odom, its anchor). The placement of the floors is the live estimate; the fused building comes from
`strataslam fuse`.

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
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg
from rcl_interfaces.msg import Log
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_srvs.srv import Trigger
from strataslam_tools.live_view import LOST, LiveView
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


TRANSITION_ENDED = re.compile(r"floor transition ended: (\S+) (\d+)->(\d+) dz=(-?[0-9.]+)")


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
        self.floor_from_odom: dict[int, np.ndarray] = {}  # T_floor_odom, latest per floor (with the correction)
        self.map_from_floor_tf: dict[int, np.ndarray] = {}  # T_map_floor from the TF map -> floor_<k>, when published
        self.placed: dict[int, np.ndarray] = {}  # T_map_floor as the node places it, for a run without map
        self.last_floor: int | None = None  # the floor the robot is on, or was on before the transition
        self.prev_floor: int | None = None  # the floor before last_floor: where a new floor is placed from
        self.root_anchor = np.eye(4)  # the first floor's first floor_<k> -> odom
        self.pending_clouds: dict[int, PointCloud2] = {}  # clouds that arrived before their floor's TF
        self.pending_paths: dict[int, PathMsg] = {}  # the same for the optimized paths
        self.z: float | None = None
        self.transit_z0: float | None = None  # height when the current transition began
        self.direction_told = False
        self.floors_seen: set[int] = set()
        self.keyframes: dict[int, int] = {}  # keyframes of each floor (its latest path): what discarding loses
        self.discard_client = self.create_client(Trigger, f"{NODE_NAME}/discard_map")
        self.early_events: list[tuple[str, str]] = []  # log lines before the first data (no data time yet)
        self.baro_offset: float | None = None  # robot height minus barometric height at the first reading
        reliable = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, "strataslam/odometry", self.on_odometry, reliable)
        self.create_subscription(FloorState, "strataslam/floor_state", self.on_floor_state, reliable)
        self.create_subscription(PointCloud2, "strataslam/floor_cloud", self.on_cloud, latched)
        self.create_subscription(PathMsg, "strataslam/path", self.on_path, reliable)
        # What the sensors see right now, in the odometry frame (drawn through map -> odom).
        self.create_subscription(PointCloud2, "strataslam/scan", lambda m: self.on_sensor("lidar", m), qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "strataslam/depth_cloud", lambda m: self.on_sensor("camera", m),
                                 qos_profile_sensor_data)
        self.create_subscription(CameraView, "strataslam/camera_view", self.on_camera, qos_profile_sensor_data)
        self.create_subscription(TFMessage, "/tf", self.on_tf, reliable)
        rosout = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Log, "/rosout", self.on_log, rosout)
        # The last stretch of path is sent even when poses stop arriving (end of a bag, robot stopped).
        self.create_timer(0.5, self.view.flush_paths)

    def discard_map(self, executor: SingleThreadedExecutor, timeout_s: float = 5.0) -> bool:
        """Asks the SLAM node not to save its map when it stops (~/discard_map); whether it agreed."""
        if not self.discard_client.wait_for_service(timeout_sec=timeout_s):
            return False
        future = self.discard_client.call_async(Trigger.Request())
        end = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < end:
            executor.spin_once(timeout_sec=0.1)
        return bool(future.done() and future.result() is not None and future.result().success)

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

    def map_from_floor(self, floor: int) -> np.ndarray | None:
        """T_map_floor: the node's TF map -> floor_<k> or, without it, the same placement made here."""
        if floor in self.map_from_floor_tf:
            return self.map_from_floor_tf[floor]
        return self.placed.get(floor)

    def place(self, floor: int, anchor: np.ndarray) -> None:
        """A floor seen for the first time, without map: placed as the node places it (the first floor is map; any
        other through the floor the robot came from and that floor's correction now)."""
        if not self.placed:
            self.placed[floor] = np.eye(4)
            self.root_anchor = anchor
            return
        src = self.prev_floor if self.last_floor == floor else self.last_floor  # the floor state may come first
        if src is not None and src in self.placed and src in self.floor_from_odom:
            self.placed[floor] = self.placed[src] @ self.floor_from_odom[src] @ np.linalg.inv(anchor)
        else:  # the floor it came from is not known (a viewer started late): the anchors
            self.placed[floor] = self.root_anchor @ np.linalg.inv(anchor)

    def map_from_odom(self) -> np.ndarray:
        """T_map_odom through the robot's floor: its anchor and the graph's correction (identity before any TF)."""
        floor = self.last_floor
        if floor is None or floor not in self.floor_from_odom:
            return np.eye(4)
        m = self.map_from_floor(floor)
        return np.eye(4) if m is None else m @ self.floor_from_odom[floor]

    def on_odometry(self, m: Odometry) -> None:
        t = stamp_s(m.header.stamp)
        self.at(t)
        if self.floor is None:
            return
        q = m.pose.pose.position
        T = self.map_from_odom()
        x, y, z = (T[:3, :3] @ np.array([q.x, q.y, q.z]) + T[:3, 3]).tolist()
        # -1: a transition; LOST: lost tracking or a session whose floor is not known yet (pending).
        floor = self.floor if self.floor >= 0 else (LOST if self.state in ("lost", "pending") else -1)
        self.view.pose(t, (x, y, z), floor)
        self.z = z
        if self.transit_z0 is not None and not self.direction_told and abs(z - self.transit_z0) >= DIRECTION_M:
            dz = z - self.transit_z0
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
        if m.floor >= 0 and m.floor != self.last_floor:
            self.prev_floor, self.last_floor = self.last_floor, m.floor

    def on_log(self, m: Log) -> None:
        if m.name == NODE_NAME:
            self.event(m.msg, LEVELS.get(m.level, "INFO"))
            # "floor transition ended: stairs 1->0 dz=-3.660844": where the floors join is marked and labelled.
            ended = TRANSITION_ENDED.match(m.msg)
            if ended and self.t is not None:
                kind, a, b, dz = ended.groups()
                self.view.transition(self.t, kind, int(a), int(b), float(dz))

    def on_tf(self, m: TFMessage) -> None:
        for tr in m.transforms:
            floor = floor_of_frame(tr.header.frame_id)
            child = floor_of_frame(tr.child_frame_id)
            if floor is None and child is not None:  # map -> floor_<k>
                floor = child
                self.map_from_floor_tf[floor] = matrix(tr.transform)
            elif floor is not None and child is None:  # floor_<k> -> odom
                t_floor_odom = matrix(tr.transform)
                if floor not in self.placed:
                    self.place(floor, t_floor_odom)  # before this floor's own correction replaces its anchor
                self.floor_from_odom[floor] = t_floor_odom
            else:
                continue
            if floor in self.pending_clouds:
                self.on_cloud(self.pending_clouds.pop(floor))
            if floor in self.pending_paths:
                self.on_path(self.pending_paths.pop(floor))

    def on_sensor(self, sensor: str, m: PointCloud2) -> None:
        if self.floor is None:
            return
        p = point_cloud2.read_points_numpy(m, field_names=("x", "y", "z"), skip_nans=True).astype(np.float64)
        T = self.map_from_odom()
        self.view.sensor_cloud(stamp_s(m.header.stamp), sensor, p @ T[:3, :3].T + T[:3, 3])

    def on_camera(self, m: CameraView) -> None:
        # Drawn from the first frame, floor or not: a camera that never starts tracking is what it shows best.
        t = stamp_s(m.header.stamp)
        self.at(t)
        uv = np.stack([np.asarray(m.u, dtype=np.float32), np.asarray(m.v, dtype=np.float32)], axis=1)
        use = np.frombuffer(bytes(m.use), dtype=np.uint8)
        self.view.camera(t, image_array(m.image), depth_array(m.depth), uv, use)

    def on_path(self, m: PathMsg) -> None:
        """The floor's optimized keyframes (in `floor_<k>`), placed in map as its cloud is. A gap of more than
        PATH_GAP_S between keyframes (another visit of the floor) starts a new piece."""
        floor = floor_of_frame(m.header.frame_id)
        if floor is None or not m.poses:
            return
        self.keyframes[floor] = len(m.poses)
        map_from_floor = self.map_from_floor(floor)
        if map_from_floor is None:
            self.pending_paths[floor] = m
            return
        xyz = np.array([[q.pose.position.x, q.pose.position.y, q.pose.position.z] for q in m.poses])
        xyz = xyz @ map_from_floor[:3, :3].T + map_from_floor[:3, 3]
        stamps = np.array([stamp_s(q.header.stamp) for q in m.poses])
        cuts = np.flatnonzero(np.diff(stamps) > PATH_GAP_S) + 1
        pieces = [[tuple(float(v) for v in row) for row in part] for part in np.split(xyz, cuts)]
        parts = np.split(stamps, cuts)
        spans = [(float(part[0]), float(part[-1])) for part in parts]
        self.view.floor_path(floor, pieces, spans, [[float(v) for v in part] for part in parts])

    def on_cloud(self, m: PointCloud2) -> None:
        floor = floor_of_frame(m.header.frame_id)
        if floor is None:
            return
        map_from_floor = self.map_from_floor(floor)
        if map_from_floor is None:
            self.pending_clouds[floor] = m
            return
        p = point_cloud2.read_points_numpy(m, field_names=("x", "y", "z"), skip_nans=True).astype(np.float64)
        colours = None
        if any(f.name == "rgb" for f in m.fields):  # the camera's colours (colorize_cloud, ADR-0069)
            colours = rgb_of(point_cloud2.read_points_numpy(m, field_names=("rgb",), skip_nans=True))
        p = p @ map_from_floor[:3, :3].T + map_from_floor[:3, 3]
        self.view.floor_cloud(self.t or 0.0, floor, p, colours)


def rgb_of(packed: np.ndarray) -> np.ndarray:
    """N x 3 uint8 from PCL's packed `rgb` (a FLOAT32 field holding 0x00RRGGBB)."""
    u = np.ascontiguousarray(packed, dtype=np.float32).reshape(-1).view(np.uint32)
    return np.stack([(u >> 16) & 0xFF, (u >> 8) & 0xFF, u & 0xFF], axis=1).astype(np.uint8)


PATH_GAP_S = 5.0  # keyframes further apart in time are drawn as separate pieces of a floor's path

# Exit code of the viewer when the user closed only the viewer and the SLAM goes on: the launch does not end then.
VIEWER_ONLY = 3
REOPEN, STOP_SLAM, DISCARD = 10, 12, 13  # answers of the dialog

# The question when the window is closed while the SLAM runs, in a process of its own (Tk wants the main thread, and
# the viewer keeps spinning meanwhile): reopen, stop and save, or stop without saving (asked again, with what would be
# lost). Closing the dialog itself is "no, keep it": the window is opened again. argv: floors, keyframes.
DIALOG = r"""
import sys
import tkinter as tk
from tkinter import messagebox
floors, keyframes = int(sys.argv[1]), int(sys.argv[2])
root = tk.Tk()
root.title("StrataSLAM")
root.attributes("-topmost", True)
answer = [10]
def pick(code):
    if code == 13 and not messagebox.askyesno(
            "StrataSLAM", f"Discard the map of this run ({floors} floor(s), {keyframes} keyframes)?\n"
                          "It will not be saved.", icon="warning", default="no", parent=root):
        return
    answer[0] = code
    root.destroy()
tk.Label(root, text="The viewer window was closed, but the SLAM is still running.", padx=20, pady=12).pack()
for text, code in (("Reopen the viewer", 10), ("Stop the SLAM and save the map", 12),
                   ("Stop the SLAM without saving the map", 13)):
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
                if answer == STOP_SLAM:
                    node.get_logger().info("stopping the SLAM: it saves its map")
                elif answer == DISCARD:
                    if node.discard_map(executor):
                        node.get_logger().warn("stopping the SLAM without saving its map")
                    else:  # an old node, or no answer: stopping saves, which loses nothing
                        node.get_logger().error("the SLAM did not accept discarding its map: it saves it as it stops")
                else:  # no Tk (python3-tk is not installed): the viewer closes, as before
                    node.get_logger().warn("no dialog (python3-tk is missing): the viewer is closed, the SLAM goes on")
                    code = VIEWER_ONLY
                break
            if exited(window):
                window = None
                if finished or not node.slam_running():
                    break  # nothing runs any more: the launch ends
                dialog = subprocess.Popen([sys.executable, "-c", DIALOG, str(max(len(node.keyframes), 1)),
                                           str(sum(node.keyframes.values()))])
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
