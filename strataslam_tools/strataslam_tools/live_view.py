"""Live view of a StrataSLAM run in Rerun: the 3D map with each floor in its colour, the top view, the
level line (height of the robot over time, coloured by floor; with a barometer, its height too) and the
log of the run below them. With a camera, its image with the keypoints and the depth image used appear
next to the log.

ROS-free: `strataslam_ros/viewer.py` feeds it from the node's topics; anything that has poses, floor
states and floor clouds can. Needs the optional extra `viewer` (`pip install rerun-sdk`); with ROS 2
Jazzy, `rerun-sdk==0.22.1`, the last release that works with its numpy 1.x.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Points = NDArray[np.floating[Any]]

# Distinct colours, one per floor (cycled after ten floors); transitions and lost tracking in grey.
FLOOR_COLOURS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                 (140, 86, 75), (227, 119, 194), (188, 189, 34), (23, 190, 207), (127, 127, 127)]
TRANSIT_COLOUR = (160, 160, 160)
# The robot and its path must stand out on clouds of any floor colour.
ROBOT_COLOUR = (255, 215, 0)
PATH_COLOUR = (245, 245, 245)
SENSOR_COLOUR = (255, 64, 255)  # what the sensors see right now (and the barometer in the level line)
# Keypoints of the camera view by what the tracking did with them (`use` of strataslam_msgs/CameraView).
KEYPOINT_CLASSES = [(0, "unmatched", (230, 40, 40)), (1, "matched, rejected", (255, 140, 0)),
                    (2, "used for tracking", (40, 220, 40)), (3, "kept for loop closures", (0, 200, 255))]
# The views fit what is drawn: the live scan (far returns that change every frame) is cut to the map's extent
# plus this margin, so the framing follows the map and does not jump with each scan.
SENSOR_MARGIN_M = 1.0
# The robot's pole in 3D crosses every level mapped so far: from below the lowest point to above the highest.
POLE_BELOW_M, POLE_ABOVE_M = 1.0, 3.0
ROBOT_RADIUS, POLE_RADIUS = 14.0, 3.0  # UI points: the ball stands out around the pole
# Top view layering (Rerun draws 2D points above lines by default): clouds, live scan, path, robot.
DRAW_SENSOR, DRAW_PATH, DRAW_ROBOT = 35.0, 40.0, 50.0
KEYPOINT_RADIUS = 3.0  # UI points
PATH_PERIOD_S = 0.25  # the paths are re-sent at most this often (wall clock): they grow with the run
TOPDOWN_MAX_POINTS = 300_000


def floor_colour(floor: int) -> tuple[int, int, int]:
    return FLOOR_COLOURS[floor % len(FLOOR_COLOURS)] if floor >= 0 else TRANSIT_COLOUR


def _rerun() -> Any:
    try:
        import rerun as rr
    except ImportError as e:
        raise SystemExit("the live view needs Rerun: pip install rerun-sdk "
                         "(with ROS 2 Jazzy: pip install rerun-sdk==0.22.1)") from e
    return rr


class LiveView:
    """One Rerun recording. Times are the stamps of the data in seconds, shown from the first one."""

    def __init__(self, app_id: str = "StrataSLAM", spawn: bool = True, save: Path | None = None) -> None:
        self.rr = rr = _rerun()
        rr.init(app_id)
        if save is not None:
            rr.save(str(save))
        elif spawn:
            rr.spawn()
        self._camera = False
        self._floor_clouds: dict[int, NDArray[np.float32]] = {}  # latest cloud of each floor (sent again on reopen)
        self._send_static()
        self._baro_styled = False
        self._strips: dict[int, list[list[tuple[float, float, float]]]] = {}  # floor -> path pieces
        self._last_floor: int | None = None
        self._t0: float | None = None
        self._series: set[int] = set()
        self._last_path_wall = 0.0
        self._dirty = False
        self._z_top: float | None = None  # highest and lowest points mapped so far (ends of the robot's pole)
        self._z_bottom: float | None = None
        self._map_lo: NDArray[np.float32] | None = None  # extent of the map drawn so far (live scan is cut to it)
        self._map_hi: NDArray[np.float32] | None = None

    def _send_static(self) -> None:
        rr = self.rr
        self._send_blueprint(camera=self._camera)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("camera", rr.AnnotationContext(KEYPOINT_CLASSES), static=True)  # colour and name of each `use`

    def reopen(self) -> None:
        """A new window after the user closed the previous one: the layout, the floor clouds and the paths are sent
        again (they are static, latest only); the history of the timeline (the level line, the log) is not."""
        self.rr.spawn()
        self._send_static()
        self._series.clear()  # styles of the level line, sent again with the next value
        self._baro_styled = False
        for floor, p in self._floor_clouds.items():
            self._log_floor_cloud(floor, p)
        self._dirty = True
        self.flush_paths()

    def _send_blueprint(self, camera: bool) -> None:
        """The layout. With a camera, its views share the bottom row with the log (sent again on its first frame)."""
        import rerun.blueprint as rrb

        log = rrb.TextLogView(origin="/events", name="Log")
        bottom: Any = log
        if camera:
            bottom = rrb.Horizontal(
                log,
                rrb.Spatial2DView(origin="/camera", name="Camera: green used, orange rejected, red unmatched, cyan for loops"),
                rrb.Spatial2DView(origin="/depth", name="Depth used"),
                column_shares=[2, 1, 1],
            )
        self.rr.send_blueprint(rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="/world", name="3D"),
                    rrb.Vertical(
                        rrb.Spatial2DView(origin="/topdown", name="Top view"),
                        rrb.TimeSeriesView(origin="/levels", name="Level line (z over time)",
                                           plot_legend=rrb.PlotLegend(corner=rrb.Corner2D.LeftBottom)),
                    ),
                    column_shares=[3, 2],
                ),
                bottom,
                row_shares=[3, 2] if camera else [4, 1],
            ),
            # Left panel open: its entity tree has an eye per item to show or hide it (all floor clouds at
            # once with `floors`, or one floor, the path, the robot); the selection panel stays closed.
            rrb.BlueprintPanel(state="expanded"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="expanded"),  # the streams show when each floor, transition and log line happened
        ))

    # ---- time and version differences (rerun 0.22 for numpy 1.x, 0.23+ otherwise) ----

    def _time(self, t: float) -> None:
        rr = self.rr
        if self._t0 is None:
            self._t0 = t
        t -= self._t0  # seconds since the start of the run, not a date
        if hasattr(rr, "set_time"):
            rr.set_time("time", duration=t)
        else:
            rr.set_time_seconds("time", t)

    def _scalar(self, path: str, value: float) -> None:
        rr = self.rr
        self.rr.log(path, rr.Scalars(value) if hasattr(rr, "Scalars") else rr.Scalar(value))

    def _series_style(self, path: str, colour: tuple[int, int, int], name: str) -> None:
        rr = self.rr
        if hasattr(rr, "SeriesLines"):
            rr.log(path, rr.SeriesLines(colors=[colour], names=[name]), static=True)
        else:
            rr.log(path, rr.SeriesLine(color=colour, name=name), static=True)

    @staticmethod
    def _series_path(floor: int) -> str:
        return f"levels/floor_{floor}" if floor >= 0 else "levels/transition"

    def _radius(self, ui_points: float) -> Any:
        return self.rr.Radius.ui_points(ui_points)

    # ---- data ----

    def pose(self, t: float, xyz: tuple[float, float, float], floor: int) -> None:
        """A robot position in the common (odometry) frame; `floor` -1 while in a transition or lost."""
        rr = self.rr
        self._time(t)
        colour = floor_colour(floor)
        x, y, z = xyz
        rr.log("world/robot", rr.Points3D([xyz], colors=[ROBOT_COLOUR], radii=self._radius(ROBOT_RADIUS)))
        top = max(z, self._z_top if self._z_top is not None else z) + POLE_ABOVE_M
        bottom = min(z, self._z_bottom if self._z_bottom is not None else z) - POLE_BELOW_M
        rr.log("world/robot/pole", rr.LineStrips3D([[(x, y, bottom), (x, y, top)]], colors=[ROBOT_COLOUR],
                                                   radii=self._radius(POLE_RADIUS)))
        rr.log("topdown/robot", rr.Points2D([(x, -y)], colors=[ROBOT_COLOUR], radii=self._radius(8.0),
                                            draw_order=DRAW_ROBOT))
        if floor != self._last_floor:  # a new piece of path, so floors are not joined by a straight line
            if self._last_floor is not None:  # and a gap in the level line of the floor just left
                rr.log(self._series_path(self._last_floor), rr.Clear(recursive=False))
            self._strips.setdefault(floor, []).append([])
            self._last_floor = floor
        self._strips[floor][-1].append(xyz)
        self._dirty = True
        series = self._series_path(floor)
        if floor not in self._series:
            self._series.add(floor)
            self._series_style(series, colour, f"floor {floor}" if floor >= 0 else "transition / lost")
        self._scalar(series, z)
        if time.monotonic() - self._last_path_wall >= PATH_PERIOD_S:
            self.flush_paths()

    def flush_paths(self) -> None:
        """Sends the paths travelled so far (they are sent at most every PATH_PERIOD_S while posing)."""
        if not self._dirty:
            return
        rr = self.rr
        # Static (latest only): the whole path is re-sent as it grows, and keeping every version for the
        # timeline would take gigabytes on a long run. The robot, the level line and the log keep their history.
        for floor, strips in self._strips.items():
            name = f"floor_{floor}" if floor >= 0 else "transition"
            pieces = [s for s in strips if len(s) >= 2]
            rr.log(f"world/path/{name}", rr.LineStrips3D(pieces, colors=[PATH_COLOUR], radii=self._radius(2.0)),
                   static=True)
            rr.log(f"topdown/path/{name}", rr.LineStrips2D([[(x, -y) for x, y, _ in s] for s in pieces],
                                                           colors=[PATH_COLOUR], radii=self._radius(2.0),
                                                           draw_order=DRAW_PATH), static=True)
        self._last_path_wall = time.monotonic()
        self._dirty = False

    def floor_cloud(self, t: float, floor: int, points: Points) -> None:
        """The cloud of one floor in the common frame (N x 3); replaces the previous one of that floor (static,
        latest only: every refresh carries the whole floor)."""
        self._time(t)
        p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(p):
            bottom, top = (float(v) for v in np.percentile(p[:, 2], [1, 99]))  # stray points do not stretch the pole
            self._z_top = top if self._z_top is None else max(self._z_top, top)
            self._z_bottom = bottom if self._z_bottom is None else min(self._z_bottom, bottom)
            lo, hi = np.percentile(p, [1, 99], axis=0).astype(np.float32)
            self._map_lo = lo if self._map_lo is None else np.minimum(self._map_lo, lo)
            self._map_hi = hi if self._map_hi is None else np.maximum(self._map_hi, hi)
        self._floor_clouds[floor] = p
        self._log_floor_cloud(floor, p)

    def _log_floor_cloud(self, floor: int, p: NDArray[np.float32]) -> None:
        rr = self.rr
        colour = floor_colour(floor)
        rr.log(f"world/floors/floor_{floor}", rr.Points3D(p, colors=[colour], radii=self._radius(1.0)), static=True)
        step = max(1, len(p) // TOPDOWN_MAX_POINTS)
        q = p[::step, :2] * np.array([1.0, -1.0], dtype=np.float32)
        rr.log(f"topdown/floors/floor_{floor}", rr.Points2D(q, colors=[colour], radii=self._radius(1.0)), static=True)

    def sensor_cloud(self, t: float, sensor: str, points: Points) -> None:
        """What a sensor (`lidar`, `camera`) sees right now, in the common frame (N x 3), within the map's
        extent (nothing before the first floor cloud). Static, latest only: under `sensor` in the entity tree,
        whose eye shows or hides it."""
        rr = self.rr
        if self._map_lo is None or self._map_hi is None:
            return
        self._time(t)
        p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        inside = np.all((p >= self._map_lo - SENSOR_MARGIN_M) & (p <= self._map_hi + SENSOR_MARGIN_M), axis=1)
        p = p[inside]
        rr.log(f"world/sensor/{sensor}", rr.Points3D(p, colors=[SENSOR_COLOUR], radii=self._radius(1.5)), static=True)
        q = p[:, :2] * np.array([1.0, -1.0], dtype=np.float32)
        rr.log(f"topdown/sensor/{sensor}", rr.Points2D(q, colors=[SENSOR_COLOUR], radii=self._radius(1.5),
                                                       draw_order=DRAW_SENSOR), static=True)

    def camera(self, t: float, image: NDArray[np.uint8] | None, depth: NDArray[np.float32] | None, uv: Points,
               use: NDArray[np.uint8]) -> None:
        """A camera frame: the image (H x W x 3 or H x W; None with feature-frame input), the depth used in metres
        (0 where invalid) and the keypoints (N x 2 pixels) with what the tracking did with each (`use`, the
        classes of KEYPOINT_CLASSES). Static, latest only: a long run would otherwise keep gigabytes of images."""
        rr = self.rr
        if not self._camera:  # the camera views appear with the first frame
            self._camera = True
            self._send_blueprint(camera=True)
        self._time(t)
        if image is not None:
            rr.log("camera/image", rr.Image(image), static=True)
        rr.log("camera/keypoints", rr.Points2D(np.asarray(uv, dtype=np.float32).reshape(-1, 2), class_ids=use,
                                               radii=self._radius(KEYPOINT_RADIUS), draw_order=DRAW_ROBOT), static=True)
        if depth is not None:
            rr.log("depth/image", rr.DepthImage(depth, meter=1.0, colormap="turbo"), static=True)

    def baro(self, t: float, z: float) -> None:
        """Barometric height at `t`, already in the robot's height reference: a line of its own in the level line."""
        self._time(t)
        if not self._baro_styled:
            self._baro_styled = True
            self._series_style("levels/barometer", SENSOR_COLOUR, "barometer")
        self._scalar("levels/barometer", z)

    def event(self, t: float, text: str, level: str = "INFO") -> None:
        """A line of the log panel at time `t`: the node's log, a transition, a new or a known floor.
        `level` is INFO, WARN or ERROR (the panel colours and filters by it)."""
        rr = self.rr
        self._time(t)
        rr.log("events", rr.TextLog(text, level=level))
