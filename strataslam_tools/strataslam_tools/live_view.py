"""Live view of a StrataSLAM run in Rerun: the 3D map with each floor in its colour (or, layers of the same view
shown with their eye, coloured by height, or with the camera's colours when the map has them), the top view, the
level line (height of the robot over time, coloured by floor; with a barometer, its height too) and the
log of the run below them. With a camera, its image with the keypoints and the depth image used appear
next to the log.

ROS-free: `strataslam_ros/viewer.py` feeds it from the node's topics; anything that has poses, floor
states and floor clouds can. Needs the optional extra `viewer` (`pip install rerun-sdk`); with ROS 2
Jazzy, `rerun-sdk==0.22.1` with `pyarrow==16.0.0`, the releases that work with its numpy 1.x.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Points = NDArray[np.floating[Any]]

# Distinct colours, one per floor (cycled after ten floors); transitions in cyan and lost tracking in grey, in the level
# line and on the path (where the path is white on the floors): a ride the odometry did not see is the join of its two
# floors. `pose` takes floor -1 for a transition and LOST (-2) for lost tracking or a session whose floor is pending.
FLOOR_COLOURS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                 (140, 86, 75), (227, 119, 194), (188, 189, 34), (23, 190, 207), (127, 127, 127)]
TRANSIT_COLOUR = (0, 220, 255)
LOST_COLOUR = (160, 160, 160)
LOST = -2
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
# Top view layering (Rerun draws 2D points above lines by default): clouds, live scan, path, transitions, robot.
DRAW_SENSOR, DRAW_PATH, DRAW_TRANSITION, DRAW_ROBOT = 35.0, 40.0, 45.0, 50.0
# Where floors join: a transition's arrival (labelled with its type and height) and its departure, in UI points.
ARRIVAL_RADIUS, DEPARTURE_RADIUS = 8.0, 5.0
RIDE_SHARE = 0.5  # a join that climbs this share of the transition's height is a ride the odometry did not see
KEYPOINT_RADIUS = 3.0  # UI points
PATH_PERIOD_S = 0.25  # the paths are re-sent at most this often (wall clock): they grow with the run
JOIN_WINDOW_S = 10.0  # s: a join meets a floor's path or a transition's trail only at a point this near its time (a floor
# reached gets its first keyframe once the transition settles, 3 s after on the Summit's elevator bag)
# s: a transition's trail cut by a floor's path (the part of a climb that joined the floor) is joined to it when the
# floor's path has a point this near the cut in time.
CUT_WINDOW_S = 2.0
TRANSIT_MARGIN_S = 0.5  # s around a floor's keyframes whose transition poses its path replaces (a keyframe's spacing)
TOPDOWN_MAX_POINTS = 300_000
# Points of one floor sent to the 3D views (each tab): a floor of a 1 km drive has 9 million, more than the viewer
# and the graphics card take every refresh.
WORLD_MAX_POINTS = 1_500_000
# The height colours span the lowest to the highest point mapped so far; when that range grows by more than this, every
# floor is coloured again, so that one height has one colour across the map.
HEIGHT_RECOLOUR_M = 0.5
TURBO_LO, TURBO_HI = 0.1, 0.9  # the part of the Turbo map used: blue to red


def height_colours(z: NDArray[np.float32], lo: float, hi: float) -> NDArray[np.uint8]:
    """Turbo colour map (blue low, red high) of heights between lo and hi: N x 3 RGB. Polynomial fit of Google's Turbo
    (no matplotlib needed)."""
    x = np.clip((np.asarray(z, dtype=np.float64) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    x = TURBO_LO + x * (TURBO_HI - TURBO_LO)  # without the near-black ends, lost on the viewer's dark background
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    rgb: NDArray[np.uint8] = (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb


def floor_colour(floor: int) -> tuple[int, int, int]:
    if floor >= 0:
        return FLOOR_COLOURS[floor % len(FLOOR_COLOURS)]
    return LOST_COLOUR if floor == LOST else TRANSIT_COLOUR


def _floor_name(floor: int) -> str:
    return f"floor_{floor}" if floor >= 0 else ("lost" if floor == LOST else "transition")


def _rerun() -> Any:
    try:
        import rerun as rr
    except ImportError as e:
        raise SystemExit(f"the live view needs Rerun ({e}): pip install rerun-sdk "
                         "(with ROS 2 Jazzy: pip install rerun-sdk==0.22.1 pyarrow==16.0.0)") from e
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
        self._floor_colours: dict[int, NDArray[np.uint8]] = {}  # the camera's colours of those clouds, when they have
        self._colour = False  # some cloud came with colours: the tab "3D: colour" is in the layout
        self._send_static()
        self._baro_styled = False
        self._strips: dict[int, list[list[tuple[float, float, float]]]] = {}  # floor -> path pieces
        self._transit_t: list[list[float]] = []  # the stamps of the points of the transitions' pieces (floor -1)
        self._last_floor: int | None = None
        # Where the floor changed (time, floor left, floor reached, last pose before, first pose after), drawn as a
        # transition from the path left to the path reached. An elevator ride the odometry did not see (the cabin does
        # not move against the LiDAR) is all here: its height comes at once.
        self._joins: list[tuple[float, int, int, tuple[float, float, float], tuple[float, float, float]]] = []
        self._last_xyz: tuple[float, float, float] | None = None
        # The stamps of each floor's optimized path, piece by piece as in _strips: where a join meets it.
        self._path_times: dict[int, list[list[float]]] = {}
        # The transitions that changed floor, as the node closed them: (time, type, from, to, height).
        self._transitions: list[tuple[float, str, int, int, float]] = []
        self._t0: float | None = None
        self._series: set[int] = set()
        self._last_path_wall = 0.0
        self._dirty = False
        self._z_top: float | None = None  # highest and lowest points mapped so far (ends of the robot's pole)
        self._z_bottom: float | None = None
        self._map_lo: NDArray[np.float32] | None = None  # extent of the map drawn so far (live scan is cut to it)
        self._map_hi: NDArray[np.float32] | None = None
        self._height_range: tuple[float, float] | None = None  # range of the height colours sent so far

    def _send_static(self) -> None:
        rr = self.rr
        self._send_blueprint(camera=self._camera)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._height_range = None  # a new window: the height colours are sent again with the clouds
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
        self._log_height_layer()
        self._log_colour_layer()
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
                    # One 3D view, so that one camera looks at every layer of the map: each floor in its colour
                    # (`floors`, shown), coloured by height (`height`) and with the camera's colours (`colour`, when
                    # the map has them), the last two hidden; the eye of each in the panel shows or hides it.
                    rrb.Spatial3DView(origin="/world", name="3D (layers in the panel: floors, height, colour)",
                                      overrides={"world/height": [rrb.components.Visible(False)],
                                                 "world/colour": [rrb.components.Visible(False)]}),
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
        return f"levels/{_floor_name(floor)}"

    def _radius(self, ui_points: float) -> Any:
        return self.rr.Radius.ui_points(ui_points)

    # ---- data ----

    def pose(self, t: float, xyz: tuple[float, float, float], floor: int) -> None:
        """A robot position in the map frame; `floor` -1 while in a transition or lost."""
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
        if floor != self._last_floor:  # a new piece of path, joined to the last one as a transition
            if self._last_floor is not None:  # and a gap in the level line of the floor just left
                rr.log(self._series_path(self._last_floor), rr.Clear(recursive=False))
                if self._last_xyz is not None:
                    self._joins.append((t, self._last_floor, floor, self._last_xyz, (x, y, z)))
            self._strips.setdefault(floor, []).append([])
            if floor == -1:
                self._transit_t.append([])
            self._last_floor = floor
        self._strips[floor][-1].append(xyz)
        self._last_xyz = (x, y, z)
        if floor == -1:
            self._transit_t[-1].append(t)
        self._dirty = True
        series = self._series_path(floor)
        if floor not in self._series:
            self._series.add(floor)
            label = f"floor {floor}" if floor >= 0 else ("tracking lost" if floor == LOST else "transition")
            self._series_style(series, colour, label)
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
        floors = set(self._strips) | ({-1} if self._joins else set())  # the joins are drawn with the transitions
        for floor in floors:
            name = _floor_name(floor)
            pieces = [s for s in self._strips.get(floor, []) if len(s) >= 2]
            colour = PATH_COLOUR if floor >= 0 else floor_colour(floor)
            if floor == -1:
                pieces = pieces + [[self._path_point(a, t, p), self._path_point(b, t, q)] for t, a, b, p, q in self._joins]
                pieces = pieces + self._cut_joins()
            rr.log(f"world/path/{name}", rr.LineStrips3D(pieces, colors=[colour], radii=self._radius(2.0)),
                   static=True)
            rr.log(f"topdown/path/{name}", rr.LineStrips2D([[(x, -y) for x, y, _ in s] for s in pieces],
                                                           colors=[colour], radii=self._radius(2.0),
                                                           draw_order=DRAW_PATH), static=True)
        arrivals, labels, departures = self._transition_marks()
        if arrivals:
            rr.log("world/transitions/arrival", rr.Points3D(arrivals, colors=[TRANSIT_COLOUR], labels=labels,
                                                            radii=self._radius(ARRIVAL_RADIUS)), static=True)
            rr.log("world/transitions/departure", rr.Points3D(departures, colors=[TRANSIT_COLOUR],
                                                              radii=self._radius(DEPARTURE_RADIUS)), static=True)
            rr.log("topdown/transitions/arrival", rr.Points2D([(x, -y) for x, y, _ in arrivals], colors=[TRANSIT_COLOUR],
                                                              labels=labels, radii=self._radius(ARRIVAL_RADIUS),
                                                              draw_order=DRAW_TRANSITION), static=True)
            rr.log("topdown/transitions/departure", rr.Points2D([(x, -y) for x, y, _ in departures],
                                                                colors=[TRANSIT_COLOUR],
                                                                radii=self._radius(DEPARTURE_RADIUS),
                                                                draw_order=DRAW_TRANSITION), static=True)
        self._last_path_wall = time.monotonic()
        self._dirty = False

    def transition(self, t: float, kind: str, from_floor: int, to_floor: int, dz: float) -> None:
        """A transition the node closed from one floor to another (`floor transition ended: ...`): its departure and
        arrival are marked, the arrival labelled, where the path joins the two floors."""
        if from_floor != to_floor:
            self._transitions.append((t, kind, from_floor, to_floor, dz))
            self._dirty = True

    def _transition_marks(self) -> tuple[list[tuple[float, float, float]], list[str], list[tuple[float, float, float]]]:
        """Arrivals with their labels, and departures. The arrival is where the join into the floor reached ends (the
        join nearest in time to the node's message); the departure is the foot of that join when it climbs most of the
        transition's height (a ride the odometry did not see), or else where the robot left the floor it came from
        (the top or the foot of the stairs or the ramp)."""
        arrivals: list[tuple[float, float, float]] = []
        labels: list[str] = []
        departures: list[tuple[float, float, float]] = []
        for t, kind, a, b, dz in self._transitions:
            cands = [j for j in self._joins if j[2] == b and j[1] < 0 and abs(j[0] - t) <= 60.0]
            if not cands:
                continue
            tj, ja, jb, p0, q0 = min(cands, key=lambda j: abs(j[0] - t))
            p, q = self._path_point(ja, tj, p0), self._path_point(jb, tj, q0)
            arrivals.append(q)
            labels.append(f"{kind} {a}\u2192{b} {dz:+.2f} m")
            # A ride the odometry did not see climbs in the join (most of the height the node measured): it left from
            # the join's foot, in the cabin. Stairs or a ramp climb along the trail: they left where the robot left
            # the floor it came from, the latest join from that floor into the transition.
            left = [j for j in self._joins if j[1] == a and j[2] < 0 and j[0] <= tj]
            if abs(q[2] - p[2]) >= RIDE_SHARE * abs(dz) or not left:
                departures.append(p)
            else:
                lt, _, lb, _, lq = max(left, key=lambda j: j[0])
                departures.append(self._path_point(lb, lt, lq))
        return arrivals, labels, departures

    def _floor_point_between(self, t0: float, t1: float, nearest: float) -> tuple[float, float, float] | None:
        """The point of any floor's optimized path stamped within [t0, t1], the nearest in time to `nearest`."""
        best: tuple[float, tuple[float, float, float]] | None = None
        for floor, stamps in self._path_times.items():
            for piece, times in zip(self._strips.get(floor, []), stamps):
                for point, s in zip(piece, times):
                    if t0 <= s <= t1 and (best is None or abs(s - nearest) < best[0]):
                        best = (abs(s - nearest), point)
        return None if best is None else best[1]

    def _cut_joins(self) -> list[list[tuple[float, float, float]]]:
        """Where a floor's path took over part of a transition's trail (stairs whose lower half joined the floor
        reached), the trail and the path meet at the cut: the trail as the odometry left it, the path as the floor's
        loop closures corrected it. They are joined there, so that the path goes on and the step between the two
        shows (Elevator-VIGS Office4, 2.1 m on the stairs back to floor 0)."""
        joins: list[list[tuple[float, float, float]]] = []
        for piece, times in zip(self._strips.get(-1, []), self._transit_t):
            if not piece or not times:
                continue
            before = self._floor_point_between(times[0] - CUT_WINDOW_S, times[0], times[0])
            if before is not None and before != piece[0]:
                joins.append([before, piece[0]])
            after = self._floor_point_between(times[-1], times[-1] + CUT_WINDOW_S, times[-1])
            if after is not None and after != piece[-1]:
                joins.append([piece[-1], after])
        return joins

    def _path_point(self, floor: int, t: float, pose: tuple[float, float, float]) -> tuple[float, float, float]:
        """Where a join at time `t` meets `floor`: the point nearest in time of the floor's optimized path, or of the
        transitions' trail as drawn, within JOIN_WINDOW_S, or else the pose drawn then (a floor with no path yet, lost
        tracking, a trail a floor's path took). The floor
        state and the odometry's jump at the end of a ride do not arrive together: the first pose on the new floor can
        still be at the old height, or the last of the transition already at the new one (and then taken by the
        floor's path)."""
        best: tuple[float, tuple[float, float, float]] | None = None
        # A floor's optimized path, or the transitions' trail as drawn (what a floor's path took is no longer in it).
        stamps = self._path_times.get(floor, []) if floor >= 0 else (self._transit_t if floor == -1 else [])
        for piece, times in zip(self._strips.get(floor, []) if floor >= -1 else [], stamps):
            for point, s in zip(piece, times):
                if abs(s - t) <= JOIN_WINDOW_S and (best is None or abs(s - t) < best[0]):
                    best = (abs(s - t), point)
        return pose if best is None else best[1]

    def floor_path(self, floor: int, pieces: list[list[tuple[float, float, float]]],
                   spans: list[tuple[float, float]] | None = None, times: list[list[float]] | None = None) -> None:
        """The optimized path of a floor (its keyframes, in pieces: one per visit), in the common frame. It replaces
        what the poses drew there, which a loop closure leaves with the drift it corrected; the poses that follow
        continue it from its last keyframe. `spans`, the first and last stamp of each piece: what the poses drew
        as a transition over that time is the floor's now (a transition that ended on a floor, or the part of a
        climb that joined it), and is no longer drawn twice. `times`, the stamp of every point: where the floor
        changes are joined to it."""
        if times is not None:
            times = [list(ts) for q, ts in zip(pieces, times) if q]
        pieces = [list(q) for q in pieces if q]
        if not pieces:
            return
        if times is not None:
            self._path_times[floor] = times
        if spans:
            self._drop_transit(spans)
        if floor == self._last_floor:
            pieces.append([pieces[-1][-1]])  # the live tail, from the last keyframe
        self._strips[floor] = pieces
        self._dirty = True

    def _drop_transit(self, spans: list[tuple[float, float]]) -> None:
        """Removes from the transitions' pieces the points stamped within `spans`, splitting a piece where they go."""
        old = self._strips.get(-1)
        if not old:
            return
        strips: list[list[tuple[float, float, float]]] = []
        times: list[list[float]] = []
        for xyz, ts in zip(old, self._transit_t):
            keep_xyz: list[tuple[float, float, float]] = []
            keep_t: list[float] = []
            for p, t in zip(xyz, ts):
                if any(t0 - TRANSIT_MARGIN_S <= t <= t1 + TRANSIT_MARGIN_S for t0, t1 in spans):
                    if keep_xyz:
                        strips.append(keep_xyz)
                        times.append(keep_t)
                    keep_xyz, keep_t = [], []
                else:
                    keep_xyz.append(p)
                    keep_t.append(t)
            if keep_xyz or (xyz is old[-1] and self._last_floor == -1):  # the live piece goes on being drawn
                strips.append(keep_xyz)
                times.append(keep_t)
        self._strips[-1] = strips
        self._transit_t = times
        self._dirty = True

    def floor_cloud(self, t: float, floor: int, points: Points, colours: NDArray[np.uint8] | None = None) -> None:
        """The cloud of one floor in the common frame (N x 3), with the camera's colours (N x 3 RGB) when the map has
        them; replaces the previous one of that floor (static, latest only: every refresh carries the whole floor)."""
        self._time(t)
        p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if colours is not None and len(colours) == len(p):
            self._colour = True
            self._floor_colours[floor] = np.asarray(colours, dtype=np.uint8).reshape(-1, 3)
        if len(p):
            bottom, top = (float(v) for v in np.percentile(p[:, 2], [1, 99]))  # stray points do not stretch the pole
            self._z_top = top if self._z_top is None else max(self._z_top, top)
            self._z_bottom = bottom if self._z_bottom is None else min(self._z_bottom, bottom)
            lo, hi = np.percentile(p, [1, 99], axis=0).astype(np.float32)
            self._map_lo = lo if self._map_lo is None else np.minimum(self._map_lo, lo)
            self._map_hi = hi if self._map_hi is None else np.maximum(self._map_hi, hi)
        self._floor_clouds[floor] = p
        self._recolour_heights()
        self._log_floor_cloud(floor, p)
        self._log_height_layer()
        if floor in self._floor_colours:
            self._log_colour_layer()

    def _recolour_heights(self) -> bool:
        """Whether the range of the map's heights moved enough to colour every floor again (and takes it)."""
        if self._z_bottom is None or self._z_top is None:
            return False
        lo, hi = self._z_bottom, self._z_top
        if self._height_range is not None and abs(lo - self._height_range[0]) < HEIGHT_RECOLOUR_M \
                and abs(hi - self._height_range[1]) < HEIGHT_RECOLOUR_M:
            return False
        self._height_range = (lo, hi)
        return True

    def _log_floor_cloud(self, floor: int, p: NDArray[np.float32]) -> None:
        rr = self.rr
        colour = floor_colour(floor)
        p = p[::max(1, len(p) // WORLD_MAX_POINTS)]
        rr.log(f"world/floors/floor_{floor}", rr.Points3D(p, colors=[colour], radii=self._radius(1.0)), static=True)
        step = max(1, len(p) // TOPDOWN_MAX_POINTS)
        q = p[::step, :2] * np.array([1.0, -1.0], dtype=np.float32)
        rr.log(f"topdown/floors/floor_{floor}", rr.Points2D(q, colors=[colour], radii=self._radius(1.0)), static=True)

    # The layers `height` and `colour` are one entity each, every floor in it: Rerun 0.22 hides an entity from the
    # layout (they start hidden) but not the floors under a parent, and a layout sent again would reset the camera.

    def _log_colour_layer(self) -> None:
        """Every floor with the camera's colours (grey where no image saw it): the layer `colour`."""
        floors = [f for f in self._floor_clouds if f in self._floor_colours and len(self._floor_colours[f]) ==
                  len(self._floor_clouds[f])]
        if not floors:
            return
        p = np.concatenate([self._floor_clouds[f] for f in floors])
        c = np.concatenate([self._floor_colours[f] for f in floors])
        step = max(1, len(p) // WORLD_MAX_POINTS)
        self.rr.log("world/colour", self.rr.Points3D(p[::step], colors=c[::step], radii=self._radius(1.0)), static=True)

    def _log_height_layer(self) -> None:
        """Every floor coloured by height, one colour per height across the map: the layer `height`."""
        if not self._floor_clouds or (self._height_range is None and not self._recolour_heights()):
            return
        assert self._height_range is not None
        lo, hi = self._height_range
        p = np.concatenate(list(self._floor_clouds.values()))
        p = p[::max(1, len(p) // WORLD_MAX_POINTS)]
        self.rr.log("world/height", self.rr.Points3D(p, colors=height_colours(p[:, 2], lo, hi), radii=self._radius(1.0)),
                    static=True)

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
