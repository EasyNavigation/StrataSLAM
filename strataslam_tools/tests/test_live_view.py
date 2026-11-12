"""strataslam_tools.live_view writes a Rerun recording (skipped without the optional extra `viewer`)."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from conftest import require_module

from strataslam_tools import live_view


@pytest.fixture(autouse=True)
def _close_the_recording() -> Iterator[None]:
    """Each test closes the recording it opened (its file is flushed and its threads end) before the next one
    opens another: a recording left open hung the next `rr.init` on CI."""
    yield
    try:
        import rerun as rr
    except ImportError:
        return
    if hasattr(rr, "disconnect"):
        rr.disconnect()


def test_floor_colours_are_distinct_and_transitions_apart() -> None:
    assert len({live_view.floor_colour(f) for f in range(10)}) == 10
    assert live_view.floor_colour(-1) == live_view.TRANSIT_COLOUR
    assert live_view.TRANSIT_COLOUR not in live_view.FLOOR_COLOURS
    assert live_view.TRANSIT_COLOUR not in (live_view.PATH_COLOUR, live_view.ROBOT_COLOUR, live_view.SENSOR_COLOUR)
    assert live_view.floor_colour(live_view.LOST) == live_view.LOST_COLOUR != live_view.TRANSIT_COLOUR
    assert live_view.LOST_COLOUR not in live_view.FLOOR_COLOURS
    assert live_view.floor_colour(10) == live_view.floor_colour(0)


def test_height_colours_go_from_blue_low_to_red_high() -> None:
    c = live_view.height_colours(np.array([-1.0, 0.0, 5.0, 10.0, 11.0], dtype=np.float32), 0.0, 10.0)
    assert c.shape == (5, 3) and c.dtype == np.uint8
    assert c[1, 2] > c[1, 0] and c[3, 0] > c[3, 2]  # blue at the bottom, red at the top
    assert (c[0] == c[1]).all() and (c[4] == c[3]).all()  # outside the range: the colour of its end
    assert min(int(v.max()) for v in c) > 150  # no near-black end: they would vanish on the dark background


def test_the_height_colours_follow_the_map_as_it_grows(tmp_path: Path) -> None:
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    view = live_view.LiveView(save=tmp_path / "run.rrd")
    sent: list[tuple[str, int]] = []
    real = view.rr.log
    view.rr.log = lambda path, *a, **kw: (sent.append((path, len(a[0].positions.as_arrow_array()) if path == "world/height"  # type: ignore[method-assign]
                                                       else 0)), real(path, *a, **kw))[1]
    view.floor_cloud(1.0, 0, np.random.default_rng(1).random((300, 3)))
    assert ("world/height", 300) in sent and view._height_range is not None
    first = view._height_range
    view.floor_cloud(2.0, 0, np.random.default_rng(2).random((300, 3)))  # same heights: the same range
    assert view._height_range == first
    sent.clear()
    view.floor_cloud(3.0, 1, np.random.default_rng(3).random((300, 3)) + [0.0, 0.0, 3.0])  # the map grows 3 m up
    assert ("world/height", 600) in sent  # one layer, both floors, coloured on the new range
    assert view._height_range[1] > 3.0


def test_a_run_across_two_floors_is_recorded(tmp_path: Path) -> None:
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    out = tmp_path / "run.rrd"
    view = live_view.LiveView(save=out)
    for i in range(30):
        floor = 0 if i < 10 else (-1 if i < 20 else 1)
        view.pose(100.0 + i, (float(i), 0.0, 0.0 if i < 10 else 0.2 * (i - 10)), floor)
    view.floor_cloud(110.0, 0, np.random.default_rng(1).random((500, 3)))
    view.floor_cloud(130.0, 1, np.random.default_rng(2).random((500, 3)) + [0.0, 0.0, 3.0])
    view.event(115.0, "going up (+0.4 m so far)")
    view.sensor_cloud(116.0, "lidar", np.random.default_rng(3).random((200, 3)))
    view.baro(117.0, 0.5)
    assert not view._camera
    view.camera(118.0, np.zeros((24, 32, 3), dtype=np.uint8), np.full((24, 32), 2.0, dtype=np.float32),
                np.array([[3.0, 4.0], [10.0, 12.0]]), np.array([2, 0], dtype=np.uint8))
    view.camera(119.0, None, None, np.zeros((0, 2)), np.zeros(0, dtype=np.uint8))  # feature frames: no image
    assert view._camera  # the camera views were added to the layout
    assert view._map_lo is not None and view._map_hi is not None  # the live scan is cut to the map's extent
    assert view._map_hi[2] > 3.0
    view.flush_paths()
    # The pole spans every floor mapped so far: from below the lowest point to above the highest.
    assert view._z_bottom is not None and view._z_bottom < 0.1
    assert view._z_top is not None and view._z_top > 3.0
    assert set(view._strips) == {0, -1, 1}
    assert view._height_range is not None and view._height_range[1] > 3.0  # the layer `height` spans both floors
    import rerun as rr

    if hasattr(rr, "disconnect"):
        rr.disconnect()  # flushes the file before it is read
    assert out.stat().st_size > 0


def test_the_optimized_path_replaces_the_drifted_one_and_the_live_tail_continues_it(tmp_path: Path) -> None:
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    view = live_view.LiveView(save=tmp_path / "run.rrd")
    for i in range(5):
        view.pose(float(i), (float(i), 0.5 * i, 0.0), 0)  # drifted odometry
    view.floor_path(0, [[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], [(3.0, 0.0, 0.0), (4.0, 0.0, 0.0)]])  # two visits
    assert view._strips[0] == [[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], [(3.0, 0.0, 0.0), (4.0, 0.0, 0.0)], [(4.0, 0.0, 0.0)]]
    view.pose(5.0, (5.0, 0.0, 0.0), 0)
    assert view._strips[0][-1] == [(4.0, 0.0, 0.0), (5.0, 0.0, 0.0)]  # the live tail starts at the last keyframe
    view.floor_path(1, [])  # nothing to draw: nothing changes
    assert 1 not in view._strips


def test_a_transition_that_joined_a_floor_is_not_drawn_twice(tmp_path: Path) -> None:
    """The poses of an open transition are drawn in the transition colour; once its keyframes are a floor's, the floor's path covers
    that time and the grey trail goes, except what no floor took (a cabin) and the transition still open."""
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    view = live_view.LiveView(save=tmp_path / "run.rrd")
    for i in range(10):
        view.pose(float(i), (float(i), 0.0, 0.0), 0 if i < 2 else -1)  # a transition from t=2 s
    view.floor_path(0, [[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)]], [(0.0, 5.0)])  # it ended on floor 0 up to t=5 s
    assert view._strips[-1] == [[(6.0, 0.0, 0.0), (7.0, 0.0, 0.0), (8.0, 0.0, 0.0), (9.0, 0.0, 0.0)]]
    view.pose(10.0, (10.0, 0.0, 0.0), -1)  # still in a transition: its piece goes on
    assert view._strips[-1][-1][-1] == (10.0, 0.0, 0.0)
    view.floor_path(1, [[(7.0, 0.0, 3.0), (10.0, 0.0, 3.0)]], [(7.0, 10.0)])  # the rest joined floor 1
    assert view._strips[-1] == [[(6.0, 0.0, 0.0)], []]  # 6 s, between the two floors' keyframes; the live piece
    view.pose(11.0, (11.0, 0.0, 3.0), 1)
    view.flush_paths()


def test_an_elevator_ride_the_odometry_did_not_see_is_drawn_between_its_floors(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """In the cabin the odometry stays at the floor it left; its height comes at once with the next floor. The two
    pieces are joined, in the transition colour, where the floor changed: the ride is that join."""
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    view = live_view.LiveView(save=tmp_path / "run.rrd")
    drawn: dict[str, tuple[list, object]] = {}
    real = view.rr.LineStrips3D

    def strips(pieces, **kw):  # type: ignore[no-untyped-def]
        drawn["last"] = (pieces, kw.get("colors"))
        return real(pieces, **kw)

    monkeypatch.setattr(view.rr, "LineStrips3D", strips)
    logged: dict[str, tuple[list, object]] = {}
    real_log = view.rr.log
    monkeypatch.setattr(view.rr, "log", lambda path, *a, **kw: (logged.__setitem__(path, drawn.get("last")),
                                                                real_log(path, *a, **kw))[1])
    for i in range(4):
        view.pose(float(i), (float(i), 0.0, 0.0), 0)
    for i in range(4, 8):
        view.pose(float(i), (4.0, 0.0, 0.0), -1)  # the ride: the odometry does not climb
    for i in range(8, 11):
        view.pose(float(i), (4.0 + i - 8, 0.0, 3.0), 1)  # on floor 1, 3 m up
    view.flush_paths()
    pieces, colours = logged["world/path/transition"]
    assert [(4.0, 0.0, 0.0), (4.0, 0.0, 3.0)] in pieces  # the ride, vertical, from the cabin to floor 1
    assert [(3.0, 0.0, 0.0), (4.0, 0.0, 0.0)] in pieces  # floor 0 to the cabin
    assert colours == [live_view.TRANSIT_COLOUR]
    assert logged["world/path/floor_1"][1] == [live_view.PATH_COLOUR]
    # Down again: the floor state comes before the odometry's jump, so the first pose on floor 0 is still 3 m up. The
    # join meets floor 0's optimized path instead, at its keyframe nearest in time.
    for i in range(11, 14):
        view.pose(float(i), (6.0, 0.0, 3.0), -1)
    view.pose(14.0, (6.0, 0.0, 3.0), 0)
    view.pose(15.0, (6.0, 0.0, 0.0), 0)
    view.floor_path(0, [[(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)], [(6.0, 0.0, 0.0), (6.5, 0.0, 0.0)]], [(0.0, 3.0), (14.5, 15.0)],
                    [[0.0, 3.0], [14.5, 15.0]])
    view.flush_paths()
    pieces, _ = logged["world/path/transition"]
    assert [(6.0, 0.0, 3.0), (6.0, 0.0, 0.0)] in pieces  # the ride down reaches floor 0
    assert [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)] in logged["world/path/floor_0"][0]


def test_a_ride_whose_height_came_within_the_transition_is_joined_from_where_its_trail_ends(tmp_path: Path) -> None:
    """Elevator-VIGS Office4: the odometry took the ride's 14.7 m while the transition was still open, so its last poses
    are already up, and floor 1's path takes them over. The join goes from the trail as drawn (in the cabin) to floor 1:
    from the last pose (up there) it was a join of zero length and the ride was not drawn."""
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    view = live_view.LiveView(save=tmp_path / "run.rrd")
    view.pose(0.0, (0.0, 0.0, 0.0), 0)
    for i in range(1, 5):
        view.pose(float(i), (1.0, 0.0, 0.0), -1)  # in the cabin
    view.pose(5.0, (1.0, 0.0, 14.7), -1)  # the ride's height, still in the transition
    view.pose(6.0, (1.5, 0.0, 14.7), 1)
    view.floor_path(1, [[(1.0, 0.0, 14.7), (1.5, 0.0, 14.7)]], [(5.0, 6.0)], [[5.0, 6.0]])
    joins = [[view._path_point(a, t, p), view._path_point(b, t, q)] for t, a, b, p, q in view._joins]
    assert [(1.0, 0.0, 0.0), (1.5, 0.0, 14.7)] in joins  # the cabin to floor 1 where the floor changed
    # A trail a floor's path took is not met far from its time: the join keeps the pose drawn then.
    view.pose(30.0, (1.5, 0.0, 14.7), -1)
    assert view._path_point(-1, 17.0, (9.0, 9.0, 9.0)) == (9.0, 9.0, 9.0)  # 13 s from both


def test_lost_tracking_is_drawn_apart_from_the_transitions(tmp_path: Path) -> None:
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    view = live_view.LiveView(save=tmp_path / "run.rrd")
    view.pose(0.0, (0.0, 0.0, 0.0), 0)
    view.pose(1.0, (1.0, 0.0, 0.0), live_view.LOST)
    view.pose(2.0, (2.0, 0.0, 0.0), live_view.LOST)
    view.pose(3.0, (3.0, 0.0, 0.0), -1)
    assert view._strips[live_view.LOST] == [[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]]
    assert view._strips[-1] == [[(3.0, 0.0, 0.0)]] and view._transit_t == [[3.0]]
    view.flush_paths()


def test_where_floors_join_is_marked_and_labelled(tmp_path: Path) -> None:
    """An elevator the odometry did not see: departure at the foot of the ride, arrival on the floor reached, labelled
    with what the node said. Stairs: departure where the robot left floor 1, even with a piece of trail at the
    bottom."""
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    view = live_view.LiveView(save=tmp_path / "run.rrd")
    view.pose(0.0, (0.0, 0.0, 0.0), 0)
    for i in range(1, 4):
        view.pose(float(i), (1.0, 0.0, 0.0), -1)  # the cabin
    view.pose(4.0, (1.0, 0.0, 3.0), 1)
    view.transition(4.0, "elevator", 0, 1, 3.0)
    view.pose(5.0, (2.0, 0.0, 3.0), 1)
    for i in range(6, 10):
        view.pose(float(i), (2.0 + i - 6, 0.0, 3.0 - 0.75 * (i - 6)), -1)  # stairs down
    view.pose(10.0, (6.0, 0.0, 0.1), -1)  # Office4: still a transition a moment on floor 0
    view.pose(11.0, (6.0, 0.0, 0.1), -1)
    view.pose(12.0, (6.0, 0.0, 0.0), 0)
    view.transition(12.0, "stairs", 1, 0, -3.0)
    view.transition(13.0, "unknown", 0, 0, 0.0)  # not a floor change: nothing marked
    arrivals, labels, departures = view._transition_marks()
    assert arrivals == [(1.0, 0.0, 3.0), (6.0, 0.0, 0.0)]
    assert departures == [(1.0, 0.0, 0.0), (2.0, 0.0, 3.0)]
    assert labels == ["elevator 0\u21921 +3.00 m", "stairs 1\u21920 -3.00 m"]
    view.flush_paths()


def test_a_trail_cut_by_a_floors_path_is_joined_to_it(tmp_path: Path) -> None:
    """Office4's stairs down: the lower half joined floor 0, whose loop closures moved it 2 m; the upper half stays the
    trail the odometry left. They are joined at the cut, so the path goes on and the step shows."""
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    view = live_view.LiveView(save=tmp_path / "run.rrd")
    view.pose(0.0, (0.0, 0.0, 3.0), 1)
    for i in range(1, 11):
        view.pose(float(i), (float(i), 0.0, 3.0 - 0.3 * i), -1)  # stairs down
    view.pose(11.0, (11.0, 0.0, 0.0), 0)
    # Floor 0's path took the lower half (t 6-11 s), placed 2 m aside by its loop closures.
    view.floor_path(0, [[(6.0, 2.0, 1.2), (8.0, 2.0, 0.6), (11.0, 2.0, 0.0)]], [(6.0, 11.0)], [[6.0, 8.0, 11.0]])
    assert view._transit_t[0][-1] == 5.0  # the trail ends where the floor's path begins (with its margin)
    assert [(5.0, 0.0, 1.5), (6.0, 2.0, 1.2)] in view._cut_joins()
    view.flush_paths()


def test_a_large_floor_is_thinned_for_the_3d_views(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    monkeypatch.setattr(live_view, "WORLD_MAX_POINTS", 1000)
    view = live_view.LiveView(save=tmp_path / "run.rrd")
    sizes: list[int] = []
    real = view.rr.Points3D
    monkeypatch.setattr(view.rr, "Points3D", lambda p, **kw: sizes.append(len(p)) or real(p, **kw))
    view.floor_cloud(1.0, 0, np.random.default_rng(1).random((10_000, 3)))
    assert sizes and max(sizes) <= 1000  # both layers


def test_the_camera_colours_are_a_layer_of_the_same_view(tmp_path: Path) -> None:
    """One 3D view, one camera: the camera's colours are its layer `colour` (every floor in one entity, hidden at
    the start), not another tab with a camera of its own."""
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    view = live_view.LiveView(save=tmp_path / "run.rrd")
    sent: list[int] = []
    view._send_blueprint = lambda camera: sent.append(-1)  # type: ignore[method-assign]
    real = view._log_colour_layer
    view._log_colour_layer = lambda: (sent.append(len(view._floor_colours)), real())[1]  # type: ignore[method-assign]
    rng = np.random.default_rng(1)
    view.floor_cloud(1.0, 0, rng.random((300, 3)))  # no colours: no layer
    assert not view._colour and sent == []
    view.floor_cloud(2.0, 0, rng.random((300, 3)), rng.integers(0, 255, (300, 3), dtype=np.uint8))
    view.floor_cloud(3.0, 1, rng.random((200, 3)), rng.integers(0, 255, (200, 3), dtype=np.uint8))
    assert view._colour and sent == [1, 2]  # the layout is not sent again (it would reset the camera)
    view.floor_cloud(4.0, 1, rng.random((300, 3)), rng.integers(0, 255, (299, 3), dtype=np.uint8))  # mismatched
    # The colours are not taken; floor 1's old ones no longer fit its points, so the layer goes without it.
    assert len(view._floor_colours[1]) == 200 and sent == [1, 2, 2]
