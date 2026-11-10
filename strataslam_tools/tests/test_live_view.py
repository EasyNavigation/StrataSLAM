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


def test_floor_colours_are_distinct_and_transitions_grey() -> None:
    assert len({live_view.floor_colour(f) for f in range(10)}) == 10
    assert live_view.floor_colour(-1) == live_view.TRANSIT_COLOUR
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
    sent: list[str] = []
    view._log_height_cloud = lambda floor, p: sent.append(f"floor_{floor}")  # type: ignore[method-assign]
    view.floor_cloud(1.0, 0, np.random.default_rng(1).random((300, 3)))
    assert sent == ["floor_0"] and view._height_range is not None
    view.floor_cloud(2.0, 0, np.random.default_rng(2).random((300, 3)))  # same heights: only this floor again
    assert sent == ["floor_0", "floor_0"]
    view.floor_cloud(3.0, 1, np.random.default_rng(3).random((300, 3)) + [0.0, 0.0, 3.0])  # the map grows 3 m up
    assert sorted(sent[2:]) == ["floor_0", "floor_1"]  # every floor coloured again on the new range
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
    assert view._height_range is not None and view._height_range[1] > 3.0  # the tab "3D: height" spans both floors
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


def test_a_large_floor_is_thinned_for_the_3d_views(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    monkeypatch.setattr(live_view, "WORLD_MAX_POINTS", 1000)
    view = live_view.LiveView(save=tmp_path / "run.rrd")
    sizes: list[int] = []
    real = view.rr.Points3D
    monkeypatch.setattr(view.rr, "Points3D", lambda p, **kw: sizes.append(len(p)) or real(p, **kw))
    view.floor_cloud(1.0, 0, np.random.default_rng(1).random((10_000, 3)))
    assert sizes and max(sizes) <= 1000  # both tabs


def test_the_camera_colours_get_their_own_tab_when_a_cloud_has_them(tmp_path: Path) -> None:
    require_module("rerun", "pip install rerun-sdk, the optional extra `viewer`")
    view = live_view.LiveView(save=tmp_path / "run.rrd")
    sent: list[tuple[str, int]] = []
    view._log_colour_cloud = lambda floor, p, c: sent.append((f"floor_{floor}", len(c)))  # type: ignore[method-assign]
    rng = np.random.default_rng(1)
    view.floor_cloud(1.0, 0, rng.random((300, 3)))  # no colours: no tab
    assert not view._colour and sent == []
    view.floor_cloud(2.0, 0, rng.random((300, 3)), rng.integers(0, 255, (300, 3), dtype=np.uint8))
    assert view._colour and sent == [("floor_0", 300)]
    import rerun.blueprint as rrb
    assert [v.name for v in view._map_tabs(rrb)] == ["3D: floors", "3D: height", "3D: colour"]
    view.floor_cloud(3.0, 0, rng.random((300, 3)), rng.integers(0, 255, (299, 3), dtype=np.uint8))  # mismatched
    assert sent == [("floor_0", 300)]
