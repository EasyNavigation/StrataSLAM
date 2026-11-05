"""strataslam_tools.live_view writes a Rerun recording (skipped without the optional extra `viewer`)."""

from pathlib import Path

import numpy as np
from conftest import require_module

from strataslam_tools import live_view


def test_floor_colours_are_distinct_and_transitions_grey() -> None:
    assert len({live_view.floor_colour(f) for f in range(10)}) == 10
    assert live_view.floor_colour(-1) == live_view.TRANSIT_COLOUR
    assert live_view.floor_colour(10) == live_view.floor_colour(0)


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
    import rerun as rr

    if hasattr(rr, "disconnect"):
        rr.disconnect()
    assert out.stat().st_size > 0
