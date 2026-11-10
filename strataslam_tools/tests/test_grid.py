"""The occupancy grid of `strataslam export --format grid` (nav2 map_server's PGM + YAML)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from strataslam_tools import cli, grid, mapio


def room_map(root: Path, cloud_frame: str = "base") -> Path:
    """A floor of 2 keyframes in a room of 4 x 4 m centred on the origin, 1 m apart along x, the second turned 90 deg.
    Each keyframe's cloud (in its base frame): the walls at 0.3 m above the base, a lamp at 2.5 m (above the robot)
    and floor points at 0.0 m."""
    d = root / "floor_000"
    (d / "clouds").mkdir(parents=True)
    (root / "manifest.yaml").write_text(
        "strataslam_version: 0.2.0\nformat_version: 1\nprofile: profile.yaml\ntransitions: transitions.yaml\n"
        "floors:\n  - {id: 0, dir: floor_000, keyframes: 2}\n")
    (root / "profile.yaml").write_text("strataslam: {version: 1}\n")
    (root / "transitions.yaml").write_text("transitions: []\n")
    (d / "metadata.yaml").write_text(f"floor_id: 0\ncloud_frame: {cloud_frame}\n")
    s = np.sqrt(0.5)
    poses = [(0, -0.5, 0.0, 0.0, 1.0), (1, 0.5, 0.0, s, s)]  # id, x, y, qz, qw
    (d / "keyframes.csv").write_text("id,stamp_ns,session,x,y,z,qx,qy,qz,qw\n" + "".join(
        f"{i},{(i + 1) * 10**9},0,{x},{y},0,0,0,{qz},{qw}\n" for i, x, y, qz, qw in poses))
    (d / "edges.csv").write_text("type,from,to,measurement,info_dim,information\n")
    u = np.linspace(-2.0, 2.0, 81)
    walls = np.vstack([np.c_[u, np.full_like(u, -2.0)], np.c_[u, np.full_like(u, 2.0)],
                       np.c_[np.full_like(u, -2.0), u], np.c_[np.full_like(u, 2.0), u]])
    for i, x, y, qz, qw in poses:
        yaw = 2 * np.arctan2(qz, qw)
        c, sn = np.cos(yaw), np.sin(yaw)
        local = (walls - [x, y]) @ np.array([[c, -sn], [sn, c]])  # world -> this keyframe's base
        pts = np.vstack([np.c_[local, np.full(len(local), 0.3)],
                         [[0.5, 0.5, 2.5]],  # a lamp above the robot
                         np.c_[local[::4] * 0.5, np.zeros(len(local[::4]))]])  # the floor
        mapio.write_pcd(d / "clouds" / f"{i}.pcd", pts)
    return root


def pixel(y: dict, img: np.ndarray, x: float, yy: float) -> int:
    r, (ox, oy) = y["resolution"], y["origin"][:2]
    col, row = int((x - ox) / r), int((yy - oy) / r)
    return int(img[img.shape[0] - 1 - row, col])


def read_pgm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    header = [t for t in data.split(b"\n", 4)[:4] if not t.startswith(b"#")]
    assert header[0] == b"P5" and header[2] == b"255"
    w, h = map(int, header[1].split())
    return np.frombuffer(data[-w * h:], np.uint8).reshape(h, w)


def test_the_walls_are_occupied_the_room_free_and_the_rest_unknown(tmp_path: Path) -> None:
    root = room_map(tmp_path / "m")
    assert cli.main(["export", str(root), "--format", "grid"]) == 0
    y = yaml.safe_load((root / "export" / "floor_000.yaml").read_text())
    assert y["image"] == "floor_000.pgm" and y["mode"] == "trinary" and y["resolution"] == 0.05
    img = read_pgm(root / "export" / "floor_000.pgm")
    assert pixel(y, img, 2.0, 0.3) == 0  # a wall, at the height of the band, seen from both keyframes
    assert pixel(y, img, -2.0, -1.0) == 0
    assert pixel(y, img, 0.0, 0.0) == 254  # inside the room: crossed by the rays
    assert pixel(y, img, -1.3, 1.1) == 254
    assert pixel(y, img, 2.3, 0.0) == grid.UNKNOWN  # behind the wall, within the margin: never seen
    # The lamp (2.5 m) and the floor (0 m) are outside the band of 0.05-2.0 m: no obstacle there.
    assert pixel(y, img, 0.0, 1.0) == 254


def test_the_height_band_decides_what_is_an_obstacle(tmp_path: Path) -> None:
    root = room_map(tmp_path / "m")
    # A band above the walls' points (0.3 m) and below the lamp: nothing to map.
    assert cli.main(["export", str(root), "--format", "grid", "--min-height", "0.5", "--max-height", "2.0"]) == 1
    # A band with the lamp: an obstacle where it hangs (keyframe 0 at (-0.5, 0), lamp 0.5, 0.5 ahead-left).
    assert cli.main(["export", str(root), "--format", "grid", "--max-height", "3.0"]) == 0
    y = yaml.safe_load((root / "export" / "floor_000.yaml").read_text())
    assert pixel(y, read_pgm(root / "export" / "floor_000.pgm"), 0.0, 0.5) < 128


def test_a_cell_crossed_more_than_hit_is_free() -> None:
    # Something seen once at (1, 0) and then seen through four times (a person who walked away): free.
    def ray(x: float, h: float, h0: float | None = None) -> grid.Rays:  # from the origin along x, at height h
        return grid.Rays(np.zeros((1, 2)), np.array([h if h0 is None else h0]), np.array([[x, 0.0]]), np.array([h]))

    rays = [ray(1.0, 0.3)] + [ray(2.0, 0.3)] * 4
    g = grid.build_grid(rays, 0.05, 2.0, resolution=0.1)
    occ = g.occupancy()
    i, j = int((0.0 - g.origin[1]) / 0.1), int((1.0 - g.origin[0]) / 0.1)
    assert occ[i, j] == pytest.approx(1 / 5)
    assert g.image()[g.shape[0] - 1 - i, j] > 254 * (1 - grid.FREE_THRESH)
    # Seen through a layer higher: it stays an obstacle in its own layer.
    g = grid.build_grid([ray(1.0, 0.3)] + [ray(2.0, 0.6)] * 4, 0.05, 2.0, resolution=0.1)
    assert g.occupancy()[i, j] == pytest.approx(1.0)
    # A table top at 0.8 m seen by a camera at 1.1 m, then rays from the camera to a wall behind it at 0.75 m: over
    # the table they are above 0.8 m (another layer), and the table stays an obstacle.
    g = grid.build_grid([ray(1.0, 0.8, 1.1)] + [ray(2.0, 0.75, 1.1)] * 4, 0.05, 2.0, resolution=0.1)
    assert g.occupancy()[i, j] == pytest.approx(1.0)


def test_a_map_whose_clouds_are_in_the_sensor_frame_is_refused_with_why(tmp_path: Path,
                                                                        capsys: pytest.CaptureFixture[str]) -> None:
    root = room_map(tmp_path / "m", cloud_frame="sensor")
    assert cli.main(["export", str(root), "--format", "grid"]) != 0
    assert "sensor's frame" in capsys.readouterr().err


def test_the_depth_clouds_of_a_planar_scanner_with_a_camera_add_their_obstacles(tmp_path: Path) -> None:
    # A table top 0.8 m high that the scan (0.3 m) passes under: only the camera's depth cloud sees it (ADR-0068).
    root = room_map(tmp_path / "m")
    (root / "floor_000" / "depth_clouds").mkdir()
    u = np.linspace(0.6, 1.0, 9)
    top = np.array([[x, y, 0.8] for x in u for y in np.linspace(-0.2, 0.2, 9)])  # keyframe 0's base: (-0.5, 0)
    mapio.write_pcd(root / "floor_000" / "depth_clouds" / "0.pcd", top)
    assert cli.main(["export", str(root), "--format", "grid"]) == 0
    y = yaml.safe_load((root / "export" / "floor_000.yaml").read_text())
    img = read_pgm(root / "export" / "floor_000.pgm")
    assert pixel(y, img, 0.3, 0.0) == 0  # the table, at (0.3, 0) in the floor
