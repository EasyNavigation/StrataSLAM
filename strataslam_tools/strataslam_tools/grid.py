"""Occupancy grid of a floor (nav2 map_server's format: a PGM image and its YAML), from the keyframes and their clouds.

A cell is occupied where the keyframes' points (their clouds and, with a planar scanner and a camera, their depth
clouds, ADR-0068) lie within a band of heights above the robot's base (the obstacles a ground robot meets: from just
above the floor to its own height), and free where the rays from the sensor to the points cross it within the band.
The band is cut into layers of SLAB m and a ray clears, at each step, the layer it is at: from a camera at 1.1 m to a
wall at 0.75 m the ray passes over a table at 0.8 m without clearing it, and a scan at 0.3 m passes under it. Rays to
points outside the band (the floor, the ceiling) clear the layers they cross too. In each layer a cell is as dark as
the fraction of its observations that hit it, so that a person who walked by leaves it free; the cell is the darkest
of its layers, and unknown where no layer was observed.

The rays start at the sensor, whose position in the base frame the map keeps (`cloud_origin`, `depth_origin` in the
floor's metadata). A map written before them has its rays start at the base, horizontal at each point's height: right
for a planar scanner, an approximation for a 3D sensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from . import mapio

Array = NDArray[np.float64]

UNKNOWN = 205  # map_server's trinary value for unknown
OCCUPIED_THRESH = 0.65
FREE_THRESH = 0.25
SLAB = 0.10  # m: the layers of the height band


@dataclass
class Rays:
    """One keyframe's rays in the floor frame, heights above that keyframe's base."""

    origin_xy: Array  # [n, 2] where each ray starts
    origin_h: Array  # [n]
    xy: Array  # [n, 2] its point
    h: Array  # [n]


@dataclass
class Grid:
    hits: NDArray[np.uint16]  # [layers, rows, cols], row 0 at the lowest y; observations per keyframe
    passes: NDArray[np.uint16]
    origin: tuple[float, float]  # x, y of the lower-left corner of cell (0, 0)
    resolution: float

    @property
    def shape(self) -> tuple[int, int]:
        """rows, cols"""
        return int(self.hits.shape[1]), int(self.hits.shape[2])

    def occupancy(self) -> NDArray[np.float64]:
        """For each cell, the largest over its layers of the fraction of their observations that hit it; NaN where
        no layer was observed."""
        out = np.full(self.shape, np.nan)
        for h, f in zip(self.hits, self.passes):  # layer by layer: one layer's floats at a time
            seen = h.astype(np.float64) + f
            with np.errstate(invalid="ignore", divide="ignore"):
                p = np.where(seen > 0, h / np.maximum(seen, 1), np.nan)
            out = np.fmax(out, p)  # NaN where neither was observed
        return out

    def image(self) -> NDArray[np.uint8]:
        """The PGM's pixels (top row first): 0 occupied ... 254 free, 205 unknown."""
        p = self.occupancy()
        img = np.where(np.isnan(p), UNKNOWN, np.round((1.0 - np.nan_to_num(p)) * 254.0)).astype(np.uint8)
        return img[::-1]


def quat_to_rot(q: Array) -> Array:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def floor_rays(floor_dir: Path) -> list[Rays]:
    """The rays of every keyframe with a cloud, from its clouds (`clouds/`) and depth clouds (`depth_clouds/`)."""
    meta = yaml.safe_load((floor_dir / "metadata.yaml").read_text()) or {}
    if meta.get("cloud_frame", "sensor") != "base":
        raise mapio.MapFormatError(
            f"{floor_dir}: its keyframe clouds are in the sensor's frame (a map written before cloud_frame, "
            "ADR-0062), whose pose the map does not keep: no grid can be made from it")
    sources = [("clouds", meta.get("cloud_origin")), ("depth_clouds", meta.get("depth_origin"))]
    out = []
    for row in mapio.read_keyframes(floor_dir):
        R, t = quat_to_rot(row[6:10]), row[3:6]
        o_xy, o_h, xy, h = [], [], [], []
        for folder, sensor in sources:
            path = floor_dir / folder / f"{int(row[0])}.pcd"
            if not path.exists():
                continue
            pts, _ = mapio.read_pcd(path)
            if len(pts) == 0:
                continue
            w = pts @ R.T + t
            hh = w[:, 2] - t[2]
            if sensor is not None:  # from the sensor
                s = R @ np.asarray(sensor, dtype=float) + t
                o_xy.append(np.repeat(s[None, :2], len(w), axis=0))
                o_h.append(np.full(len(w), s[2] - t[2]))
            else:  # an older map: from the base, horizontal
                o_xy.append(np.repeat(t[None, :2], len(w), axis=0))
                o_h.append(hh)
            xy.append(w[:, :2])
            h.append(hh)
        if xy:
            out.append(Rays(np.vstack(o_xy), np.concatenate(o_h), np.vstack(xy), np.concatenate(h)))
    return out


def build_grid(rays: list[Rays], min_height: float, max_height: float, resolution: float = 0.05,
               margin: float = 0.5) -> Grid:
    in_band = [(r.h >= min_height) & (r.h <= max_height) for r in rays]
    if not any(m.any() for m in in_band):
        raise ValueError("no points within the height band: nothing to map")
    allxy = np.vstack([np.vstack([r.origin_xy, r.xy[m]]) for r, m in zip(rays, in_band) if m.any()])
    lo = allxy.min(axis=0) - margin
    hi = allxy.max(axis=0) + margin
    cols, rows = (np.ceil((hi - lo) / resolution).astype(int) + 1).tolist()
    layers = max(1, int(np.ceil((max_height - min_height) / SLAB)))
    n = rows * cols
    hits = np.zeros(layers * n, dtype=np.uint16)
    passes = np.zeros(layers * n, dtype=np.uint16)

    def cells(xy: Array, h: Array) -> NDArray[np.int64]:
        """Flat index (layer, row, col); -1 outside the grid or the band."""
        ij = np.floor((xy - lo) / resolution).astype(np.int64)
        layer = np.minimum(np.floor((h - min_height) / SLAB).astype(np.int64), layers - 1)
        ok = (ij >= 0).all(axis=1) & (ij[:, 0] < cols) & (ij[:, 1] < rows) & (h >= min_height) & (h <= max_height)
        return np.where(ok, layer * n + ij[:, 1] * cols + ij[:, 0], -1)

    def bump(a: NDArray[np.uint16], idx: NDArray[np.int64]) -> None:  # idx unique: one count each
        a[idx] = np.minimum(a[idx].astype(np.int64) + 1, np.iinfo(np.uint16).max).astype(np.uint16)

    step = 0.5 * resolution
    for r, band in zip(rays, in_band):
        # Every ray sampled every half cell (in plan), up to (not into) the cell of its point, at the height it has
        # there: the free cells it crosses, each in its layer.
        d = r.xy - r.origin_xy
        length = np.linalg.norm(d, axis=1)
        ns = np.maximum(0, np.floor(length / step).astype(np.int64) - 1)
        ray = np.repeat(np.arange(len(r.xy)), ns)
        k = np.arange(ns.sum()) - np.repeat(np.cumsum(ns) - ns, ns)
        frac = (k + 1) * step / np.maximum(length[ray], 1e-9)
        crossed = cells(r.origin_xy[ray] + d[ray] * frac[:, None],
                        r.origin_h[ray] + (r.h[ray] - r.origin_h[ray]) * frac)
        end = cells(r.xy[band], r.h[band])
        # A cell counts once per keyframe, not once per ray; one that this keyframe saw as an obstacle is not also
        # free for it (rays grazing a wall cross the cells of its other points).
        hit = np.unique(end[end >= 0])
        free = np.setdiff1d(np.unique(crossed[crossed >= 0]), hit, assume_unique=True)
        bump(passes, free)
        bump(hits, hit)
    return Grid(hits.reshape(layers, rows, cols), passes.reshape(layers, rows, cols),
                (float(lo[0]), float(lo[1])), resolution)


def write_grid(grid: Grid, pgm: Path) -> Path:
    """`<name>.pgm` and `<name>.yaml`, as nav2's map_server loads them (trinary)."""
    img = grid.image()
    with open(pgm, "wb") as f:
        f.write(f"P5\n# StrataSLAM occupancy grid\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(img.tobytes())
    meta = {
        "image": pgm.name,
        "mode": "trinary",
        "resolution": grid.resolution,
        "origin": [grid.origin[0], grid.origin[1], 0.0],
        "negate": 0,
        "occupied_thresh": OCCUPIED_THRESH,
        "free_thresh": FREE_THRESH,
    }
    out = pgm.with_suffix(".yaml")
    out.write_text(yaml.safe_dump(meta, sort_keys=False))
    return out
