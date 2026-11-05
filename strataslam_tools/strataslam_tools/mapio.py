"""Reading the on-disk map format (docs/data_format.md) and its point clouds, without strataslam installed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

SUPPORTED_FORMAT_VERSION = 1
Array = NDArray[np.float64]


class MapFormatError(Exception):
    pass


def read_manifest(map_dir: str | Path) -> dict[str, Any]:
    path = Path(map_dir) / "manifest.yaml"
    if not path.exists():
        raise MapFormatError(f"{path}: missing (is this a complete map directory?)")
    m: dict[str, Any] = yaml.safe_load(path.read_text())
    if m.get("format_version") != SUPPORTED_FORMAT_VERSION:
        raise MapFormatError(
            f"unsupported map format version {m.get('format_version')} (supported: {SUPPORTED_FORMAT_VERSION})"
        )
    return m


def validate_map(map_dir: str | Path) -> list[str]:
    """Problems that make a map directory invalid (empty list = valid)."""
    problems: list[str] = []
    try:
        m = read_manifest(map_dir)
    except MapFormatError as e:
        return [str(e)]
    root = Path(map_dir)
    for f in m.get("floors", []):
        d = root / f["dir"]
        for name in ("metadata.yaml", "keyframes.csv", "edges.csv"):
            if not (d / name).exists():
                problems.append(f"{d / name}: missing")
    for name in ("profile.yaml", "transitions.yaml"):
        if not (root / name).exists():
            problems.append(f"{root / name}: missing")
    return problems


def read_keyframes(floor_dir: str | Path) -> Array:
    """Columns: id, stamp_ns, session, x, y, z, qx, qy, qz, qw."""
    rows = np.loadtxt(Path(floor_dir) / "keyframes.csv", delimiter=",", skiprows=1, ndmin=2)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 10)


def read_edges(floor_dir: str | Path) -> list[dict[str, Any]]:
    out = []
    for line in (Path(floor_dir) / "edges.csv").read_text().splitlines()[1:]:
        if not line.strip():
            continue
        typ, a, b, meas, dim, info = line.split(",")
        out.append({"type": typ, "from": int(a), "to": int(b), "measurement": [float(x) for x in meas.split(";") if x],
                    "info_dim": int(dim), "information": [float(x) for x in info.split(";") if x]})
    return out


def read_fused(map_dir: str | Path) -> dict[int, tuple[float, Array]]:
    """T_map_floor per floor as (theta, t), if the map was fused."""
    path = Path(map_dir) / "fused" / "floor_transforms.yaml"
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text())
    return {int(f["floor"]): (float(f["theta"]), np.asarray(f["t"], dtype=np.float64)) for f in doc["floors"]}


def apply_pose4(theta: float, t: Array, pts: Array) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    out: Array = pts @ R.T + t
    return out


# ------------------------------------------------------------------ point clouds

def read_pcd(path: str | Path) -> tuple[Array, NDArray[np.uint8] | None]:
    raw = Path(path).read_bytes()
    end = raw.index(b"DATA binary\n") + len(b"DATA binary\n")
    header = raw[:end].decode().splitlines()
    n = int(next(h.split()[1] for h in header if h.startswith("POINTS")))
    fields = next(h.split()[1:] for h in header if h.startswith("FIELDS"))
    has_rgb = "rgb" in fields
    step = 16 if has_rgb else 12
    buf = raw[end:end + n * step]
    if len(buf) != n * step:
        raise MapFormatError(f"{path}: truncated PCD data")
    if not has_rgb:
        return np.frombuffer(buf, dtype="<f4").reshape(n, 3).astype(np.float64), None
    arr = np.frombuffer(buf, dtype=np.dtype([("xyz", "<f4", 3), ("rgb", "<u4")]))
    rgb = arr["rgb"]
    colors = np.stack([(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255], axis=1).astype(np.uint8)
    return arr["xyz"].astype(np.float64), colors


def read_ply(path: str | Path) -> tuple[Array, NDArray[np.uint8] | None]:
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode().splitlines()
    n = int(next(h.split()[2] for h in header if h.startswith("element vertex")))
    has_rgb = any(h.startswith("property uchar red") for h in header)
    if has_rgb:
        arr = np.frombuffer(raw[end:end + n * 15], dtype=np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)]))
        return arr["xyz"].astype(np.float64), np.array(arr["rgb"])
    xyz: Array = np.frombuffer(raw[end:end + n * 12], dtype="<f4").reshape(n, 3).astype(np.float64)
    return xyz, None


def write_pcd(path: str | Path, pts: Array, colors: NDArray[np.uint8] | None = None) -> None:
    n = len(pts)
    head = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
            + ("FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n" if colors is not None else "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
            + f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    if colors is None:
        body = pts.astype("<f4").tobytes()
    else:
        packed = (colors[:, 0].astype("<u4") << 16) | (colors[:, 1].astype("<u4") << 8) | colors[:, 2].astype("<u4")
        arr = np.empty(n, dtype=np.dtype([("xyz", "<f4", 3), ("rgb", "<u4")]))
        arr["xyz"], arr["rgb"] = pts.astype("<f4"), packed
        body = arr.tobytes()
    Path(path).write_bytes(head.encode() + body)


def write_ply(path: str | Path, pts: Array, colors: NDArray[np.uint8] | None = None) -> None:
    n = len(pts)
    head = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
            + ("property uchar red\nproperty uchar green\nproperty uchar blue\n" if colors is not None else "") + "end_header\n")
    if colors is None:
        body = pts.astype("<f4").tobytes()
    else:
        arr = np.empty(n, dtype=np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)]))
        arr["xyz"], arr["rgb"] = pts.astype("<f4"), colors
        body = arr.tobytes()
    Path(path).write_bytes(head.encode() + body)


