"""Trajectory and map metrics: ATE, RPE, floor collapse, floor height and 4-DoF fusion error.

Mirrors strataslam_core/evaluation/metrics.cpp; a pytest cross-checks both implementations.
Trajectories are arrays ``[t, x, y, z, qx, qy, qz, qw]`` (TUM format).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
AlignMode = Literal["none", "se3", "4dof"]


def read_tum(path: str | Path) -> Array:
    rows = [
        [float(x) for x in line.split()]
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def quat_to_rot(q: Array) -> Array:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_vec(R: Array) -> Array:
    """Rotation vector (log map) of a rotation matrix."""
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    angle = math.atan2(float(np.linalg.norm(vee)), float((np.trace(R) - 1) / 2))  # stable near 0
    if angle < 1e-9:
        return vee
    if math.pi - angle < 1e-6:
        w, v = np.linalg.eigh((R + R.T) / 2)
        axis = v[:, np.argmax(w)]
        return axis * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(angle))
    return axis * angle


def _slerp(q0: Array, q1: Array, f: float) -> Array:
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = min(1.0, float(np.dot(q0, q1)))
    if dot > 1 - 1e-12:
        q = q0 + f * (q1 - q0)
        out: Array = q / np.linalg.norm(q)
        return out
    theta = math.acos(dot)
    return (math.sin((1 - f) * theta) * q0 + math.sin(f * theta) * q1) / math.sin(theta)


def interpolate(traj: Array, t: float) -> tuple[Array, Array] | None:
    """Position and rotation matrix at time ``t`` (None outside the trajectory)."""
    ts = traj[:, 0]
    if t < ts[0] or t > ts[-1]:
        return None
    i = int(np.searchsorted(ts, t))
    if i == 0:
        return traj[0, 1:4], quat_to_rot(traj[0, 4:8])
    a, b = traj[i - 1], traj[i]
    f = 0.0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
    q = _slerp(a[4:8] / np.linalg.norm(a[4:8]), b[4:8] / np.linalg.norm(b[4:8]), f)
    return (1 - f) * a[1:4] + f * b[1:4], quat_to_rot(q)


def align_se3(p: Array, q: Array) -> tuple[Array, Array]:
    """Rigid (R, t) minimizing |R p + t - q|^2: S. Umeyama, "Least-squares estimation of transformation parameters
    between two point patterns", IEEE TPAMI 13(4), 1991, without the scale."""
    mp, mq = p.mean(0), q.mean(0)
    C = (q - mq).T @ (p - mp)
    U, _, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return R, mq - R @ mp


def align_4dof(p: Array, q: Array) -> tuple[float, Array]:
    """Yaw about z and translation (closed form)."""
    mp, mq = p.mean(0), q.mean(0)
    a, b = p - mp, q - mq
    dot = float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    cross = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    theta = math.atan2(cross, dot)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return theta, mq - R @ mp


@dataclass
class Ate:
    rmse: float
    mean: float
    max: float
    n: int


def ate(est: Array, gt: Array, mode: AlignMode = "se3") -> Ate:
    pe, pg = [], []
    for row in est:
        g = interpolate(gt, row[0])
        if g is not None:
            pe.append(row[1:4])
            pg.append(g[0])
    if len(pe) < 3:
        # Nothing to align: not an error of 0 m (audit AUD-058 found `eval` printing 0 for a ground truth in another
        # time base). `n` says how many poses matched.
        return Ate(math.nan, math.nan, math.nan, len(pe))
    p, q = np.asarray(pe), np.asarray(pg)
    if mode == "se3":
        R, t = align_se3(p, q)
        p = p @ R.T + t
    elif mode == "4dof":
        theta, t = align_4dof(p, q)
        c, s = math.cos(theta), math.sin(theta)
        p = p @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + t
    d = np.linalg.norm(p - q, axis=1)
    return Ate(float(np.sqrt(np.mean(d**2))), float(d.mean()), float(d.max()), len(d))


@dataclass
class Rpe:
    trans_rmse: float
    rot_rmse_deg: float
    n: int


def rpe(est: Array, gt: Array, delta_s: float) -> Rpe:
    st = sr = 0.0
    n = 0
    for row in est:
        e1 = interpolate(est, row[0] + delta_s)
        g0, g1 = interpolate(gt, row[0]), interpolate(gt, row[0] + delta_s)
        if e1 is None or g0 is None or g1 is None:
            continue
        Re0 = quat_to_rot(row[4:8])
        rel_e = (Re0.T @ (e1[0] - row[1:4]), Re0.T @ e1[1])
        rel_g = (g0[1].T @ (g1[0] - g0[0]), g0[1].T @ g1[1])
        # err = rel_g^-1 * rel_e
        Rerr = rel_g[1].T @ rel_e[1]
        terr = rel_g[1].T @ (rel_e[0] - rel_g[0])
        st += float(terr @ terr)
        ang = float(np.linalg.norm(rot_to_vec(Rerr))) * 180 / math.pi
        sr += ang * ang
        n += 1
    if n == 0:
        return Rpe(0.0, 0.0, 0)
    return Rpe(math.sqrt(st / n), math.sqrt(sr / n), n)


def collapse_rate(loops: list[tuple[int, int]]) -> float:
    """Accepted loops joining keyframes with different (or undefined) true floors / all loops."""
    if not loops:
        return 0.0
    bad = sum(1 for a, b in loops if a != b or a < 0)
    return bad / len(loops)


def floor_height_errors(est_z: dict[int, list[float]], true_z: dict[int, float]) -> dict[int, float]:
    return {f: abs(float(np.mean(z)) - true_z[f]) for f, z in est_z.items() if f in true_z and z}


def fusion_errors(
    est: dict[int, tuple[float, Array]], truth: dict[int, tuple[float, Array]]
) -> dict[int, tuple[float, float]]:
    """Per-floor (translation m, rotation deg) between Pose4 transforms ``(theta, t)``."""

    def rz(a: float) -> Array:
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    out = {}
    for f, (th, t) in est.items():
        if f not in truth:
            continue
        th0, t0 = truth[f]
        dt = rz(-th0) @ (t - t0)
        dth = (th - th0 + math.pi) % (2 * math.pi) - math.pi
        out[f] = (float(np.linalg.norm(dt)), abs(dth) * 180 / math.pi)
    return out


def to_markdown(title: str, metrics: dict[str, float]) -> str:
    rows = "\n".join(f"| {k} | {v:.6g} |" for k, v in metrics.items())
    return f"### {title}\n\n| Metric | Value |\n| --- | --- |\n{rows}\n"


def to_json(title: str, metrics: dict[str, float]) -> str:
    return json.dumps({"title": title, "metrics": metrics})


def ate_dict(a: Ate) -> dict[str, float]:
    return {k: float(v) for k, v in asdict(a).items()}
