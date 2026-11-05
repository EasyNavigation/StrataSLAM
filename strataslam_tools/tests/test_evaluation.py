"""Python metrics: known values and cross-check against the C++ implementation."""

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import core_executable

from strataslam_tools import evaluation as ev


def make_traj(n: int = 200, dt: float = 0.1) -> np.ndarray:
    rows = []
    for i in range(n):
        s = i * dt
        yaw = 0.2 * s
        rows.append([s, s, 2 * math.sin(0.3 * s), 0.3 * math.cos(0.2 * s), 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)])
    return np.asarray(rows)


def transform(traj: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = traj.copy()
    out[:, 1:4] = traj[:, 1:4] @ R.T + t
    for i in range(len(out)):
        Rq = R @ ev.quat_to_rot(traj[i, 4:8])
        # rotation matrix -> quaternion (x, y, z, w)
        w = math.sqrt(max(0.0, 1 + np.trace(Rq))) / 2
        if w < 1e-6:
            raise AssertionError("test rotation too large")
        out[i, 4:8] = [(Rq[2, 1] - Rq[1, 2]) / (4 * w), (Rq[0, 2] - Rq[2, 0]) / (4 * w), (Rq[1, 0] - Rq[0, 1]) / (4 * w), w]
    return out


def rot(yaw: float, pitch: float = 0.0) -> np.ndarray:
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])


def test_ate_zero_and_alignment() -> None:
    gt = make_traj()
    assert ev.ate(gt, gt, "none").rmse == pytest.approx(0, abs=1e-12)
    est = transform(gt, rot(0.4, 0.2).T, np.array([1.0, -2.0, 0.5]))
    assert ev.ate(est, gt, "none").rmse > 0.5
    assert ev.ate(est, gt, "se3").rmse == pytest.approx(0, abs=1e-9)


def test_4dof_alignment_exact() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(-5, 5, (4, 3))
    theta, t = 0.7, np.array([3.0, -2.0, 1.5])
    c, s = math.cos(theta), math.sin(theta)
    q = p @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + t
    th, tt = ev.align_4dof(p, q)
    assert th == pytest.approx(theta, abs=1e-9)
    assert np.allclose(tt, t, atol=1e-9)


def test_collapse_height_fusion() -> None:
    assert ev.collapse_rate([]) == 0.0
    assert ev.collapse_rate([(0, 0), (1, 1), (0, 1)]) == pytest.approx(1 / 3)
    assert ev.collapse_rate([(-1, -1)]) == 1.0
    assert ev.floor_height_errors({1: [4.1, 4.3]}, {1: 4.5})[1] == pytest.approx(0.3)
    err = ev.fusion_errors({1: (0.52, np.array([1.0, 2.0, 3.05]))}, {1: (0.5, np.array([1.0, 2.0, 3.0]))})
    assert err[1][1] == pytest.approx(0.02 * 180 / math.pi)
    assert 0.04 < err[1][0] < 0.06


def test_markdown_and_json() -> None:
    assert "| ate | 0.5 |" in ev.to_markdown("S01", {"ate": 0.5})
    assert ev.to_json("S01", {"ate": 0.5}) == '{"title": "S01", "metrics": {"ate": 0.5}}'


@pytest.mark.parametrize("mode", ["none", "se3", "4dof"])
def test_matches_cpp_implementation(tmp_path: Path, mode: str) -> None:
    exe = core_executable("STRATASLAM_EVAL_BIN", "strataslam_eval_tool")
    gt = make_traj()
    rng = np.random.default_rng(3)
    est = transform(gt, rot(0.3), np.array([0.5, 0.2, -0.1]))
    est[:, 1:4] += rng.normal(0, 0.03, (len(est), 3))
    # Drop some samples and shift the stamps so that GT interpolation is exercised.
    est = est[::3].copy()
    est[:, 0] += 0.013
    for name, arr in (("est", est), ("gt", gt)):
        np.savetxt(tmp_path / f"{name}.tum", arr, fmt="%.12f")
    out = subprocess.run([exe, str(tmp_path / "est.tum"), str(tmp_path / "gt.tum"), mode, "1.0"],
                         check=True, capture_output=True, text=True).stdout
    import json

    cpp = json.loads(out)
    py = ev.ate(est, gt, mode)  # type: ignore[arg-type]
    assert cpp["n"] == py.n
    assert cpp["ate_rmse"] == pytest.approx(py.rmse, rel=1e-6, abs=1e-9)
    assert cpp["ate_max"] == pytest.approx(py.max, rel=1e-6, abs=1e-9)
    r = ev.rpe(est, gt, 1.0)
    assert cpp["rpe_n"] == r.n
    assert cpp["rpe_trans_rmse"] == pytest.approx(r.trans_rmse, rel=1e-4, abs=1e-6)
    assert cpp["rpe_rot_rmse_deg"] == pytest.approx(r.rot_rmse_deg, rel=1e-4, abs=1e-5)


def test_ate_without_matched_poses_is_undefined_not_zero() -> None:
    # A ground truth in another time base (1000 s away) matches nothing: there is no ATE (audit AUD-058).
    t = np.arange(0.0, 10.0, 0.1)
    gt = np.column_stack([t, np.sin(t), np.cos(t), 0 * t, 0 * t, 0 * t, 0 * t, 1 + 0 * t])
    est = gt.copy()
    est[:, 0] += 1000.0
    a = ev.ate(est, gt, "se3")
    assert a.n == 0 and math.isnan(a.rmse)
