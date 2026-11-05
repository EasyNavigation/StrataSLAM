#!/usr/bin/env python3
"""Plots the output of strataslam_dump_scenario: per floor, the map cloud (top view, coloured by
height) over the building walls, the ground-truth path and the estimated keyframes.

The floor frame of the map is arbitrary, so each floor is aligned to the ground truth with a
4-DoF (yaw + translation) least-squares fit over its keyframes before drawing.

Usage: plot_scenario.py DUMP_DIR OUT.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def load(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def fit_yaw_translation(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares (R2, t3) with dst ~ R(yaw) src + t, yaw about z."""
    sc, dc = src.mean(0), dst.mean(0)
    a, b = src[:, :2] - sc[:2], dst[:, :2] - dc[:2]
    yaw = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rot, dc - rot @ sc


def main() -> None:
    d, out = Path(sys.argv[1]), Path(sys.argv[2])
    kf, boxes, gt = load(d / "keyframes.csv"), load(d / "boxes.csv"), load(d / "gt_path.csv")
    floors = sorted({int(f) for f in np.atleast_1d(kf["floor"])})
    fig, axes = plt.subplots(1, len(floors), figsize=(7.5 * len(floors), 7.5), squeeze=False)
    summary = []
    for ax, fid in zip(axes[0], floors):
        m = kf["floor"] == fid
        est = np.c_[kf["ex"][m], kf["ey"][m], kf["ez"][m]]
        tru = np.c_[kf["gx"][m], kf["gy"][m], kf["gz"][m]]
        rot, t = fit_yaw_translation(est, tru)
        est_w = est @ rot.T + t
        err = np.linalg.norm(est_w[:, :2] - tru[:, :2], axis=1)
        zlo, zhi = tru[:, 2].min() - 1.0, tru[:, 2].max() + 1.5
        level_z = np.median(tru[:, 2])
        # walls of the levels this floor's keyframes walk on
        for b in boxes:
            if abs(b["cz"] - level_z) > 3.5 or b["dynamic"] or b["door"]:
                continue
            r = np.array([[b["r00"], b["r01"]], [b["r10"], b["r11"]]])
            hx, hy = b["hx"], b["hy"]
            corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) @ r.T + [b["cx"], b["cy"]]
            ax.add_patch(Polygon(corners, closed=True, fill=False, ec="0.55", lw=0.6))
        cp = d / f"cloud_{fid}.csv"
        if cp.exists():
            c = load(cp)
            pts = np.c_[c["x"], c["y"], c["z"]] @ rot.T + t
            sel = (pts[:, 2] > zlo) & (pts[:, 2] < zhi)
            sc = ax.scatter(pts[sel, 0], pts[sel, 1], c=pts[sel, 2], s=0.4, cmap="viridis", alpha=0.6, rasterized=True)
            fig.colorbar(sc, ax=ax, fraction=0.04, label="z (m)")
        g = gt["true_floor"] == fid
        ax.plot(gt["x"][g], gt["y"][g], "-", c="tab:green", lw=2.2, label="ground truth")
        ax.plot(est_w[:, 0], est_w[:, 1], "-", c="tab:red", lw=1.0, label="estimated (keyframes)")
        ax.plot(*est_w[0, :2], "k^", ms=9, label="start")
        ax.set_aspect("equal")
        ax.set_title(f"floor {fid}: {m.sum()} keyframes, XY error mean {err.mean():.3f} m / max {err.max():.3f} m")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="upper right", fontsize=8)
        summary.append(f"floor {fid}: mean {err.mean():.3f} m max {err.max():.3f} m")
    fig.suptitle(d.name + " — StrataSLAM map vs ground truth (aligned per floor)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
