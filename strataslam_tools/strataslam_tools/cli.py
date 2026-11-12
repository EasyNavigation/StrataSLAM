"""The `strataslam` command line: doctor, map, fuse, export, eval, vocab train, imu-noise."""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from strataslam_tools import evaluation, mapio


def _find_tool(name: str, package: str) -> str:
    """Locate an executable of the C++ packages: PATH, then lib/<package>/ of the workspace prefixes, then the
    repository's own CMake build (`cmake -S superbuild -B build` with `pip install -e strataslam_tools`).

    strataslam_core is a plain CMake package, which colcon lists in CMAKE_PREFIX_PATH but not in
    AMENT_PREFIX_PATH (that one only holds ament packages): both are searched."""
    found = shutil.which(name)
    if found:
        return found
    prefixes = os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) + os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep)
    for prefix in filter(None, prefixes):
        cand = Path(prefix) / "lib" / package / name
        if cand.exists():
            return str(cand)
    cand = Path(__file__).resolve().parents[2] / "build" / package / name
    if cand.exists():
        return str(cand)
    raise SystemExit(
        f"strataslam: cannot find '{name}' (package {package}). In a colcon workspace, source it in this terminal: "
        "source ~/strataslam_ws/install/setup.bash (or the workspace you built). With the CMake build, put "
        f"build/{package} on PATH."
    )


_NO_ROS = (
    "needs ROS 2: source /opt/ros/jazzy/setup.bash and your workspace (source ~/strataslam_ws/install/setup.bash) "
    "in this terminal"
)


def cmd_doctor(args: argparse.Namespace) -> int:
    from strataslam_tools import doctor

    # Exit codes: 0 all ✔, 1 some ⚠, 2 some ✘, 3 the doctor itself could not run (no profile, no bag, unreadable
    # bag): a different code, so that a script does not take a missing bag for a warning (audit A8-09).
    if args.bag and not Path(args.bag).exists():
        print(f"strataslam doctor: no such bag: {args.bag}", file=sys.stderr)
        return 3
    try:
        code, text = doctor.run(args.profile, bag=args.bag, as_json=args.json, seconds=args.seconds)
    except ImportError as e:  # rclpy / rosbag2_py: ROS is not sourced
        print(f"strataslam doctor {_NO_ROS} ({e})", file=sys.stderr)
        return 3
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as e:
        print(f"strataslam doctor: {e}", file=sys.stderr)
        return 3
    print(text)
    return code


def cmd_map(args: argparse.Namespace) -> int:
    if shutil.which("ros2") is None:
        print(f"strataslam map {_NO_ROS}", file=sys.stderr)
        return 3
    cmd = ["ros2", "launch", "strataslam_ros", "strataslam.launch.py", f"profile:={args.profile}", "mode:=map"]
    if args.vocabulary:
        cmd.append(f"vocabulary:={args.vocabulary}")
    return subprocess.call(cmd)


def cmd_fuse(args: argparse.Namespace) -> int:
    cmd = [_find_tool("strataslam_fuse", "strataslam_core"), args.map_dir, "--formulation", args.formulation]
    if args.anchors:
        cmd += ["--anchors", args.anchors]
    if args.floor_height_prior is not None:
        cmd += ["--floor-height-prior", str(args.floor_height_prior)]
    if args.same_level:
        cmd += ["--same-level", args.same_level]
    if args.stacking:
        cmd += ["--stacking", args.stacking]
    return subprocess.call(cmd)


def cmd_export(args: argparse.Namespace) -> int:
    problems = mapio.validate_map(args.map_dir)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    root = Path(args.map_dir)
    out = root / "export"
    out.mkdir(exist_ok=True)
    if args.format == "grid":
        return export_grids(args, root, out)
    reader = {".pcd": mapio.read_pcd, ".ply": mapio.read_ply}
    writer = mapio.write_ply if args.format == "ply" else mapio.write_pcd
    count = 0
    # The building: fused if `strataslam fuse` ran (PCD first, PLY from older maps), otherwise as the odometry left
    # the floors (written with the map, ADR-0064).
    candidates = (root / "fused" / "global_cloud.pcd", root / "fused" / "global_cloud.ply", root / "global_cloud.pcd")
    fused = [p for p in candidates if p.exists()][:1]
    sources = fused + sorted(root.glob("floor_*/floor_cloud.pcd"))
    for src in sources:
        if not src.exists():
            continue
        pts, colors = reader[src.suffix](src)
        name = "global_cloud" if src.name.startswith("global_cloud") else src.parent.name
        writer(out / f"{name}.{args.format}", pts, colors)
        count += 1
    print(f"exported {count} cloud(s) to {out}")
    return 0 if count else 1


def export_grids(args: argparse.Namespace, root: Path, out: Path) -> int:
    """One occupancy grid per floor, in that floor's frame (that of its floor_cloud.pcd)."""
    from . import grid

    count = 0
    for f in mapio.read_manifest(root)["floors"]:
        rays = grid.floor_rays(root / f["dir"])
        try:
            g = grid.build_grid(rays, args.min_height, args.max_height, args.resolution)
        except ValueError:
            print(f"{f['dir']}: no points between {args.min_height} and {args.max_height} m above the base",
                  file=sys.stderr)
            continue
        y = grid.write_grid(g, out / f"{Path(f['dir']).name}.pgm")
        occ = g.occupancy()
        print(f"{y}: {g.shape[1]} x {g.shape[0]} cells of {args.resolution} m, "
              f"{int(np.sum(occ >= grid.OCCUPIED_THRESH))} occupied, {int(np.sum(occ <= grid.FREE_THRESH))} free")
        count += 1
    return 0 if count else 1


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluation against a ground-truth directory: `trajectory.tum` and optionally `floors.csv` (stamp_ns,floor)."""
    problems = mapio.validate_map(args.map_dir)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    root = Path(args.map_dir)
    gt = evaluation.read_tum(Path(args.gt) / "trajectory.tum")
    manifest = mapio.read_manifest(root)
    fused = mapio.read_fused(root)
    metrics: dict[str, float] = {}
    est_rows, est_floor = [], []
    for f in manifest["floors"]:
        kf = mapio.read_keyframes(root / f["dir"])
        pts = kf[:, 3:6]
        if f["id"] in fused:
            pts = mapio.apply_pose4(fused[f["id"]][0], fused[f["id"]][1], pts)
        for row, p in zip(kf, pts):
            est_rows.append([row[1] * 1e-9, *p, *row[6:10]])
            est_floor.append((int(row[0]), f["id"]))
    est = np.asarray(sorted(est_rows, key=lambda r: r[0]))
    mode = "4dof" if fused else "se3"
    a = evaluation.ate(est, gt, mode)  # type: ignore[arg-type]
    if a.n < 3:
        print(f"only {a.n} of {len(est)} keyframes fall inside the ground truth's time span "
              f"({gt[0, 0]:.3f} to {gt[-1, 0]:.3f} s; keyframes {est[0, 0]:.3f} to {est[-1, 0]:.3f} s): "
              "are both in the same time base?", file=sys.stderr)
        return 1
    metrics.update({"ate_rmse_m": a.rmse, "ate_max_m": a.max, "keyframes": float(len(est)), "fused": float(bool(fused))})
    floors_csv = Path(args.gt) / "floors.csv"
    if floors_csv.exists():
        truth = np.loadtxt(floors_csv, delimiter=",", skiprows=1, ndmin=2)
        ts, fl = truth[:, 0] * 1e-9, truth[:, 1].astype(int)
        kf_stamp: dict[int, float] = {}
        for f in manifest["floors"]:
            for row in mapio.read_keyframes(root / f["dir"]):
                kf_stamp[int(row[0])] = row[1] * 1e-9
        loops = []
        for f in manifest["floors"]:
            for e in mapio.read_edges(root / f["dir"]):
                if e["type"] == "loop":
                    tf = [int(fl[int(np.argmin(np.abs(ts - kf_stamp[i])))]) for i in (e["from"], e["to"]) if i in kf_stamp]
                    if len(tf) == 2:
                        loops.append((tf[0], tf[1]))
        metrics["loops"] = float(len(loops))
        metrics["collapse_rate"] = evaluation.collapse_rate(loops)
    title = f"strataslam eval: {root.name}"
    print(evaluation.to_json(title, metrics) if args.json else evaluation.to_markdown(title, metrics))
    return 0


def cmd_vocab(args: argparse.Namespace) -> int:
    if args.action != "train":
        raise SystemExit("usage: strataslam vocab train <image_folder> --out vocab.bin")
    return subprocess.call([_find_tool("strataslam_vocab_train", "strataslam_ros"), args.folder, args.out])


def cmd_imu_noise(args: argparse.Namespace) -> int:
    from strataslam_tools import allan

    try:
        acc, gyr, fs = allan.read_imu_bag(args.bag, args.topic)
    except ImportError as e:  # a rosbag2 needs rosbag2_py
        print(f"strataslam imu-noise on a rosbag {_NO_ROS} ({e})", file=sys.stderr)
        return 3
    res, curves = allan.imu_noise(acc, gyr, fs, margin=args.margin)
    print(f"{len(acc)} samples at {fs:.1f} Hz ({len(acc) / fs / 3600:.2f} h), margin x{args.margin:g}")
    for w in res.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(res.noise_yaml())
    if args.plot:
        allan.plot_curves(curves, args.plot)
        print(f"curves: {args.plot}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="strataslam", description="Multi-floor 3D SLAM tools")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="validate sensors, TF and synchronization")
    d.add_argument("profile")
    d.add_argument("--bag", help="rosbag2 to check instead of a live robot")
    d.add_argument("--json", action="store_true")
    d.add_argument("--seconds", type=float, default=6.0, help="listening time when live")
    d.set_defaults(func=cmd_doctor)

    m = sub.add_parser("map", help="start mapping (wraps the launch file)")
    m.add_argument("profile")
    m.add_argument("--vocabulary", default="")
    m.set_defaults(func=cmd_map)

    f = sub.add_parser("fuse", help="offline floor fusion of a saved map")
    f.add_argument("map_dir")
    f.add_argument("--formulation", choices=["lie4", "se3soft", "euler"], default="lie4")
    f.add_argument("--anchors")
    f.add_argument("--floor-height-prior", type=float, default=None)
    f.add_argument("--same-level", help='floors confirmed to be one level, e.g. "2,3" or "2,3;0,4" (docs/user/fusion.md)')
    f.add_argument("--stacking", choices=["on", "off"], default=None, help="structural registration of the floors (default on)")
    f.set_defaults(func=cmd_fuse)

    e = sub.add_parser("export", help="export the clouds of a map")
    e.add_argument("map_dir")
    e.add_argument("--format", choices=["pcd", "ply", "grid"], default="pcd",
                   help="pcd or ply: the clouds (default pcd, the one PCL and the ROS tools read); grid: an occupancy "
                        "grid per floor, PGM + YAML for nav2's map_server")
    e.add_argument("--resolution", type=float, default=0.05, help="grid: cell size in m (default 0.05)")
    e.add_argument("--min-height", type=float, default=0.05,
                   help="grid: obstacles from this height above the robot's base, m (default 0.05: above the floor)")
    e.add_argument("--max-height", type=float, default=2.0,
                   help="grid: obstacles up to this height above the robot's base, m (default 2.0)")
    e.set_defaults(func=cmd_export)

    v = sub.add_parser("eval", help="evaluate a map against ground truth")
    v.add_argument("map_dir")
    v.add_argument("--gt", required=True, help="directory with trajectory.tum (and floors.csv)")
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_eval)

    n = sub.add_parser("imu-noise", help="IMU noise (Allan deviation) from a rosbag of the IMU at rest")
    n.add_argument("bag")
    n.add_argument("--topic", required=True, help="sensor_msgs/Imu topic, e.g. /oak/imu")
    n.add_argument("--margin", type=float, default=1.0, help="factor on the result: a moving IMU is worse than a parked one (2-3 is usual)")
    n.add_argument("--plot", help="write the Allan curves to this PNG (needs matplotlib)")
    n.set_defaults(func=cmd_imu_noise)

    c = sub.add_parser("vocab", help="visual vocabulary")
    c.add_argument("action", choices=["train"])
    c.add_argument("folder")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_vocab)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (mapio.MapFormatError, FileNotFoundError) as e:
        print(f"strataslam: {e}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError, yaml.YAMLError, struct.error) as e:
        # Corrupt or truncated map files: a message, not a traceback (audit A6-06).
        print(f"strataslam: invalid input ({type(e).__name__}): {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())


