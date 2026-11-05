"""Fusion ablation: S07 and S11 with several seeds and the three formulations.

Runs the C++ ``strataslam_fusion_ablation`` tool (SLAM once per seed, then ``lie4``, ``se3soft`` and
``euler`` on identical inputs) and prints a Markdown table with the mean and standard deviation of the
4-DoF error, the fusion time and the convergence rate. Asserted: every formulation converges without NaN
(spec 12.5). How often ``lie4`` meets the S07/S11 thresholds is reported; the thresholds themselves are
gated per seed by the multi-seed acceptance of the scenarios (ADR-0035). The comparison between
formulations is an experimental result, not an assertion.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

FORMULATIONS = ("lie4", "se3soft", "euler")
THRESHOLDS = {"S07": (0.10, 1.0), "S11": (0.10, 1.0)}  # (metres, degrees)


@dataclass
class Cell:
    trans: list[float] = field(default_factory=list)
    rot: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    converged: int = 0
    total: int = 0
    nonfinite: int = 0


def parse_lines(lines: list[str]) -> dict[tuple[str, str], Cell]:
    cells: dict[tuple[str, str], Cell] = {}
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        cell = cells.setdefault((rec["scenario"], rec["formulation"]), Cell())
        cell.total += 1
        ok = bool(rec["converged"]) and bool(rec["finite"])
        cell.converged += ok
        cell.nonfinite += not rec["finite"]
        if ok:
            cell.trans.append(float(rec["trans_err_m"]))
            cell.rot.append(float(rec["rot_err_deg"]))
            cell.seconds.append(float(rec["seconds"]))
    return cells


def _ms(values: list[float]) -> str:
    if not values:
        return "n/a"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.4f} ± {sd:.4f}"


def to_markdown(cells: dict[tuple[str, str], Cell]) -> str:
    rows = [
        "| Scenario | Formulation | 4-DoF translation (m) | 4-DoF rotation (deg) | Optimization time (ms) | Converged |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for (scen, form), c in sorted(cells.items(), key=lambda kv: (kv[0][0], FORMULATIONS.index(kv[0][1]))):
        ms = [s * 1e3 for s in c.seconds]
        rows.append(
            f"| {scen} | {form} | {_ms(c.trans)} | {_ms(c.rot)} | {_ms(ms)} | {c.converged}/{c.total} |"
        )
    return "\n".join(rows) + "\n"


def check(cells: dict[tuple[str, str], Cell]) -> list[str]:
    """Assertions: no NaN anywhere and every run converged."""
    problems = []
    for (scen, form), c in cells.items():
        if c.nonfinite:
            problems.append(f"{scen}/{form}: {c.nonfinite} non-finite result(s)")
        if c.converged != c.total:
            problems.append(f"{scen}/{form}: only {c.converged}/{c.total} runs converged")
    return problems


def threshold_rates(cells: dict[tuple[str, str], Cell]) -> list[str]:
    """On how many seeds ``lie4`` meets the S07/S11 thresholds (reported, not asserted here)."""
    lines = []
    for scen, (tmax, rmax) in THRESHOLDS.items():
        lie = cells.get((scen, "lie4"))
        if lie is None or not lie.trans:
            continue
        ok = sum(1 for t, r in zip(lie.trans, lie.rot, strict=True) if t <= tmax and r <= rmax)
        lines.append(f"{scen}/lie4 meets {tmax} m and {rmax} deg on {ok}/{len(lie.trans)} seeds")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tool", type=Path, help="strataslam_fusion_ablation executable")
    ap.add_argument("--input", type=Path, nargs="*", default=[], help="tabulate these JSONL records instead of running the tool")
    ap.add_argument("--scenarios", nargs="+", default=["S07", "S11"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="write the Markdown table here")
    ap.add_argument("--json", type=Path, default=None, help="write the raw records here")
    ap.add_argument("--jobs", type=int, default=1, help="seeds run in parallel (one tool process per seed)")
    args = ap.parse_args()

    lines: list[str] = []
    for path in args.input:
        lines += path.read_text().splitlines()
    if not args.input and not args.tool:
        ap.error("--tool or --input is required")
    runs = [(scen, seed) for scen in (args.scenarios if not args.input else [])
            for seed in range(args.first_seed, args.first_seed + args.seeds)]

    def run(job: tuple[str, int]) -> list[str]:
        proc = subprocess.run([str(args.tool), job[0], str(job[1]), "1"], check=True, capture_output=True, text=True)
        return proc.stdout.splitlines()

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for out in ex.map(run, runs):
            lines += out
    cells = parse_lines(lines)
    table = to_markdown(cells)
    print(table)
    if args.out:
        args.out.write_text(table)
    if args.json:
        args.json.write_text("\n".join(line for line in lines if line.startswith("{")) + "\n")
    for line in threshold_rates(cells):
        print(line)
    problems = check(cells)
    for p in problems:
        print(f"fusion-ablation: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
