#!/usr/bin/env python3
"""Runs the scenarios S01-S14 on several seeds and summarizes them (multi-seed acceptance, ADR-0035).

    scripts/run_seeds.py --runner build/strataslam_sim/strataslam_scenario_runner --seeds 1001-1020 \
        --out seeds.jsonl --table seeds.md [--scenarios S07,S11] [--jobs 6]

Each run is `strataslam_scenario_runner <Sxx> <seed> 1` (one thread: bit-exact, ADR-0029). The table lists,
per scenario, how many seeds pass and the mean, standard deviation and worst value of every metric the
acceptance criteria use, plus each failure.

Exit code: without --baseline, 1 if any run fails its criteria (or crashes). With --baseline FILE (JSON
{scenario: pass rate}), 1 only if a scenario passes on fewer seeds than its baseline rate: the nightly
regression gate of ADR-0035. --write-baseline FILE records the rates of this run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCENARIOS = [f"S{i:02d}" for i in range(1, 15)]

# Metrics the acceptance criteria use, and whether the worst case is the maximum (True) or minimum.
METRICS: dict[str, list[tuple[str, bool]]] = {
    "S01": [("ate_rmse_m", True), ("map_coverage", False), ("loops_accepted", False)],
    "S02": [("ate_rmse_m", True), ("map_coverage", False), ("loops_accepted", False)],
    "S03": [("collapse_rate", True), ("floor_height_err_m", True), ("ate_rmse_m", True), ("map_coverage", False), ("floors", True),
            ("cross_floor_candidates", True)],
    "S04": [("collapse_rate", False)],
    "S05": [("floor_height_err_m", True), ("collapse_rate", True), ("floors", True)],
    "S06": [("floor_height_err_m", True), ("floors", True)],
    "S07": [("fused_trans_err_m", True), ("fused_rot_err_deg", True), ("without_anchors_trans_err_m", True),
            ("without_anchors_rot_err_deg", True), ("floors", True)],
    "S08": [("ate_rmse_m", True), ("map_coverage", False), ("floors", True)],
    "S09": [("floor_accuracy", False), ("floor_accuracy_all", False)],
    "S10": [("ate_rmse_m", True), ("map_coverage", False), ("ate_camera_only_m", True), ("ate_lidar_only_m", True)],
    "S11": [("fused_trans_err_m", True), ("fused_rot_err_deg", True), ("fused_height_err_m", True),
            ("without_anchors_trans_err_m", True), ("without_anchors_rot_err_deg", True), ("collapse_rate", True)],
    "S12": [("ate_rmse_m", True), ("map_coverage", False), ("inter_session_loops", False), ("floors", True)],
    "S13": [("ate_rmse_m", True), ("map_coverage", False), ("floor_height_err_m", True), ("floors", True)],
    "S14": [("ate_rmse_m", True), ("map_coverage", False), ("floors", True)],
}


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_one(runner: str, scenario: str, seed: int, extra: list[str], env: dict[str, str]) -> dict[str, object]:
    p = subprocess.run([runner, scenario, str(seed), "1", *extra], capture_output=True, text=True, check=False, env=env)
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            row: dict[str, object] = json.loads(line)
            return row
    return {"scenario": scenario, "seed": seed, "passed": False, "values": {},
            "failures": [f"runner exited with code {p.returncode}: {p.stderr.strip()[-300:]}"]}


def table(rows: list[dict[str, object]]) -> str:
    by: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by[str(r["scenario"])].append(r)
    out = ["| Scenario | Seeds | Pass | Metric | Mean | Std | Worst |", "|---|---|---|---|---|---|---|"]
    for s in sorted(by):
        rs = by[s]
        passed = sum(1 for r in rs if r["passed"])
        first = True
        for metric, worst_is_max in METRICS.get(s, []):
            v = [float(r["values"][metric]) for r in rs if metric in r["values"]]  # type: ignore[index, operator]
            v = [x for x in v if math.isfinite(x)]
            if not v:
                continue
            worst = max(v) if worst_is_max else min(v)
            sd = statistics.pstdev(v) if len(v) > 1 else 0.0
            out.append(f"| {s if first else ''} | {len(rs) if first else ''} | {passed if first else ''} | {metric} | "
                       f"{statistics.fmean(v):.4g} | {sd:.2g} | {worst:.4g} |")
            first = False
        # Failures of the same kind are grouped (numbers masked to find the kind), and each is listed with its own
        # value and threshold: "#" alone hid how far from the limit each seed was (audit A13-10).
        fails: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for r in rs:
            for f in r["failures"]:  # type: ignore[attr-defined]
                kind = re.sub(r"[-0-9.]+(e[-+]?\d+)?", "#", str(f))[:90]
                fails[kind].append((int(r["seed"]), str(f)))  # type: ignore[call-overload]
        for _, items in sorted(fails.items(), key=lambda x: -len(x[1])):
            listed = "; ".join(f"{seed}: {msg}" for seed, msg in items)
            out.append(f"| | | | *failure* | | | {len(items)}x: {listed} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", required=True, help="strataslam_scenario_runner executable")
    ap.add_argument("--seeds", default="1-5", help="e.g. 1-5 or 1001-1020 or 3,7,9")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, help="JSONL with one line per run")
    ap.add_argument("--table", type=Path, help="Markdown summary")
    ap.add_argument("--baseline", type=Path, help="JSON {scenario: pass rate}; fail only below it")
    ap.add_argument("--write-baseline", type=Path, help="write this run's pass rates as a baseline")
    ap.add_argument("--runner-args", default="", help="extra runner options, e.g. '--pixel-sigma 1 --image 640x480'")
    a = ap.parse_args()
    jobs = [(s, seed) for s in a.scenarios.split(",") for seed in parse_seeds(a.seeds)]
    # The simulator's ray casting uses OpenMP: share the cores between the parallel runs (the SLAM itself
    # runs on one thread, and the simulated data do not depend on the number of OpenMP threads).
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // a.jobs)))
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(lambda j: run_one(a.runner, j[0], j[1], a.runner_args.split(), env), jobs))
    if a.out:
        a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    md = table(rows)
    if a.table:
        a.table.write_text(md)
    print(md)
    failed = [f"{r['scenario']}/{r['seed']}" for r in rows if not r["passed"]]
    print(f"{len(rows) - len(failed)}/{len(rows)} runs pass" + (f"; failing: {' '.join(failed)}" if failed else ""))
    rates: dict[str, float] = {}
    for s in sorted({str(r["scenario"]) for r in rows}):
        rs = [r for r in rows if r["scenario"] == s]
        rates[s] = sum(1 for r in rs if r["passed"]) / len(rs)
    if a.write_baseline:
        a.write_baseline.write_text(json.dumps(rates, indent=2, sort_keys=True) + "\n")
    if a.baseline:
        base: dict[str, float] = json.loads(a.baseline.read_text())
        worse = [f"{s}: {rates[s]:.0%} < baseline {base[s]:.0%}" for s in rates if s in base and rates[s] + 1e-9 < base[s]]
        for w in worse:
            print(f"regression: {w}")
        return 1 if worse else 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
