#!/usr/bin/env bash
# Regenerates every published result with fixed seeds, from a Release superbuild and the datasets
# (ADR-0035). One command, run by the nightly CI; docs/scenarios.md and docs/results.md are copies of
# <out_dir>/scenarios.md and <out_dir>/results.md.
#
#   scripts/download_datasets.sh data
#   scripts/make_results.sh build data results [jobs]
# STRATASLAM_RESULTS_FROM=N resumes at step N (the outputs of the earlier steps must be in results/).
# Split runs (the nightly, whose single job outgrew the 6 h a runner allows): STRATASLAM_RESULTS_STEPS=2,3 runs only
# those steps, STRATASLAM_RESULTS_SCENARIOS=S01,S02 only those scenarios in steps 2 and 3, and
# STRATASLAM_RESULTS_STEPS=none runs nothing: the parts found in <out_dir>/parts/*/ are merged and the pages written.
set -euo pipefail
b="$1"; data="$2"; out="$3"; jobs="${4:-4}"
from="${STRATASLAM_RESULTS_FROM:-1}"
steps=",${STRATASLAM_RESULTS_STEPS:-1,2,3,4,5,6},"
step() { [ "$1" -ge "$from" ] && [[ "$steps" == *",$1,"* ]]; }
scen=()
if [ -n "${STRATASLAM_RESULTS_SCENARIOS:-}" ]; then scen=(--scenarios "$STRATASLAM_RESULTS_SCENARIOS"); fi
repo="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$out/scenarios_seed42"
out="$(cd "$out" && pwd)"  # absolute: the parts of a split run are copied from inside each one
sim="$b/strataslam_sim"; core="$b/strataslam_core"
# Every step below goes on past a failed run (`|| true`) so that one bad case does not lose the rest: a binary
# that cannot even start (a shared library not found) would then publish zero passes everywhere. Refuse up front.
[ "$steps" = ",none," ] || for t in "$sim/strataslam_scenario_tests" "$sim/strataslam_scenario_runner" "$sim/strataslam_benchmarks" \
         "$core/strataslam_tum_run" "$core/strataslam_kitti_run"; do
  if [ ! -x "$t" ] || ldd "$t" | grep -q "not found"; then
    echo "make_results: $t is missing or cannot load its libraries:" >&2; ldd "$t" 2>&1 | grep "not found" >&2
    exit 1
  fi
done

if step 1; then
echo "1/6 scenarios S01-S17, seed 42 (the CI seed)"
STRATASLAM_REPORT_DIR="$out/scenarios_seed42" "$sim/strataslam_scenario_tests" --gtest_filter='Scenario.S??_*-Scenario.S02_MapSurvivesDiskRoundTrip' \
  > "$out/scenarios_seed42.log" 2>&1 || true

fi
if step 2; then
echo "2/6 scenarios S01-S17, seeds 1001-1020"
python3 "$repo/scripts/run_seeds.py" --runner "$sim/strataslam_scenario_runner" --seeds 1001-1020 --jobs "$jobs" "${scen[@]}" \
  --out "$out/seeds.jsonl" --table "$out/seeds.md" --write-baseline "$out/seeds_baseline.json" > "$out/seeds.log" || true

fi
if step 3; then
echo "3/6 camera noise: 1 px at 640x480 and at 320x240, seeds 1001-1005"
python3 "$repo/scripts/run_seeds.py" --runner "$sim/strataslam_scenario_runner" --seeds 1001-1005 --jobs "$jobs" "${scen[@]}" \
  --runner-args '--pixel-sigma 1 --image 640x480' --out "$out/noise_640.jsonl" --table "$out/noise_640.md" > "$out/noise_640.log" || true
python3 "$repo/scripts/run_seeds.py" --runner "$sim/strataslam_scenario_runner" --seeds 1001-1005 --jobs "$jobs" "${scen[@]}" \
  --runner-args '--pixel-sigma 1' --out "$out/noise_320.jsonl" --table "$out/noise_320.md" > "$out/noise_320.log" || true

fi
if step 4; then
echo "4/6 TUM RGB-D fr1/xyz and fr2/xyz (visual odometry)"
: > "$out/tum.jsonl"
for s in rgbd_dataset_freiburg1_xyz rgbd_dataset_freiburg2_xyz; do
  if [ -d "$data/$s" ]; then "$core/strataslam_tum_run" "$data/$s" "$out/$s.tum" | tail -1 >> "$out/tum.jsonl"; fi
done

fi
if step 5; then
echo "5/6 KITTI raw 2011_09_30_drive_0027 (full LiDAR pipeline)"
k="$data/2011_09_30/2011_09_30_drive_0027_sync"
if [ -d "$k" ]; then "$core/strataslam_kitti_run" "$k" "$data/2011_09_30" "$out/kitti_0027" 100000 1 | tail -1 > "$out/kitti.json"; fi

fi
if step 6; then
echo "6/6 benchmarks (spec 12.8)"
"$sim/strataslam_benchmarks" --benchmark_format=json --benchmark_out="$out/benchmarks.json" > "$out/benchmarks.txt" 2>&1 || true
fi

# The parts of a split run: the runs of steps 2 and 3 joined and summarized again, everything else copied.
if [ -d "$out/parts" ]; then
  for k in seeds noise_640 noise_320; do
    shopt -s nullglob; f=("$out"/parts/*/"$k.jsonl"); shopt -u nullglob
    [ ${#f[@]} -gt 0 ] || continue
    extra=(); [ "$k" = seeds ] && extra=(--write-baseline "$out/seeds_baseline.json")
    python3 "$repo/scripts/run_seeds.py" --input "${f[@]}" --out "$out/$k.jsonl" --table "$out/$k.md" "${extra[@]}" > /dev/null || true
  done
  for d in "$out"/parts/*/; do
    (cd "$d" && find . -type f ! -name 'seeds*' ! -name 'noise_*' -exec cp --parents {} "$out/" \;)
  done
fi

# When and from what: the date, and the commit when the sources are a git checkout (AUD-054).
stamp="$(date -u +%Y-%m-%d), $(git -C "$repo" describe --always --dirty 2>/dev/null || echo "sources outside git")"
python3 - "$out" "$stamp" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
reports = sorted((out / "scenarios_seed42").glob("S*.md"))
(out / "scenarios.md").write_text(
    "# Scenario results (seed 42)\n\nGenerated by `scripts/make_results.sh` from the scenario tests (the CI seed). The "
    "distribution over 20 other seeds is in [results.md](results.md).\n\n" + "\n".join(r.read_text() for r in reports))
def read(name): return (out / name).read_text() if (out / name).exists() else "(not run)\n"
tum = [json.loads(l) for l in read("tum.jsonl").splitlines() if l.startswith("{")]
kitti = json.loads(read("kitti.json")) if (out / "kitti.json").exists() and read("kitti.json").startswith("{") else None
md = ["# Results", "", f"Generated by `scripts/make_results.sh` ({sys.argv[2]}; one thread, fixed seeds). Acceptance over seeds: ADR-0035.", "",
      "## Scenarios S01-S17 over seeds 1001-1020", "", read("seeds.md"),
      "## Camera noise: 1 px at 640x480 (seeds 1001-1005)", "", read("noise_640.md"),
      "## Camera noise: 1 px at 320x240, twice the default angular noise (seeds 1001-1005)", "", read("noise_320.md"),
      "## Real data", "", "| Dataset | Pipeline | Frames | ATE RMSE (m) | ATE max (m) |", "| --- | --- | --- | --- | --- |"]
for t in tum:
    md.append(f"| TUM {Path(t['sequence']).name} | visual odometry | {t['frames']} | {t['ate_rmse']:.4f} | {t['ate_max']:.4f} |")
if kitti:
    md.append(f"| KITTI 2011_09_30_drive_0027 ({kitti['path_length_m']:.0f} m) | full LiDAR SLAM | {kitti['scans']} | "
              f"{kitti['ate_rmse_m']:.4f} | {kitti['ate_max_m']:.4f} |")
    md += ["", f"KITTI: {kitti['keyframes']} keyframes, {kitti['loops']} loop closure{'' if kitti['loops'] == 1 else 's'}, {kitti['ms_per_scan']:.0f} ms per scan, one thread."]
if (out / "benchmarks.json").exists():
    budgets = {"BM_OrbExtract_640x480_1000": "<= 33 ms", "BM_LidarFrontend_64beams": "<= 50 ms per scan",
               "BM_Isam2KeyframeUpdate": "<= 20 ms per keyframe"}
    md += ["", "## Performance (spec 12.8)", "", "| Benchmark | Time | Per item | Budget |", "| --- | --- | --- | --- |"]
    for bm in json.loads((out / "benchmarks.json").read_text())["benchmarks"]:
        per = [f"{v * 1e3:.3f} ms" for k, v in bm.items() if k.startswith("time_per")]
        name = bm["name"].split("/")[0]
        md.append(f"| {name} | {bm['real_time']:.2f} {bm['time_unit']} | {per[0] if per else ''} | {budgets.get(name, '')} |")
(out / "results.md").write_text("\n".join(md) + "\n")
print("wrote", out / "results.md", "and", out / "scenarios.md")
PY
