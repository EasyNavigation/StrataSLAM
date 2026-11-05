#!/usr/bin/env python3
"""Coverage gate (spec 12.9): >= 85 % in geometry, backend/factors, floors and config; >= 65 % overall.

Usage: check_coverage.py <build_dir> <source_root>   (the build was made with -DSTRATASLAM_ENABLE_COVERAGE=ON
and the tests were run). Needs gcovr.
"""
import json
import subprocess
import sys
from pathlib import Path

GATES = {
    "geometry": (["pose4", "gravity"], 85.0),
    "backend/factors": (["factors.hpp", "factors4"], 85.0),
    "floors": (["floor_manager", "transition"], 85.0),
    "config": (["config_loader"], 85.0),
}
OVERALL = 65.0


def main() -> int:
    build, root = Path(sys.argv[1]), Path(sys.argv[2])
    out = subprocess.run(
        ["gcovr", "-r", str(root), "--object-directory", str(build), "--json-summary", "-", "--exclude", ".*/test/.*",
         "--exclude", ".*/tools/.*", "--exclude", ".*_deps/.*", "--exclude", "/usr/.*", "--exclude", "/opt/.*"],
        check=True, capture_output=True, text=True).stdout
    summary = json.loads(out)
    files = summary["files"]
    problems = []
    for name, (needles, minimum) in GATES.items():
        sel = [f for f in files if any(n in f["filename"] for n in needles)]
        if not sel:
            problems.append(f"{name}: no source files matched {needles}")
            continue
        covered = sum(f["line_covered"] for f in sel)
        total = sum(f["line_total"] for f in sel)
        pct = 100.0 * covered / total if total else 100.0
        print(f"{name}: {pct:.1f} % ({covered}/{total} lines)")
        if pct < minimum:
            problems.append(f"{name}: {pct:.1f} % < {minimum} %")
    total_pct = 100.0 * summary["line_covered"] / summary["line_total"]
    print(f"overall: {total_pct:.1f} %")
    if total_pct < OVERALL:
        problems.append(f"overall: {total_pct:.1f} % < {OVERALL} %")
    for p in problems:
        print(f"coverage: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
