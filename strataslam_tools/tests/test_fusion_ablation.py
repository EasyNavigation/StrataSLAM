"""The ablation script's parsing, table and CI checks, on canned tool output."""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fusion_ablation", REPO / "strataslam_tools" / "bench" / "fusion_ablation.py")
assert spec and spec.loader
fa = importlib.util.module_from_spec(spec)
sys.modules["fusion_ablation"] = fa
spec.loader.exec_module(fa)


def rec(scen: str, seed: int, form: str, trans: float, rot: float, ok: bool = True, finite: bool = True) -> str:
    return json.dumps(
        {"scenario": scen, "seed": seed, "formulation": form, "converged": ok, "finite": finite,
         "trans_err_m": trans, "rot_err_deg": rot, "seconds": 0.002}
    )


def test_table_and_checks_on_clean_output() -> None:
    lines = [rec("S07", s, f, 0.05 + 0.01 * s, 0.3) for s in range(3) for f in fa.FORMULATIONS]
    cells = fa.parse_lines(lines)
    assert cells[("S07", "lie4")].total == 3
    md = fa.to_markdown(cells)
    assert "| S07 | lie4 |" in md and "3/3" in md
    assert fa.check(cells) == []


def test_nan_is_flagged_and_lie4_threshold_rate_is_reported() -> None:
    lines = [rec("S11", 1, "lie4", 0.3, 0.2), rec("S11", 2, "lie4", 0.05, 0.2),
             rec("S11", 1, "euler", 0.0, 0.0, ok=True, finite=False)]
    cells = fa.parse_lines(lines)
    problems = fa.check(cells)
    assert any("euler: 1 non-finite" in p for p in problems)
    assert not any("lie4" in p for p in problems)  # thresholds are gated per seed elsewhere (ADR-0035)
    assert fa.threshold_rates(cells) == ["S11/lie4 meets 0.1 m and 1.0 deg on 1/2 seeds"]


def test_unconverged_runs_are_flagged_and_excluded_from_statistics() -> None:
    lines = [rec("S07", 1, "se3soft", 0.04, 0.1), rec("S07", 2, "se3soft", 9.9, 9.9, ok=False)]
    cells = fa.parse_lines(lines)
    c = cells[("S07", "se3soft")]
    assert c.converged == 1 and c.total == 2 and c.trans == [0.04]
    assert any("only 1/2 runs converged" in p for p in fa.check(cells))


def _rec(scen: str, form: str, t: float, r: float, s: float = 0.01) -> str:
    return json.dumps({"scenario": scen, "formulation": form, "seed": 1, "trans_err_m": t, "rot_err_deg": r,
                       "seconds": s, "converged": True, "finite": True})


def test_table_values_are_mean_and_sample_deviation_in_formulation_order() -> None:
    # Audit AUD-032: a deviation always 0, a median instead of the mean and the row order went unseen.
    lines = [_rec("S07", form, t, 0.0) for form in ("euler", "se3soft", "lie4") for t in (0.1, 0.2, 0.6)]
    md = fa.to_markdown(fa.parse_lines(lines))
    rows = [ln for ln in md.splitlines() if ln.startswith("| S07")]
    assert [r.split("|")[2].strip() for r in rows] == ["lie4", "se3soft", "euler"]
    # mean 0.3 (median 0.2), sample standard deviation 0.2646
    assert "0.3000 ± 0.2646" in rows[0]


def test_threshold_is_inclusive() -> None:
    # A seed exactly at 0.10 m and 1.0 deg meets the S07 threshold ("<=", not "<").
    cells = fa.parse_lines([_rec("S07", "lie4", 0.10, 1.0), _rec("S07", "lie4", 0.11, 0.5)])
    assert fa.threshold_rates(cells) == ["S07/lie4 meets 0.1 m and 1.0 deg on 1/2 seeds"]
