"""Tests for tools/b4_t1_trigger.py — D-193 B4 T1 mechanical trigger evaluator.

Spec source (BINDING): programs/program_2_dense_vs_moe_sub100m/phase3_p6_prereg.md
§8.1 (1h Item 1 verbatim) + §9.1 (σ-proxy chain σ_HS=1.0, σ_WG=1.4).

Coverage matrix (per d193 design doc §7 test plan):
  1.  Fire: synthetic data with σ̂_HS = 1.6 → ratio 1.60 > 1.50 → FIRE.
  2.  No-fire: synthetic data with σ̂_HS = 0.8, σ̂_WG = 1.2 → ratios 0.80 / 0.857 → NO FIRE.
  3.  Edge case at threshold: ratio exactly 1.50 → NO FIRE (strict `>`).
  4.  Edge case ratio = 1.501 → FIRE.
  5.  Pooled-σ formula correctness vs manual + scipy/numpy reference.
  6.  Fewer than 8 scores → exit 1 + FATAL.
  7.  NaN handling → exit 1 + FATAL identifying which cell-seed.
  8.  Score values outside [0, 100] → WARN, do not fail.
  9.  JSON output schema: all required fields present, correct types.
  10. df = 4 documented in output (k=4 cells × (N=2 - 1) per cell).
  11. Reference σ values match prereg §9.1 (default σ_HS=1.0, σ_WG=1.4).
  12. Threshold matches prereg §8.1 (default 1.50).
  13. CLI --threshold override works (parametrized).
  14. CLI --sigma-ref-{hs,wg} override works (parametrized).
  15. Symmetry: WG-only excess fires.
  16. End-to-end CLI subprocess (rc=0 success, rc=1 FATAL).

Per CLAUDE.md "TDD always" — tests written before tools/b4_t1_trigger.py exists.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Optional

import pytest

TOOLS_DIR = pathlib.Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

b4_t1_trigger = importlib.import_module("b4_t1_trigger")


# ---------------------------------------------------------------------------
# Locked constants test (§11 + §12 prereg-default verification)
# ---------------------------------------------------------------------------


def test_default_sigma_ref_matches_prereg_9_1() -> None:
    """Default σ_ref values must equal prereg §9.1 (math_theorist 1f σ-proxy)."""
    assert b4_t1_trigger.SIGMA_REF["HS"] == 1.0
    assert b4_t1_trigger.SIGMA_REF["WG"] == 1.4


def test_default_threshold_matches_prereg_8_1() -> None:
    """Default threshold must equal prereg §8.1 1h-amended verbatim (1.50)."""
    assert b4_t1_trigger.T1_THRESHOLD == 1.50


def test_required_benchmarks_are_hs_wg_only() -> None:
    """Trigger statistic is max over {HS, WG} only; not MMLU, not GSM8K."""
    assert tuple(b4_t1_trigger.REQUIRED_BENCHMARKS) == ("HS", "WG")


def test_expected_cells_match_locked_launch_order() -> None:
    """The 8 cells are the first 8 of the locked launch order: {A,B,C,D}×{42,43}."""
    expected = {"A42", "A43", "B42", "B43", "C42", "C43", "D42", "D43"}
    assert set(b4_t1_trigger.EXPECTED_CELL_SEEDS) == expected


def test_df_constant_is_4() -> None:
    """df = k·(N-1) = 4·(2-1) = 4 per prereg §8.1."""
    assert b4_t1_trigger.POOLED_DF == 4


# ---------------------------------------------------------------------------
# Helper: build an 8-cell consolidated input
# ---------------------------------------------------------------------------


def _consolidated_input(
    hs: Dict[str, float],
    wg: Dict[str, float],
    mmlu: Optional[Dict[str, float]] = None,
    gsm8k: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    scores: Dict[str, Dict[str, float]] = {}
    cells = ("A42", "A43", "B42", "B43", "C42", "C43", "D42", "D43")
    for k in cells:
        cell_scores: Dict[str, float] = {"HS": hs[k], "WG": wg[k]}
        if mmlu is not None and k in mmlu:
            cell_scores["MMLU"] = mmlu[k]
        if gsm8k is not None and k in gsm8k:
            cell_scores["GSM8K"] = gsm8k[k]
        scores[k] = cell_scores
    return {
        "schema_version": "d193_b4_t1_input_v1",
        "run_8_completion_marker": "C43",
        "scores": scores,
    }


def _write_input(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _build_uniform_delta_scores(
    delta_hs: float,
    delta_wg: float,
    base_hs: float = 50.0,
    base_wg: float = 60.0,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Build 8 scores where each cell has (seed42, seed43) pair with given deltas.

    Returns (hs_dict, wg_dict) keyed by 'A42','A43',...,'D43'.

    With uniform delta across cells, per-cell var = delta^2 / 2 (N=2 closed form),
    pooled var = delta^2 / 2 (mean of identical per-cell vars), so
    σ̂_pooled = |delta| / sqrt(2). To target σ̂_pooled = X, set delta = X * sqrt(2).
    """
    hs: Dict[str, float] = {}
    wg: Dict[str, float] = {}
    for cell in ("A", "B", "C", "D"):
        hs[f"{cell}42"] = base_hs
        hs[f"{cell}43"] = base_hs + delta_hs
        wg[f"{cell}42"] = base_wg
        wg[f"{cell}43"] = base_wg + delta_wg
    return hs, wg


# ---------------------------------------------------------------------------
# Test 1: Fire on σ̂_HS = 1.6 → ratio 1.60 > 1.50 → FIRE
# ---------------------------------------------------------------------------


def test_fires_on_hs_sigma_1p6(tmp_path: pathlib.Path) -> None:
    """σ̂_pooled,HS = 1.6 → ratio_HS = 1.60 > 1.50 → trigger fires."""
    target_sigma_hs = 1.6
    delta_hs = target_sigma_hs * math.sqrt(2.0)
    hs, wg = _build_uniform_delta_scores(delta_hs=delta_hs, delta_wg=0.0)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, _consolidated_input(hs, wg))

    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    assert rc == 0
    decision = json.loads(output_path.read_text())
    assert math.isclose(decision["pooled_sigma_HS"], 1.6, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(decision["ratio_HS"], 1.60, rel_tol=1e-9, abs_tol=1e-9)
    assert decision["trigger_fires"] is True
    assert decision["decision"] == "ESCALATE_N4"


# ---------------------------------------------------------------------------
# Test 2: No fire on σ̂_HS = 0.8, σ̂_WG = 1.2
# ---------------------------------------------------------------------------


def test_no_fire_on_hs_0p8_wg_1p2(tmp_path: pathlib.Path) -> None:
    """σ̂_HS=0.8, σ̂_WG=1.2 → ratios 0.80 / 0.857 → max=0.857 < 1.50 → NO FIRE."""
    delta_hs = 0.8 * math.sqrt(2.0)
    delta_wg = 1.2 * math.sqrt(2.0)
    hs, wg = _build_uniform_delta_scores(delta_hs=delta_hs, delta_wg=delta_wg)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, _consolidated_input(hs, wg))

    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    assert rc == 0
    decision = json.loads(output_path.read_text())
    assert math.isclose(decision["pooled_sigma_HS"], 0.8, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(decision["pooled_sigma_WG"], 1.2, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(decision["ratio_HS"], 0.80, rel_tol=1e-9, abs_tol=1e-9)
    expected_ratio_wg = 1.2 / 1.4
    assert math.isclose(decision["ratio_WG"], expected_ratio_wg,
                        rel_tol=1e-9, abs_tol=1e-9)
    assert decision["trigger_fires"] is False
    assert decision["decision"] == "HOLD_N3"


# ---------------------------------------------------------------------------
# Test 3: Boundary at threshold (ratio == 1.50 → NO FIRE; strict >)
# ---------------------------------------------------------------------------


def test_exact_threshold_does_not_fire() -> None:
    """Ratio EXACTLY 1.50 must NOT fire (1h Item 1 verbatim: strict `> 1.50`).

    Because float arithmetic of `delta_hs = sqrt(2 * sigma_target^2)` does NOT
    round-trip exactly to `sigma_target = 1.5` (the +50 base introduces a
    rounding ULP that pushes the recovered ratio to 1.5+epsilon), we test the
    strict-`>` semantics at the evaluator-API level where we can supply
    constructed scores producing an exact pooled variance of 2.25.

    Construction: each cell has scores (0, sqrt_4_5) where sqrt_4_5 is set so
    that var = sqrt_4_5^2 / 2 evaluates to exactly 2.25. Since float square of
    `math.sqrt(4.5)` is NOT exactly 4.5, we instead construct deltas as exactly
    representable rationals: pick s_42=0, s_43=d with d^2 = 4.5 unattainable
    in float — so we pivot: pick var directly via custom test helper that
    bypasses the closed-form delta path.

    The construction must avoid irrational deltas (sqrt(2 * sigma^2)).
    """
    # Construction: pick per-cell deltas such that pooled_var = 9/4 = 2.25
    # (so sigma_pooled = sqrt(2.25) = 1.5 EXACTLY in IEEE-754).
    # Pooled var = mean(per_cell_var) = 2.25 → sum(per_cell_var) = 9.
    # Per-cell var (N=2) = delta^2 / 2 → sum(delta^2) = 18.
    # Use deltas (3, 3, 0, 0): integer deltas, exactly representable.
    # Per-cell vars: 4.5, 4.5, 0, 0. Mean = 9/4 = 2.25. sqrt = 1.5 exact.
    deltas_per_cell = {"A": 3.0, "B": 3.0, "C": 0.0, "D": 0.0}
    scores: Dict[str, Dict[str, float]] = {}
    for cell, d in deltas_per_cell.items():
        scores[f"{cell}42"] = {"HS": 0.0, "WG": 0.0}
        scores[f"{cell}43"] = {"HS": d, "WG": 0.0}

    decision = b4_t1_trigger.evaluate_trigger(
        scores=scores,
        sigma_ref={"HS": 1.0, "WG": 1.4},
        threshold=1.50,
    )
    # Verify the construction produces sigma_pooled exactly 1.5 in float.
    assert decision["pooled_sigma_HS"] == 1.5  # exact float equality
    assert decision["ratio_HS"] == 1.5
    assert decision["max_ratio"] == 1.5
    # STRICT `>` per §8.1: 1.50 > 1.50 is False → NO FIRE.
    assert decision["trigger_fires"] is False
    assert decision["decision"] == "HOLD_N3"


# ---------------------------------------------------------------------------
# Test 4: Just above threshold (ratio = 1.501) → FIRE
# ---------------------------------------------------------------------------


def test_just_above_threshold_fires(tmp_path: pathlib.Path) -> None:
    """ratio = 1.501 → trigger fires (strict `>` boundary companion)."""
    sigma_target = 1.501  # slightly over the threshold
    delta_hs = sigma_target * math.sqrt(2.0)
    hs, wg = _build_uniform_delta_scores(delta_hs=delta_hs, delta_wg=0.0)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, _consolidated_input(hs, wg))

    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    assert rc == 0
    decision = json.loads(output_path.read_text())
    assert decision["ratio_HS"] > 1.50
    assert decision["trigger_fires"] is True
    assert decision["decision"] == "ESCALATE_N4"


# ---------------------------------------------------------------------------
# Test 5: Pooled σ formula correctness against manual + statistics module
# ---------------------------------------------------------------------------


def test_pooled_sigma_formula_against_manual(tmp_path: pathlib.Path) -> None:
    """Heterogeneous deltas; verify pooled σ̂ matches the textbook formula.

    Per-cell variances (N=2, ddof=1): A=0.5, B=2.0, C=4.5, D=8.0.
    Pooled var = mean(0.5, 2.0, 4.5, 8.0) = 3.75.
    σ̂_pooled = sqrt(3.75) ≈ 1.9365.
    Manual cross-check: equivalent to using statistics.variance per cell with
    ddof=1, then averaging.
    """
    deltas = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    # per-cell var = delta^2 / 2 (N=2 closed form)
    expected_per_cell_vars = {c: (d * d) / 2.0 for c, d in deltas.items()}
    expected_pooled_var = sum(expected_per_cell_vars.values()) / 4.0
    expected_pooled_sigma = math.sqrt(expected_pooled_var)

    # Cross-check with statistics.variance (ddof=1 by default).
    for cell, d in deltas.items():
        ref_var = statistics.variance([50.0, 50.0 + d])
        assert math.isclose(ref_var, expected_per_cell_vars[cell],
                            rel_tol=1e-12, abs_tol=1e-12)

    hs: Dict[str, float] = {}
    wg: Dict[str, float] = {}
    for cell, d in deltas.items():
        hs[f"{cell}42"] = 50.0
        hs[f"{cell}43"] = 50.0 + d
        wg[f"{cell}42"] = 60.0
        wg[f"{cell}43"] = 60.0  # all WG vars = 0
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, _consolidated_input(hs, wg))

    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    assert rc == 0
    decision = json.loads(output_path.read_text())
    assert math.isclose(decision["pooled_sigma_HS"], expected_pooled_sigma,
                        rel_tol=1e-12, abs_tol=1e-12)
    # Per-cell variance dict present with the expected values.
    for cell, expected_var in expected_per_cell_vars.items():
        assert math.isclose(decision["per_cell_variance_HS"][cell],
                            expected_var, rel_tol=1e-12, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Test 6: Fewer than 8 scores → FATAL exit 1
# ---------------------------------------------------------------------------


def test_missing_cell_exits_fatal(tmp_path: pathlib.Path,
                                   capsys: pytest.CaptureFixture[str]) -> None:
    """Drop one cell from the consolidated input → exit 1 + FATAL stderr."""
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=0.0)
    payload = _consolidated_input(hs, wg)
    # Drop B43 (one of the run-8 set).
    del payload["scores"]["B43"]
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, payload)

    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "FATAL" in captured.err
    assert "B43" in captured.err
    # No decision file written on FATAL.
    assert not output_path.exists()


def test_only_seven_cells_total_exits_fatal(tmp_path: pathlib.Path,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    """Generic <8 cells path: 7 cells provided → exit 1 + FATAL."""
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=0.0)
    payload = _consolidated_input(hs, wg)
    # Drop two cells, keep 6.
    del payload["scores"]["B43"]
    del payload["scores"]["D42"]
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, payload)
    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "FATAL" in captured.err


# ---------------------------------------------------------------------------
# Test 7: NaN handling → FATAL exit 1, identifies cell-seed
# ---------------------------------------------------------------------------


def test_nan_score_exits_fatal(tmp_path: pathlib.Path,
                                capsys: pytest.CaptureFixture[str]) -> None:
    """NaN in any score → exit 1 + FATAL identifying the cell-seed and benchmark."""
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=0.0)
    hs["C42"] = float("nan")
    payload = _consolidated_input(hs, wg)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    # Use Python's json (allow_nan=True default) to write NaN literally.
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(payload))

    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "FATAL" in captured.err
    assert "C42" in captured.err
    assert "HS" in captured.err
    assert not output_path.exists()


def test_inf_score_exits_fatal(tmp_path: pathlib.Path,
                                capsys: pytest.CaptureFixture[str]) -> None:
    """+inf in any score → exit 1 + FATAL."""
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=0.0)
    wg["A43"] = float("inf")
    payload = _consolidated_input(hs, wg)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(payload))
    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "FATAL" in captured.err
    assert "A43" in captured.err


# ---------------------------------------------------------------------------
# Test 8: Out-of-range scores → WARN, do not fail
# ---------------------------------------------------------------------------


def test_out_of_range_score_warns_does_not_fail(tmp_path: pathlib.Path,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    """Score = 110 (> 100) → WARN to stderr, evaluation still completes."""
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=0.0)
    hs["A42"] = 110.0  # out of range
    payload = _consolidated_input(hs, wg)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, payload)

    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    captured = capsys.readouterr()
    assert rc == 0
    # WARN must appear in stderr.
    assert "WARNING" in captured.err or "WARN" in captured.err
    # Evaluation still produced a valid decision.
    assert output_path.exists()
    decision = json.loads(output_path.read_text())
    assert "warnings" in decision
    assert len(decision["warnings"]) >= 1


def test_negative_score_warns_does_not_fail(tmp_path: pathlib.Path,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    """Score = -5 (< 0) → WARN, evaluation still completes."""
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=0.0)
    wg["D43"] = -5.0
    payload = _consolidated_input(hs, wg)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, payload)
    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "WARN" in captured.err
    assert output_path.exists()


# ---------------------------------------------------------------------------
# Test 9: JSON output schema validity
# ---------------------------------------------------------------------------


def test_output_json_schema_complete(tmp_path: pathlib.Path) -> None:
    """All required fields present in the decision JSON, with correct types."""
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=0.0)
    payload = _consolidated_input(hs, wg)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, payload)
    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    assert rc == 0
    decision = json.loads(output_path.read_text())

    required_fields = {
        "tool", "tool_version", "spec_source",
        "timestamp",
        "trigger_fires", "decision",
        "threshold", "threshold_comparison",
        "df", "n_per_cell", "n_cells",
        "sigma_ref", "sigma_ref_source",
        "pooled_sigma_HS", "pooled_sigma_WG",
        "ratio_HS", "ratio_WG", "max_ratio",
        "per_cell_variance_HS", "per_cell_variance_WG",
        "inputs_summary",
        "warnings",
    }
    assert required_fields.issubset(set(decision.keys()))

    # Type assertions.
    assert isinstance(decision["trigger_fires"], bool)
    assert decision["decision"] in ("HOLD_N3", "ESCALATE_N4")
    assert isinstance(decision["threshold"], float)
    assert decision["threshold_comparison"] == "strict_greater_than"
    assert isinstance(decision["df"], int)
    assert decision["df"] == 4
    assert isinstance(decision["sigma_ref"], dict)
    assert isinstance(decision["pooled_sigma_HS"], float)
    assert isinstance(decision["pooled_sigma_WG"], float)
    assert isinstance(decision["ratio_HS"], float)
    assert isinstance(decision["ratio_WG"], float)
    assert isinstance(decision["max_ratio"], float)
    assert isinstance(decision["per_cell_variance_HS"], dict)
    assert isinstance(decision["per_cell_variance_WG"], dict)
    assert isinstance(decision["inputs_summary"], dict)
    assert isinstance(decision["warnings"], list)

    # Tool identity.
    assert decision["tool"] == "b4_t1_trigger"
    assert decision["tool_version"] == "d193_b4_t1_trigger_v1"
    assert "§8.1" in decision["spec_source"] and "§9.1" in decision["spec_source"]

    # All 8 cells echoed in inputs_summary with both HS and WG.
    expected_cells = {"A42", "A43", "B42", "B43", "C42", "C43", "D42", "D43"}
    assert set(decision["inputs_summary"].keys()) == expected_cells
    for cell, payload_inner in decision["inputs_summary"].items():
        assert "HS" in payload_inner
        assert "WG" in payload_inner

    # Per-cell variance dicts have keys {A,B,C,D}.
    assert set(decision["per_cell_variance_HS"].keys()) == {"A", "B", "C", "D"}
    assert set(decision["per_cell_variance_WG"].keys()) == {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# Test 10: df = 4 documented in output
# ---------------------------------------------------------------------------


def test_df_in_output_is_4(tmp_path: pathlib.Path) -> None:
    """Output JSON must document df=4 (k=4 cells × (N=2 - 1) per cell)."""
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=0.0)
    payload = _consolidated_input(hs, wg)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, payload)
    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    assert rc == 0
    decision = json.loads(output_path.read_text())
    assert decision["df"] == 4
    assert decision["n_per_cell"] == 2
    assert decision["n_cells"] == 4


# ---------------------------------------------------------------------------
# Test 13: CLI --threshold override
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threshold,sigma_target,expect_fire", [
    (1.0, 0.99, False),    # ratio 0.99 < threshold 1.0 → no fire
    (1.0, 1.01, True),     # ratio 1.01 > threshold 1.0 → fire
    (2.0, 1.99, False),    # ratio 1.99 < threshold 2.0 → no fire
    (2.0, 2.01, True),     # ratio 2.01 > threshold 2.0 → fire
])
def test_cli_threshold_override(tmp_path: pathlib.Path, threshold: float,
                                 sigma_target: float, expect_fire: bool) -> None:
    """Custom --threshold value changes fire/no-fire boundary."""
    delta_hs = sigma_target * math.sqrt(2.0)
    hs, wg = _build_uniform_delta_scores(delta_hs=delta_hs, delta_wg=0.0)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, _consolidated_input(hs, wg))

    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
        "--threshold", str(threshold),
    ])
    assert rc == 0
    decision = json.loads(output_path.read_text())
    assert decision["threshold"] == threshold
    assert decision["trigger_fires"] is expect_fire


# ---------------------------------------------------------------------------
# Test 14: CLI --sigma-ref-{hs,wg} overrides
# ---------------------------------------------------------------------------


def test_cli_sigma_ref_hs_override(tmp_path: pathlib.Path) -> None:
    """Custom --sigma-ref-hs changes ratio_HS denominator."""
    delta_hs = 1.0 * math.sqrt(2.0)  # σ̂_HS = 1.0
    hs, wg = _build_uniform_delta_scores(delta_hs=delta_hs, delta_wg=0.0)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, _consolidated_input(hs, wg))

    # With σ_ref_HS = 0.5, ratio_HS = 1.0 / 0.5 = 2.0 > 1.50 → FIRE.
    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
        "--sigma-ref-hs", "0.5",
    ])
    assert rc == 0
    decision = json.loads(output_path.read_text())
    assert decision["sigma_ref"]["HS"] == 0.5
    assert math.isclose(decision["ratio_HS"], 2.0, rel_tol=1e-9, abs_tol=1e-9)
    assert decision["trigger_fires"] is True


def test_cli_sigma_ref_wg_override(tmp_path: pathlib.Path) -> None:
    """Custom --sigma-ref-wg changes ratio_WG denominator (symmetry test)."""
    delta_wg = 1.0 * math.sqrt(2.0)  # σ̂_WG = 1.0
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=delta_wg)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, _consolidated_input(hs, wg))
    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
        "--sigma-ref-wg", "0.4",
    ])
    assert rc == 0
    decision = json.loads(output_path.read_text())
    assert decision["sigma_ref"]["WG"] == 0.4
    # ratio_WG = 1.0 / 0.4 = 2.5
    assert math.isclose(decision["ratio_WG"], 2.5, rel_tol=1e-9, abs_tol=1e-9)
    assert decision["trigger_fires"] is True


# ---------------------------------------------------------------------------
# Test 15: WG-only excess fires (symmetry)
# ---------------------------------------------------------------------------


def test_wg_only_excess_fires(tmp_path: pathlib.Path) -> None:
    """σ̂_HS = 0.0, σ̂_WG = 1.6 × 1.4 = 2.24 → ratio_WG = 1.60 > 1.50 → FIRE."""
    sigma_target_wg = 1.6 * 1.4  # so ratio_WG = sigma_target_wg / 1.4 = 1.6
    delta_wg = sigma_target_wg * math.sqrt(2.0)
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=delta_wg)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, _consolidated_input(hs, wg))

    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    assert rc == 0
    decision = json.loads(output_path.read_text())
    assert math.isclose(decision["ratio_HS"], 0.0, abs_tol=1e-12)
    assert math.isclose(decision["ratio_WG"], 1.60, rel_tol=1e-9, abs_tol=1e-9)
    assert decision["trigger_fires"] is True
    assert decision["decision"] == "ESCALATE_N4"


# ---------------------------------------------------------------------------
# Test 16: End-to-end CLI subprocess
# ---------------------------------------------------------------------------


def test_cli_subprocess_no_fire(tmp_path: pathlib.Path) -> None:
    """End-to-end subprocess: rc=0, decision file written, stdout summary printed."""
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=0.0)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, _consolidated_input(hs, wg))

    script = pathlib.Path(__file__).resolve().parent.parent / "tools" / "b4_t1_trigger.py"
    res = subprocess.run(
        [sys.executable, str(script),
         "--results-json", str(input_path),
         "--output", str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"stderr={res.stderr}"
    assert output_path.exists()
    # Stdout has the human-readable trigger summary.
    assert "B4 T1 TRIGGER" in res.stdout
    assert "NO FIRE" in res.stdout


def test_cli_subprocess_fires(tmp_path: pathlib.Path) -> None:
    """End-to-end subprocess: trigger fires path."""
    delta_hs = 1.6 * math.sqrt(2.0)
    hs, wg = _build_uniform_delta_scores(delta_hs=delta_hs, delta_wg=0.0)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, _consolidated_input(hs, wg))

    script = pathlib.Path(__file__).resolve().parent.parent / "tools" / "b4_t1_trigger.py"
    res = subprocess.run(
        [sys.executable, str(script),
         "--results-json", str(input_path),
         "--output", str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "FIRES" in res.stdout
    decision = json.loads(output_path.read_text())
    assert decision["decision"] == "ESCALATE_N4"


def test_cli_subprocess_fatal_missing_input(tmp_path: pathlib.Path) -> None:
    """End-to-end FATAL: nonexistent input file → rc=1 + FATAL on stderr."""
    bogus = tmp_path / "does_not_exist.json"
    output_path = tmp_path / "decision.json"
    script = pathlib.Path(__file__).resolve().parent.parent / "tools" / "b4_t1_trigger.py"
    res = subprocess.run(
        [sys.executable, str(script),
         "--results-json", str(bogus),
         "--output", str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 1
    assert "FATAL" in res.stderr
    assert not output_path.exists()


def test_cli_subprocess_fatal_missing_cell(tmp_path: pathlib.Path) -> None:
    """End-to-end FATAL: <8 cells in input → rc=1."""
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=0.0)
    payload = _consolidated_input(hs, wg)
    del payload["scores"]["A42"]
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, payload)

    script = pathlib.Path(__file__).resolve().parent.parent / "tools" / "b4_t1_trigger.py"
    res = subprocess.run(
        [sys.executable, str(script),
         "--results-json", str(input_path),
         "--output", str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 1
    assert "FATAL" in res.stderr
    assert "A42" in res.stderr


# ---------------------------------------------------------------------------
# Test 17 (bonus): malformed input JSON
# ---------------------------------------------------------------------------


def test_malformed_json_exits_fatal(tmp_path: pathlib.Path,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    """Non-JSON input file → exit 1 + FATAL."""
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    input_path.write_text("not valid json {{{")
    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "FATAL" in captured.err


def test_missing_scores_top_level_key_exits_fatal(tmp_path: pathlib.Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    """Top-level JSON missing 'scores' key → exit 1 + FATAL."""
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    input_path.write_text(json.dumps({"schema_version": "x"}))
    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "FATAL" in captured.err


# ---------------------------------------------------------------------------
# Test 18 (bonus): MMLU/GSM8K audit-extras path
# ---------------------------------------------------------------------------


def test_mmlu_gsm8k_audit_extras_present(tmp_path: pathlib.Path) -> None:
    """When MMLU + GSM8K provided in input, they appear in audit_extras
    (not in trigger statistic)."""
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=0.0)
    mmlu = {k: 25.0 + (i % 4) for i, k in enumerate(
        ("A42", "A43", "B42", "B43", "C42", "C43", "D42", "D43"))}
    gsm8k = {k: 5.0 + (i % 3) for i, k in enumerate(
        ("A42", "A43", "B42", "B43", "C42", "C43", "D42", "D43"))}
    payload = _consolidated_input(hs, wg, mmlu=mmlu, gsm8k=gsm8k)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, payload)
    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    assert rc == 0
    decision = json.loads(output_path.read_text())
    assert "audit_extras" in decision
    assert "pooled_sigma_MMLU" in decision["audit_extras"]
    assert "pooled_sigma_GSM8K" in decision["audit_extras"]
    # MMLU/GSM8K must NOT influence trigger.
    assert decision["trigger_fires"] is False  # HS+WG vars are 0


def test_audit_extras_null_when_missing(tmp_path: pathlib.Path) -> None:
    """When MMLU/GSM8K absent from input, audit_extras values are null."""
    hs, wg = _build_uniform_delta_scores(delta_hs=0.0, delta_wg=0.0)
    payload = _consolidated_input(hs, wg)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "decision.json"
    _write_input(input_path, payload)
    rc = b4_t1_trigger.main([
        "--results-json", str(input_path),
        "--output", str(output_path),
    ])
    assert rc == 0
    decision = json.loads(output_path.read_text())
    assert decision["audit_extras"]["pooled_sigma_MMLU"] is None
    assert decision["audit_extras"]["pooled_sigma_GSM8K"] is None
