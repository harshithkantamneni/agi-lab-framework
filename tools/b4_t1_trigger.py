#!/usr/bin/env python3
"""B4 T1 mechanical trigger evaluator (D-193, Phase 3, Program 2).

Evaluates the T1 pooled-σ trigger after run 8 in the locked launch order
A42 → D42 → B42 → C42 → A43 → D43 → B43 → C43, per the binding spec at
``programs/program_2_example/phase3_p6_prereg.md`` §8.1 (1h Item 1
verbatim, locked unanimous D-192) and §9.1 (math_theorist 1f σ-proxy chain).

    Trigger T1 fires if max_{i ∈ {HS, WG}} (σ̂_pooled,i / σ_ref,i) > 1.50
    (strict greater-than; ratio == 1.50 does NOT fire).

Mechanical: no PI sign required (per §8 + §8.1). The launch orchestrator reads
this tool's output JSON `decision` field and routes the run-8 → run-9 transition
accordingly. The escalation cells (A44, D44, B44, C44) are already in the locked
launch order; a fire DOES NOT cause a re-launch but flags the continuation as
the "N=4 escalation" rather than "N=3 hold".

Inputs: a single consolidated `--results-json` file containing the 8 final-
checkpoint scores per benchmark (HS + WG required; MMLU + GSM8K optional, audit
only — they do NOT contribute to the trigger statistic).

Outputs:
  - Stdout: human-readable one-line trigger summary + per-benchmark breakdown.
  - JSON file at `--output` (default `data/runs/phase3_b4_t1_decision.json`):
    structured decision artifact for the orchestrator + audit trail.

Exit codes:
  - 0: evaluation completed successfully (regardless of fire/no-fire).
  - 1: FATAL (missing cells, NaN/Inf scores, malformed input JSON, parse error).

The trigger semantics live in the JSON `decision` field, NOT in the exit code,
to decouple launch-flow control from trigger-state evaluation. Exit code reports
*tool health*; JSON reports *trigger state*.

CLI:
    python3 tools/b4_t1_trigger.py \\
      --results-json data/runs/phase3_factorial_run8_eval.json \\
      --output data/runs/phase3_b4_t1_decision.json \\
      [--sigma-ref-hs 1.0] [--sigma-ref-wg 1.4] [--threshold 1.50]

Spec source: phase3_p6_prereg.md §8.1 / §9.1. Coexists with the pre-existing
informational tool `tools/b4_t1_evaluator.py` (different contract — that one is
exit-0-always; THIS tool is exit-1 on FATAL for the launch orchestrator). See
`data/engineering/d193_b4_t1_trigger_design.md` for the full design contract.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Locked constants (single-source-of-truth references in comments).
# ---------------------------------------------------------------------------

# σ_ref values from prereg §9.1 (math_theorist 1f §3.3 σ-proxy chain).
# Madaan-7B → Pythia-160M → 34.7M-active scale-corrected estimates.
# Source: programs/program_2_example/phase3_p6_prereg.md §9.1
SIGMA_REF: Dict[str, float] = {
    "HS": 1.0,  # HellaSwag, MEDIUM confidence band [0.5, 1.4]
    "WG": 1.4,  # WinoGrande, MEDIUM confidence band [0.7, 1.8]
}

# T1 threshold, locked at 1h Item 1 verbatim (re-stamped at 1j §1 Item B4).
# Strict comparison: ratio > 1.50 fires; ratio == 1.50 does NOT fire.
# Source: programs/program_2_example/phase3_p6_prereg.md §8.1.
T1_THRESHOLD: float = 1.50

# df for the pooled-σ statistic. k=4 cells × (N=2 - 1 per cell) = 4.
# Documented in output JSON for audit reproducibility.
POOLED_DF: int = 4

# Cells × seeds = 4 × 2 = 8 final-checkpoint scores per benchmark.
# Source: prereg §2.1 + §8.1 launch-order spec.
CELLS: Tuple[str, ...] = ("A", "B", "C", "D")
SEEDS: Tuple[int, ...] = (42, 43)
EXPECTED_CELL_SEEDS: Tuple[str, ...] = tuple(
    f"{cell}{seed}" for cell in CELLS for seed in SEEDS
)
# == ('A42', 'A43', 'B42', 'B43', 'C42', 'C43', 'D42', 'D43')

# Required benchmarks (those that contribute to the trigger statistic).
REQUIRED_BENCHMARKS: Tuple[str, ...] = ("HS", "WG")

# Optional audit-only benchmarks (echoed but not in trigger statistic).
AUDIT_BENCHMARKS: Tuple[str, ...] = ("MMLU", "GSM8K")

# Tool identity for the output JSON.
TOOL_NAME: str = "b4_t1_trigger"
TOOL_VERSION: str = "d193_b4_t1_trigger_v1"
SPEC_SOURCE: str = "phase3_p6_prereg.md §8.1 / §9.1"
SIGMA_REF_SOURCE: str = "prereg §9.1"

# Default output path (CLI-overridable).
DEFAULT_OUTPUT_PATH: str = "data/runs/phase3_b4_t1_decision.json"
DEFAULT_INPUT_PATH: str = "data/runs/phase3_factorial_run8_eval.json"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FatalInputError(Exception):
    """FATAL error in input data: missing cells, malformed JSON, NaN, etc.

    Caught at the CLI boundary; converted to a single ``FATAL: ...`` stderr
    line + exit 1 (so the launch orchestrator's && chain halts).
    """


# ---------------------------------------------------------------------------
# Input loading + validation
# ---------------------------------------------------------------------------


def load_results_json(path: str) -> Dict[str, Dict[str, float]]:
    """Load and schema-validate the consolidated results JSON file.

    Returns a flat dict keyed by ``"<cell><seed>"`` (e.g. ``"A42"``) mapping
    to ``{"HS": <float>, "WG": <float>, optionally "MMLU"/"GSM8K": <float>}``.

    Raises FatalInputError on:
      - file does not exist / non-JSON / not a dict,
      - missing top-level 'scores' key,
      - any expected cell-seed missing from scores,
      - any required benchmark key absent / non-numeric,
      - NaN / Inf values in any benchmark.
    """
    if not os.path.isfile(path):
        raise FatalInputError(f"results JSON file not found: {path}")
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise FatalInputError(f"failed to read/parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FatalInputError(
            f"{path}: top-level JSON must be an object, got {type(payload).__name__}"
        )
    if "scores" not in payload:
        raise FatalInputError(
            f"{path}: missing top-level 'scores' key (expected per d193 schema)"
        )
    raw_scores = payload["scores"]
    if not isinstance(raw_scores, dict):
        raise FatalInputError(
            f"{path}: 'scores' must be an object, got {type(raw_scores).__name__}"
        )

    # Verify all 8 expected cells are present.
    missing = [c for c in EXPECTED_CELL_SEEDS if c not in raw_scores]
    if missing:
        raise FatalInputError(
            f"missing cell-seeds in 'scores': {missing} "
            f"(expected all 8 of {list(EXPECTED_CELL_SEEDS)})"
        )

    # Schema-validate each cell.
    out: Dict[str, Dict[str, float]] = {}
    for cell_seed in EXPECTED_CELL_SEEDS:
        cell_data = raw_scores[cell_seed]
        if not isinstance(cell_data, dict):
            raise FatalInputError(
                f"scores[{cell_seed!r}]: must be an object, "
                f"got {type(cell_data).__name__}"
            )
        validated: Dict[str, float] = {}
        # Required benchmarks (HS, WG).
        for bench in REQUIRED_BENCHMARKS:
            if bench not in cell_data:
                raise FatalInputError(
                    f"scores[{cell_seed!r}]: missing required benchmark {bench!r}"
                )
            validated[bench] = _validate_score(
                cell_seed, bench, cell_data[bench]
            )
        # Optional audit-only benchmarks (MMLU, GSM8K). Validate if present.
        for bench in AUDIT_BENCHMARKS:
            if bench in cell_data:
                validated[bench] = _validate_score(
                    cell_seed, bench, cell_data[bench]
                )
        out[cell_seed] = validated
    return out


def _validate_score(cell_seed: str, bench: str, value: Any) -> float:
    """Validate a single benchmark score: numeric + finite."""
    # Reject booleans explicitly: bool is a numeric subclass in Python and
    # would silently coerce to 0/1, which is almost certainly an upstream bug.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FatalInputError(
            f"scores[{cell_seed!r}][{bench!r}]: must be numeric, "
            f"got {type(value).__name__}={value!r}"
        )
    fv = float(value)
    if math.isnan(fv):
        raise FatalInputError(
            f"scores[{cell_seed!r}][{bench!r}]: NaN value (cell={cell_seed}, "
            f"benchmark={bench})"
        )
    if math.isinf(fv):
        raise FatalInputError(
            f"scores[{cell_seed!r}][{bench!r}]: Inf value (cell={cell_seed}, "
            f"benchmark={bench})"
        )
    return fv


def collect_warnings(scores: Dict[str, Dict[str, float]]) -> List[str]:
    """Collect non-fatal warnings (out-of-range scores etc.)."""
    warnings: List[str] = []
    for cell_seed, bench_scores in scores.items():
        for bench, value in bench_scores.items():
            if value < 0.0 or value > 100.0:
                warnings.append(
                    f"out-of-range score: scores[{cell_seed!r}][{bench!r}]={value!r} "
                    f"(expected [0, 100])"
                )
    return warnings


# ---------------------------------------------------------------------------
# Pooled-σ math
# ---------------------------------------------------------------------------


def per_cell_variance_n2(s_42: float, s_43: float) -> float:
    """Sample variance (ddof=1) of N=2 scores at seeds 42 and 43.

    Closed form for N=2: var = (s_42 - s_43)^2 / 2. Algebraically identical to
    statistics.variance([s_42, s_43]).
    """
    delta = s_42 - s_43
    return (delta * delta) / 2.0


def pooled_variance_equal_n(per_cell_vars: List[float]) -> float:
    """Pooled variance across cells with constant N per cell.

    With constant N, the (N-1)-weighted pool reduces to the simple arithmetic
    mean of per-cell variances. df = k * (N - 1) where k = len(per_cell_vars).
    """
    if not per_cell_vars:
        raise ValueError("pooled_variance_equal_n requires ≥1 per-cell variance")
    return sum(per_cell_vars) / len(per_cell_vars)


def compute_pooled_sigma_for_benchmark(
    scores: Dict[str, Dict[str, float]],
    benchmark: str,
) -> Tuple[Dict[str, float], float]:
    """Compute per-cell variances + pooled σ̂ for one benchmark.

    Returns (per_cell_var_dict, pooled_sigma_hat).
    """
    per_cell_var: Dict[str, float] = {}
    for cell in CELLS:
        s_42 = scores[f"{cell}42"][benchmark]
        s_43 = scores[f"{cell}43"][benchmark]
        per_cell_var[cell] = per_cell_variance_n2(s_42, s_43)
    pooled_var = pooled_variance_equal_n(list(per_cell_var.values()))
    return per_cell_var, math.sqrt(pooled_var)


# ---------------------------------------------------------------------------
# Decision evaluation
# ---------------------------------------------------------------------------


def evaluate_trigger(
    scores: Dict[str, Dict[str, float]],
    sigma_ref: Dict[str, float],
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate T1 mechanical trigger; return the in-memory decision dict."""
    per_cell_var: Dict[str, Dict[str, float]] = {}
    pooled_sigma: Dict[str, float] = {}
    ratios: Dict[str, float] = {}

    for bench in REQUIRED_BENCHMARKS:
        cell_vars, sigma_pooled_b = compute_pooled_sigma_for_benchmark(
            scores, bench
        )
        per_cell_var[bench] = cell_vars
        pooled_sigma[bench] = sigma_pooled_b
        ratios[bench] = sigma_pooled_b / sigma_ref[bench]

    max_ratio = max(ratios.values())
    fires = max_ratio > threshold  # STRICT > per §8.1
    decision_label = "ESCALATE_N4" if fires else "HOLD_N3"

    # Audit-only: pooled σ for MMLU/GSM8K if present in all 8 cells.
    audit_extras: Dict[str, Optional[float]] = {}
    for bench in AUDIT_BENCHMARKS:
        if all(bench in scores[cs] for cs in EXPECTED_CELL_SEEDS):
            _, sigma_b = compute_pooled_sigma_for_benchmark(scores, bench)
            audit_extras[f"pooled_sigma_{bench}"] = sigma_b
        else:
            audit_extras[f"pooled_sigma_{bench}"] = None

    warnings = collect_warnings(scores)

    inputs_summary: Dict[str, Dict[str, float]] = {
        cell_seed: dict(scores[cell_seed]) for cell_seed in EXPECTED_CELL_SEEDS
    }

    timestamp = _datetime.datetime.now(_datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    return {
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "spec_source": SPEC_SOURCE,
        "timestamp": timestamp,
        "trigger_fires": fires,
        "decision": decision_label,
        "threshold": float(threshold),
        "threshold_comparison": "strict_greater_than",
        "df": POOLED_DF,
        "n_per_cell": 2,
        "n_cells": len(CELLS),
        "sigma_ref": dict(sigma_ref),
        "sigma_ref_source": SIGMA_REF_SOURCE,
        "pooled_sigma_HS": pooled_sigma["HS"],
        "pooled_sigma_WG": pooled_sigma["WG"],
        "ratio_HS": ratios["HS"],
        "ratio_WG": ratios["WG"],
        "max_ratio": max_ratio,
        "per_cell_variance_HS": per_cell_var["HS"],
        "per_cell_variance_WG": per_cell_var["WG"],
        "inputs_summary": inputs_summary,
        "audit_extras": audit_extras,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Output emission
# ---------------------------------------------------------------------------


def write_decision_json(decision: Dict[str, Any], output_path: str) -> None:
    """Write decision JSON atomically to output_path."""
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(decision, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, output_path)  # atomic on POSIX


def format_human_summary(decision: Dict[str, Any]) -> str:
    """Format the human-readable stdout summary."""
    fires = decision["trigger_fires"]
    max_ratio = decision["max_ratio"]
    threshold = decision["threshold"]
    head = (
        f"B4 T1 TRIGGER: FIRES (max_ratio={max_ratio:.4f} > {threshold:.2f})"
        if fires else
        f"B4 T1 TRIGGER: NO FIRE (max_ratio={max_ratio:.4f} <= {threshold:.2f})"
    )
    breakdown_lines = [
        f"  HellaSwag : pooled_sigma={decision['pooled_sigma_HS']:.4f}pp  "
        f"ratio={decision['ratio_HS']:.4f}  (sigma_ref={decision['sigma_ref']['HS']:.2f})",
        f"  WinoGrande: pooled_sigma={decision['pooled_sigma_WG']:.4f}pp  "
        f"ratio={decision['ratio_WG']:.4f}  (sigma_ref={decision['sigma_ref']['WG']:.2f})",
        f"  decision  : {decision['decision']}  "
        f"(prereg {decision['spec_source']})",
    ]
    if decision["warnings"]:
        breakdown_lines.append(
            f"  warnings  : {len(decision['warnings'])} (see decision JSON)"
        )
    return head + "\n" + "\n".join(breakdown_lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="b4_t1_trigger",
        description=(
            "B4 T1 mechanical trigger evaluator (Phase 3, Program 2). "
            "Evaluates the pooled-sigma trigger after run 8 per phase3_p6_prereg.md "
            "§8.1. Mechanical: no PI sign required. Output JSON's `decision` field "
            "drives the launch orchestrator's run-9 routing."
        ),
    )
    parser.add_argument(
        "--results-json",
        default=DEFAULT_INPUT_PATH,
        help=(
            f"Path to consolidated 8-cell results JSON. "
            f"Default: {DEFAULT_INPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=(
            f"Path to write decision JSON. Default: {DEFAULT_OUTPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--sigma-ref-hs",
        type=float,
        default=SIGMA_REF["HS"],
        help=(
            f"Reference sigma for HellaSwag (pp). Default per prereg §9.1: "
            f"{SIGMA_REF['HS']}"
        ),
    )
    parser.add_argument(
        "--sigma-ref-wg",
        type=float,
        default=SIGMA_REF["WG"],
        help=(
            f"Reference sigma for WinoGrande (pp). Default per prereg §9.1: "
            f"{SIGMA_REF['WG']}"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=T1_THRESHOLD,
        help=(
            f"T1 trigger threshold (max ratio over HS/WG > threshold fires). "
            f"Default per prereg §8.1: {T1_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also dump the full decision JSON to stdout after the human summary.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Build the override-able sigma_ref dict.
    sigma_ref = {
        "HS": float(args.sigma_ref_hs),
        "WG": float(args.sigma_ref_wg),
    }
    threshold = float(args.threshold)

    # FATAL path: any error from input loading exits 1.
    try:
        scores = load_results_json(args.results_json)
    except FatalInputError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — unexpected; surface as FATAL
        print(f"FATAL: unexpected error loading {args.results_json}: {exc}",
              file=sys.stderr)
        return 1

    # Emit any non-fatal warnings to stderr.
    warnings = collect_warnings(scores)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    # Evaluate trigger.
    try:
        decision = evaluate_trigger(scores, sigma_ref, threshold)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: evaluation error: {exc}", file=sys.stderr)
        return 1

    # Write decision JSON.
    try:
        write_decision_json(decision, args.output)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: failed to write {args.output}: {exc}", file=sys.stderr)
        return 1

    # Stdout: human-readable summary.
    print(format_human_summary(decision))
    if args.print_json:
        json.dump(decision, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
