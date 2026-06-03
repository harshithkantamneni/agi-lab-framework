"""Training run digest parser — converts stdout.log + run_index.json into
compact LLM-targeted digests at data/digests/training/<cell_id>.{md,json}.

Per spec: docs/superpowers/specs/2026-05-05-training-digest-observer.md.

Step row schema source-of-truth: src/training/scale_experiment.c:363-371.
8 columns: step | loss (pred + bal + lm) | ppl | gnorm | ent | vn_d | ep | ms

CLI:
    python3 tools/training_digest.py <cell_id> [--phase <phase>] [--regen]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# Ensure the repo root (parent of tools/) is on sys.path so direct invocation
# works the same way pytest does. Mirrors the pattern in tools/queue_scanner.py.
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_ON_PATH = str(_TOOLS_DIR.parent)
if _REPO_ROOT_ON_PATH not in sys.path:
    sys.path.insert(0, _REPO_ROOT_ON_PATH)

REPO = Path(__file__).resolve().parent.parent

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Step row parser
#
# Format from src/training/scale_experiment.c:363-371 (printf format):
#   "%5d  | %8.4f (%6.3f + %6.4f + %6.3f) | %9.2f | %6.3f | %7.5f | %6.4f | %2d | %.0f\n"
# 8 columns: step | loss_total (loss_pred + loss_bal + loss_lm) | ppl | gnorm |
#            ent | vn_d | ep | ms
# Negative components allowed (loss_pred can dip below zero).
# ---------------------------------------------------------------------------

_STEP_ROW_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*"          # step
    r"(-?[\d.]+)\s*"               # loss_total
    r"\(\s*(-?[\d.]+)\s*\+\s*"    # loss_pred
    r"(-?[\d.]+)\s*\+\s*"         # loss_bal
    r"(-?[\d.]+)\s*\)\s*\|\s*"    # loss_lm
    r"(-?[\d.]+)\s*\|\s*"         # ppl
    r"(-?[\d.]+)\s*\|\s*"         # gnorm
    r"(-?[\d.]+)\s*\|\s*"         # ent
    r"(-?[\d.]+)\s*\|\s*"         # vn_d
    r"(\d+)\s*\|\s*"               # ep
    r"(-?[\d.]+)\s*$"              # ms
)


def parse_step_row(line: str) -> dict | None:
    """Parse a single training step row.

    Returns dict with keys:
        step, loss_total, loss_pred, loss_bal, loss_lm,
        ppl, gnorm, ent, vn_d, ep, ms
    Or None if the line doesn't match the format.
    """
    m = _STEP_ROW_RE.match(line)
    if not m:
        return None
    return {
        "step": int(m.group(1)),
        "loss_total": float(m.group(2)),
        "loss_pred": float(m.group(3)),
        "loss_bal": float(m.group(4)),
        "loss_lm": float(m.group(5)),
        "ppl": float(m.group(6)),
        "gnorm": float(m.group(7)),
        "ent": float(m.group(8)),
        "vn_d": float(m.group(9)),
        "ep": int(m.group(10)),
        "ms": float(m.group(11)),
    }


# ---------------------------------------------------------------------------
# Step summary parser (every-500-step >>> Step N summary: ... <<< blocks)
# ---------------------------------------------------------------------------

_STEP_SUMMARY_RE = re.compile(
    r">>>\s*Step\s+(\d+)\s+summary:\s+"
    r"avg_lm=(-?[\d.]+),\s*"
    r"best_ppl=(-?[\d.]+),\s*"
    r"epochs=(\d+),\s*"
    r"tokens=(\d+)K,\s*"
    r"NaN=(\d+)\s*<<<"
)


def parse_step_summary(line: str) -> dict | None:
    """Parse a `>>> Step N summary: ... <<<` block emitted every 500 steps.

    Returns dict with keys: step, avg_lm, best_ppl, epochs, tokens_k, nan
    Or None if the line doesn't match.
    """
    m = _STEP_SUMMARY_RE.search(line)
    if not m:
        return None
    return {
        "step": int(m.group(1)),
        "avg_lm": float(m.group(2)),
        "best_ppl": float(m.group(3)),
        "epochs": int(m.group(4)),
        "tokens_k": int(m.group(5)),
        "nan": int(m.group(6)),
    }


# ---------------------------------------------------------------------------
# MoE load balance parsers
#
# Per-layer line: per-step per-layer EMA of expert load distribution.
#   [load] step=N layer=L K=k ema=[v1,v2,...,vk] min=X max=Y ratio=R
# SUMMARY line: per-step worst-ratio across all layers + gate/kill counters.
#   [load] step=N SUMMARY worst_ratio=R worst_layer=L gate_hit=G kill_hit=K
# ---------------------------------------------------------------------------

_LOAD_PER_LAYER_RE = re.compile(
    r"\[load\]\s+step=(\d+)\s+layer=(\d+)\s+K=(\d+)\s+"
    r"ema=\[([\d.,\s\-]+)\]\s+"
    r"min=(-?[\d.]+)\s+max=(-?[\d.]+)\s+ratio=(-?[\d.]+)"
)
_LOAD_SUMMARY_RE = re.compile(
    r"\[load\]\s+step=(\d+)\s+SUMMARY\s+"
    r"worst_ratio=(-?[\d.]+)\s+worst_layer=(\d+)\s+"
    r"gate_hit=(\d+)\s+kill_hit=(\d+)"
)


def parse_load_per_layer(line: str) -> dict | None:
    """Parse a `[load] step=N layer=L K=k ema=[...] min=X max=Y ratio=R` line.

    Returns dict with keys: step, layer, K, ema (list of floats), min_load,
    max_load, ratio. None for non-match (including [load] SUMMARY lines).
    """
    m = _LOAD_PER_LAYER_RE.search(line)
    if not m:
        return None
    ema_str = m.group(4)
    ema = [float(x.strip()) for x in ema_str.split(",") if x.strip()]
    return {
        "step": int(m.group(1)),
        "layer": int(m.group(2)),
        "K": int(m.group(3)),
        "ema": ema,
        "min_load": float(m.group(5)),
        "max_load": float(m.group(6)),
        "ratio": float(m.group(7)),
    }


def parse_load_summary(line: str) -> dict | None:
    """Parse a `[load] step=N SUMMARY worst_ratio=... gate_hit=... kill_hit=...` line.

    Returns dict with keys: step, worst_ratio, worst_layer, gate_hit, kill_hit.
    None for non-match (including per-layer [load] lines).
    """
    m = _LOAD_SUMMARY_RE.search(line)
    if not m:
        return None
    return {
        "step": int(m.group(1)),
        "worst_ratio": float(m.group(2)),
        "worst_layer": int(m.group(3)),
        "gate_hit": int(m.group(4)),
        "kill_hit": int(m.group(5)),
    }


# ---------------------------------------------------------------------------
# Final summary block parser
#
# Operates on the entire stdout text (or the post-Training-Complete tail).
# Returns dict with model, steps, ppl, loss, streams, verdict, atexit_clean.
# Returns None if "--- Training Complete ---" marker is absent.
# ---------------------------------------------------------------------------

_TRAINING_COMPLETE_MARKER = "--- Training Complete ---"
_ATEXIT_MARKER_RE = re.compile(r"\*\*\*\s*ATEXIT:\s*clean process exit\s*\*\*\*")
_VERDICT_RE = re.compile(r"Verdict:\s+(.+?)(?:\s+--\s+.+?\(([\d.]+)%\s+of\s+random\))?\s*$", re.MULTILINE)

# Field-extraction regexes for the Summary block
_MODEL_RE = re.compile(r"Model:\s*(\S+)\s*\(~([\d.]+)M\s+params\)")
_STEPS_RE = re.compile(r"Steps trained:\s*(\d+)\s*/\s*(\d+)")
_EPOCHS_RE = re.compile(r"Epochs:\s*(\d+)")
_TOKENS_RE = re.compile(r"Tokens trained:\s*(\d+)K")
_INITIAL_PPL_RE = re.compile(r"Initial PPL:\s*([\d.]+)")
_BEST_PPL_RE = re.compile(r"Best PPL:\s*([\d.]+)")
_BEST_LOSS_RE = re.compile(r"Best loss:\s*([\d.]+)")
_RECENT_AVG_LM_RE = re.compile(r"Recent avg LM:\s*([\d.]+)")
_RANDOM_BASELINE_RE = re.compile(r"Random baseline:\s*([\d.]+)")
_NAN_EVENTS_RE = re.compile(r"NaN events:\s*(\d+)\s*\(total\)")

# Stream lines: `[STREAM_NAME] bytes=N`
_STREAM_LINE_RE = re.compile(r"\[(\w+)\]\s+bytes=(\d+)")


def parse_final_summary(text: str) -> dict | None:
    """Parse the post-`--- Training Complete ---` summary block.

    Operates on the full stdout (or its tail). Returns None if the
    'Training Complete' marker is absent. Otherwise extracts every available
    field; missing fields return None for floats/ints, "" for strings.

    `atexit_clean` is True iff the `*** ATEXIT: clean process exit ***`
    marker is present after the Summary block (truncated logs lack it).
    """
    if _TRAINING_COMPLETE_MARKER not in text:
        return None

    # Restrict parsing to text AFTER the Training Complete marker
    tail = text[text.find(_TRAINING_COMPLETE_MARKER):]

    out: dict[str, Any] = {
        "model": None,
        "params_M": None,
        "steps_trained": None,
        "steps_planned": None,
        "epochs": None,
        "tokens_k": None,
        "initial_ppl": None,
        "best_ppl": None,
        "best_loss": None,
        "recent_avg_lm": None,
        "random_baseline_loss": None,
        "nan_events_total": None,
        "streams": {},
        "verdict_line": None,
        "pct_of_random": None,
        "atexit_clean": False,
    }

    if (m := _MODEL_RE.search(tail)):
        out["model"] = m.group(1)
        out["params_M"] = float(m.group(2))
    if (m := _STEPS_RE.search(tail)):
        out["steps_trained"] = int(m.group(1))
        out["steps_planned"] = int(m.group(2))
    if (m := _EPOCHS_RE.search(tail)):
        out["epochs"] = int(m.group(1))
    if (m := _TOKENS_RE.search(tail)):
        out["tokens_k"] = int(m.group(1))
    if (m := _INITIAL_PPL_RE.search(tail)):
        out["initial_ppl"] = float(m.group(1))
    if (m := _BEST_PPL_RE.search(tail)):
        out["best_ppl"] = float(m.group(1))
    if (m := _BEST_LOSS_RE.search(tail)):
        out["best_loss"] = float(m.group(1))
    if (m := _RECENT_AVG_LM_RE.search(tail)):
        out["recent_avg_lm"] = float(m.group(1))
    if (m := _RANDOM_BASELINE_RE.search(tail)):
        out["random_baseline_loss"] = float(m.group(1))
    if (m := _NAN_EVENTS_RE.search(tail)):
        out["nan_events_total"] = int(m.group(1))

    # Stream diagnostics — TOT goes to total_bytes; named streams to <NAME>_bytes
    for m in _STREAM_LINE_RE.finditer(tail):
        name = m.group(1)
        bytes_val = int(m.group(2))
        if name == "TOT":
            out["streams"]["total_bytes"] = bytes_val
        else:
            out["streams"][f"{name}_bytes"] = bytes_val

    # Verdict line + percent-of-random
    if (m := _VERDICT_RE.search(tail)):
        out["verdict_line"] = m.group(1).strip()
        if m.group(2):
            out["pct_of_random"] = float(m.group(2))

    # ATEXIT marker
    out["atexit_clean"] = bool(_ATEXIT_MARKER_RE.search(tail))

    return out


# ---------------------------------------------------------------------------
# Loss trajectory classifier
#
# Inputs: list of step-summary dicts from parse_step_summary().
# Output: dict with shape_class + evidence + key inflection points.
#
# Classes (priority high→low):
#   collapsed          best_ppl at last step > 50% of best_ppl at first step
#   oscillating        avg_lm direction changes ≥4 times
#   plateau_after_N    |Δ avg_lm| < 0.05 for ≥3 consecutive summaries
#   monotone_decreasing best_ppl strictly non-increasing (allow 1 noisy step ≤2% rise)
#   insufficient_data  fewer than 2 summaries
# ---------------------------------------------------------------------------

_PLATEAU_DELTA_THRESHOLD = 0.05    # |Δ avg_lm| < this for plateau
_PLATEAU_MIN_CONSECUTIVE = 3
_OSCILLATION_MIN_DIRECTION_CHANGES = 4
_COLLAPSE_BEST_PPL_FRACTION = 0.5  # last best_ppl > 50% of first → collapsed
_MONOTONE_NOISY_STEP_TOLERANCE = 0.02  # ≤2% rise in best_ppl allowed once


def classify_loss_trajectory(summaries: list[dict]) -> dict:
    """Classify a list of step-summary dicts into a trajectory shape class.

    Returns dict with shape_class, shape_class_evidence (str), and optionally:
        - best_ppl_first_seen_step
        - plateau_start_step (if plateau_after_N)
        - direction_change_count (if oscillating)
    """
    if len(summaries) < 2:
        return {"shape_class": "insufficient_data", "shape_class_evidence": "fewer than 2 step summaries available"}

    sorted_s = sorted(summaries, key=lambda s: s["step"])
    first, last = sorted_s[0], sorted_s[-1]
    best_ppl_series = [s["best_ppl"] for s in sorted_s]
    avg_lm_series = [s["avg_lm"] for s in sorted_s]

    # Track step at which best_ppl first reached its minimum
    min_best_ppl = min(best_ppl_series)
    best_ppl_first_seen_step = next(
        (s["step"] for s in sorted_s if abs(s["best_ppl"] - min_best_ppl) < 1e-6),
        None,
    )

    out: dict[str, Any] = {
        "shape_class": None,
        "shape_class_evidence": "",
        "best_ppl_first_seen_step": best_ppl_first_seen_step,
    }

    # 1. Collapsed
    if first["best_ppl"] > 0 and last["best_ppl"] > _COLLAPSE_BEST_PPL_FRACTION * first["best_ppl"]:
        out["shape_class"] = "collapsed"
        out["shape_class_evidence"] = (
            f"last best_ppl {last['best_ppl']:.2f} > "
            f"{_COLLAPSE_BEST_PPL_FRACTION:.0%} × first best_ppl {first['best_ppl']:.2f} "
            f"(training failed to converge)"
        )
        return out

    # 2. Oscillating
    direction_changes = 0
    prev_dir = 0
    for i in range(1, len(avg_lm_series)):
        d = avg_lm_series[i] - avg_lm_series[i - 1]
        cur_dir = 1 if d > 0 else (-1 if d < 0 else 0)
        if cur_dir != 0 and prev_dir != 0 and cur_dir != prev_dir:
            direction_changes += 1
        if cur_dir != 0:
            prev_dir = cur_dir
    if direction_changes >= _OSCILLATION_MIN_DIRECTION_CHANGES:
        out["shape_class"] = "oscillating"
        out["shape_class_evidence"] = (
            f"avg_lm direction changed {direction_changes} times across "
            f"{len(sorted_s)} summaries (≥{_OSCILLATION_MIN_DIRECTION_CHANGES} threshold)"
        )
        out["direction_change_count"] = direction_changes
        return out

    # 3. Plateau
    plateau_start = None
    consecutive_flat = 1
    for i in range(1, len(avg_lm_series)):
        if abs(avg_lm_series[i] - avg_lm_series[i - 1]) < _PLATEAU_DELTA_THRESHOLD:
            consecutive_flat += 1
            if consecutive_flat >= _PLATEAU_MIN_CONSECUTIVE and plateau_start is None:
                plateau_start = sorted_s[i - consecutive_flat + 1]["step"]
        else:
            consecutive_flat = 1
    if plateau_start is not None:
        out["shape_class"] = "plateau_after_N"
        out["shape_class_evidence"] = (
            f"|Δ avg_lm| < {_PLATEAU_DELTA_THRESHOLD} for "
            f"≥{_PLATEAU_MIN_CONSECUTIVE} consecutive summaries starting at step {plateau_start}"
        )
        out["plateau_start_step"] = plateau_start
        return out

    # 4. Monotone decreasing (allow 1 noisy rise ≤2%)
    noisy_rises = 0
    for i in range(1, len(best_ppl_series)):
        if best_ppl_series[i - 1] <= 0:
            continue
        delta_frac = (best_ppl_series[i] - best_ppl_series[i - 1]) / best_ppl_series[i - 1]
        if delta_frac > _MONOTONE_NOISY_STEP_TOLERANCE:
            noisy_rises += 1
    if noisy_rises <= 1:
        out["shape_class"] = "monotone_decreasing"
        out["shape_class_evidence"] = (
            f"best_ppl monotone non-increasing across {len(sorted_s)} summaries "
            f"({noisy_rises} noisy rises within ≤{_MONOTONE_NOISY_STEP_TOLERANCE:.0%} tolerance)"
        )
        return out

    # Fall-through: more than 1 noisy rise but didn't hit other classes → noisy_no_class
    out["shape_class"] = "noisy_no_class"
    out["shape_class_evidence"] = (
        f"trajectory has {noisy_rises} non-monotone rises but does not meet "
        f"oscillation/plateau/collapse thresholds"
    )
    return out


# ---------------------------------------------------------------------------
# MoE regime classifier
# ---------------------------------------------------------------------------

_MOE_COLLAPSE_RATIO_THRESHOLD = 5.0   # max_load / min_load > 5 → collapsed
_MOE_OK_RATIO_THRESHOLD = 1.5         # worst_ratio < 1.5 = ok
_MOE_BAD_RATIO_THRESHOLD = 2.0        # worst_ratio > 2.0 = bad
_MOE_OSCILLATION_MIN_SWINGS = 3       # ok→bad transitions count
_MOE_UNIFORM_EMA_STD_THRESHOLD = 0.02 # std-dev of per-expert EMA < this → uniform


def classify_moe_regime(per_layer_loads: list[dict], summary_loads: list[dict]) -> dict:
    """Classify MoE balance regime from per-layer + summary load traces.

    Classes (priority high→low):
        n/a            - all entries have K=1 (dense model)
        collapsed      - any layer at any step has ratio > 5
        oscillating    - worst_ratio swings ok→bad ≥3 times
        uniform        - per-expert EMA std-dev < 0.02 throughout
        mostly_uniform - K>1 with mild imbalance, no other class triggered

    Returns dict with: regime_class, regime_class_evidence, gate_hit_total,
    kill_hit_total, imbalance_step_ranges (list of [start, end] for collapsed
    regions).
    """
    out: dict[str, Any] = {
        "regime_class": "n/a",
        "regime_class_evidence": "no MoE data",
        "gate_hit_total": 0,
        "kill_hit_total": 0,
        "imbalance_step_ranges": [],
    }

    if not per_layer_loads:
        return out

    # If all per-layer entries have K=1, it's a dense model → n/a
    all_k1 = all(p.get("K", 1) == 1 for p in per_layer_loads)
    if all_k1:
        return out

    # Aggregate gate/kill counters from summaries
    out["gate_hit_total"] = sum(s.get("gate_hit", 0) for s in summary_loads)
    out["kill_hit_total"] = sum(s.get("kill_hit", 0) for s in summary_loads)

    # 1. Collapsed — any per-layer entry with ratio > threshold
    collapsed_steps = sorted({
        p["step"] for p in per_layer_loads
        if p.get("ratio", 1.0) > _MOE_COLLAPSE_RATIO_THRESHOLD
    })
    if collapsed_steps:
        out["regime_class"] = "collapsed"
        out["regime_class_evidence"] = (
            f"per-layer ratio > {_MOE_COLLAPSE_RATIO_THRESHOLD} at "
            f"{len(collapsed_steps)} step(s); first at step {collapsed_steps[0]}"
        )
        out["imbalance_step_ranges"] = [[collapsed_steps[0], collapsed_steps[-1]]]
        return out

    # 2. Oscillating — count ok→bad transitions in worst_ratio
    swings = 0
    state = None  # "ok" or "bad"
    for s in sorted(summary_loads, key=lambda x: x["step"]):
        wr = s.get("worst_ratio", 1.0)
        if wr < _MOE_OK_RATIO_THRESHOLD:
            new_state = "ok"
        elif wr > _MOE_BAD_RATIO_THRESHOLD:
            new_state = "bad"
        else:
            continue  # in-between, no state change
        if state is not None and state != new_state:
            swings += 1
        state = new_state
    if swings >= _MOE_OSCILLATION_MIN_SWINGS:
        out["regime_class"] = "oscillating"
        out["regime_class_evidence"] = (
            f"worst_ratio swung between ok (<{_MOE_OK_RATIO_THRESHOLD}) and "
            f"bad (>{_MOE_BAD_RATIO_THRESHOLD}) {swings} times"
        )
        return out

    # 3. Uniform — per-expert EMA std-dev < threshold across all per-layer entries
    import statistics
    stds = []
    for p in per_layer_loads:
        ema = p.get("ema", [])
        if len(ema) > 1:
            stds.append(statistics.stdev(ema))
    if stds and max(stds) < _MOE_UNIFORM_EMA_STD_THRESHOLD:
        out["regime_class"] = "uniform"
        out["regime_class_evidence"] = (
            f"max per-layer EMA std-dev {max(stds):.4f} < "
            f"{_MOE_UNIFORM_EMA_STD_THRESHOLD} threshold"
        )
        return out

    # 4. Default for K>1 cases
    out["regime_class"] = "mostly_uniform"
    out["regime_class_evidence"] = (
        f"K>1 model with mild imbalance — no class triggered "
        f"(max EMA std={max(stds):.4f} if any, no oscillation, no collapse)"
        if stds
        else "K>1 model, no per-layer EMA stats available"
    )
    return out


# ---------------------------------------------------------------------------
# NaN classifier
# ---------------------------------------------------------------------------


def classify_nan(raw_count: int, step_summary_cumulative: int,
                 fatal: int, rc: int) -> dict:
    """Classify NaN incidents per D-291 false-positive class semantics.

    Classes:
        none           - both raw and cumulative are 0
        false_positive - raw > 0, fatal == 0, rc == 0 (D-291 class)
        hard_nan       - raw > 0 AND (fatal > 0 OR rc != 0) — actual failure
        unclassified   - inconsistent (raw == 0 but cumulative > 0, etc.)

    Returns dict: classification, classification_reason.
    """
    if raw_count == 0 and step_summary_cumulative == 0:
        return {
            "classification": "none",
            "classification_reason": "no NaN incidents recorded",
        }
    if raw_count > 0 and fatal == 0 and rc == 0:
        return {
            "classification": "false_positive",
            "classification_reason": (
                f"raw_count={raw_count} > 0 AND fatal=0 AND rc=0 "
                f"(D-291 false-positive class — NaN counter fired but training "
                f"trajectory unaffected)"
            ),
        }
    if raw_count > 0 and (fatal > 0 or rc != 0):
        return {
            "classification": "hard_nan",
            "classification_reason": (
                f"raw_count={raw_count} > 0 AND (fatal={fatal} > 0 OR rc={rc} != 0) "
                f"— actual training failure"
            ),
        }
    return {
        "classification": "unclassified",
        "classification_reason": (
            f"inconsistent state: raw_count={raw_count}, "
            f"step_summary_cumulative={step_summary_cumulative}, fatal={fatal}, rc={rc}"
        ),
    }


# ---------------------------------------------------------------------------
# generate_digest — end-to-end pipeline
# ---------------------------------------------------------------------------

from datetime import datetime, timezone

# Anomaly line patterns
_ANOMALY_PATTERNS = [
    (re.compile(r"^FATAL[:\s]"), "FATAL"),
    (re.compile(r"^ERROR[:\s]"), "ERROR"),
    (re.compile(r"^WARNING[:\s]"), "WARNING"),
]

# Config-from-cmd extraction patterns (run_index.json's cmd field)
_CMD_LR_RE = re.compile(r"--lr\s+(\S+)")
_CMD_SEED_RE = re.compile(r"--weight-seed\s+(\d+)")
_CMD_STEPS_RE = re.compile(r"--steps\s+(\d+)")
_CMD_MODEL_RE = re.compile(r"--model\s+(\S+)")


def _parse_config_from_cmd(cmd: str) -> dict:
    out: dict[str, Any] = {}
    if (m := _CMD_MODEL_RE.search(cmd)):
        out["model"] = m.group(1)
    if (m := _CMD_LR_RE.search(cmd)):
        try:
            out["lr"] = float(m.group(1))
        except ValueError:
            pass
    if (m := _CMD_SEED_RE.search(cmd)):
        out["seed"] = int(m.group(1))
    if (m := _CMD_STEPS_RE.search(cmd)):
        out["steps_planned"] = int(m.group(1))
    return out


def _detect_anomalies(stdout_text: str) -> list[dict]:
    out: list[dict] = []
    for line_no, line in enumerate(stdout_text.splitlines(), start=1):
        for pattern, level in _ANOMALY_PATTERNS:
            if pattern.search(line.lstrip()):
                out.append({
                    "line_no": line_no,
                    "level": level,
                    "text": line.strip(),
                })
                break  # one classification per line
    return out


def generate_digest(
    cell_id: str,
    phase: str,
    repo_root: Path | None = None,
) -> dict | None:
    """Read cell's stdout.log + run_index entry, run all parsers + classifiers,
    write JSON sidecar to data/digests/training/<cell_id>.json. Returns the
    digest dict or None on missing inputs.
    """
    root = repo_root if repo_root is not None else REPO

    # Resolve run_index first to get the canonical run_id / log_path
    run_index_path = root / "data" / "checkpoints" / phase / "run_index.json"
    if not run_index_path.exists():
        return None
    run_index = json.loads(run_index_path.read_text())
    cell_entry = run_index.get(cell_id, {})

    # Derive log path: prefer run_index entry's log_path, fall back to
    # conventional naming from run_id, then phase+cell_id.
    log_path: Path | None = None
    if cell_entry.get("log_path"):
        candidate = root / cell_entry["log_path"]
        if candidate.exists():
            log_path = candidate
    if log_path is None:
        run_id = cell_entry.get("run_id", f"{phase}_{cell_id}")
        candidate = root / "data" / "runs" / run_id / "stdout.log"
        if candidate.exists():
            log_path = candidate
    if log_path is None:
        # Final fallback: phase3_<cell_id> convention
        candidate = root / "data" / "runs" / f"phase3_{cell_id}" / "stdout.log"
        if candidate.exists():
            log_path = candidate
    if log_path is None:
        return None

    stdout_text = log_path.read_text()
    log_stat = log_path.stat()

    # Parse all line classes
    step_rows = []
    step_summaries = []
    per_layer_loads = []
    summary_loads = []
    for line in stdout_text.splitlines():
        if (r := parse_step_row(line)) is not None:
            step_rows.append(r)
        if (r := parse_step_summary(line)) is not None:
            step_summaries.append(r)
        if (r := parse_load_per_layer(line)) is not None:
            per_layer_loads.append(r)
        if (r := parse_load_summary(line)) is not None:
            summary_loads.append(r)

    final_summary = parse_final_summary(stdout_text) or {}
    loss_traj = classify_loss_trajectory(step_summaries)
    moe = classify_moe_regime(per_layer_loads, summary_loads)
    nan_class = classify_nan(
        raw_count=cell_entry.get("nan", 0),
        step_summary_cumulative=final_summary.get("nan_events_total", 0) or 0,
        fatal=cell_entry.get("fatal", 0),
        rc=cell_entry.get("rc", 0),
    )
    anomalies = _detect_anomalies(stdout_text)

    config = _parse_config_from_cmd(cell_entry.get("cmd", ""))

    # Compute eta_overshoot_pct if both expected and elapsed are present
    expected = cell_entry.get("expected_eta_hours")
    actual = cell_entry.get("elapsed_hours")
    eta_overshoot_pct = None
    if expected and actual and expected > 0:
        eta_overshoot_pct = round((actual - expected) / expected * 100, 2)
    config["expected_eta_hours"] = expected
    config["actual_hours"] = actual
    config["eta_overshoot_pct"] = eta_overshoot_pct
    if "steps_planned" not in config:
        config["steps_planned"] = final_summary.get("steps_planned")
    config["steps_completed"] = final_summary.get("steps_trained")

    # moe_balance.applicable: True only when regime_class is not "n/a"
    moe_applicable = moe["regime_class"] != "n/a"

    digest = {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell_id,
        "phase": phase,
        "config": config,
        "outcome": {
            "rc": cell_entry.get("rc"),
            "atexit_clean": cell_entry.get("atexit_clean"),
            "fatal": cell_entry.get("fatal"),
            "error": cell_entry.get("error"),
            "warning": cell_entry.get("warning"),
            "verdict_line": final_summary.get("verdict_line"),
            "best_ppl": final_summary.get("best_ppl"),
            "best_loss": final_summary.get("best_loss"),
            "recent_avg_lm": final_summary.get("recent_avg_lm"),
            "initial_ppl": final_summary.get("initial_ppl"),
            "tokens_trained": (final_summary.get("tokens_k") or 0) * 1000,
        },
        "loss_trajectory": {
            **loss_traj,
            "step_summaries": step_summaries,
            "convergence_pct_of_random": final_summary.get("pct_of_random"),
        },
        "moe_balance": {
            "applicable": moe_applicable,
            **moe,
        },
        "entropy_penalty_trace": {
            "applicable": False,  # Task 9 stub — entropy-penalty parsing deferred to v2
        },
        "nan_incidents": {
            "raw_count_run_index": cell_entry.get("nan", 0),
            "step_summary_cumulative": final_summary.get("nan_events_total", 0) or 0,
            **nan_class,
        },
        "stream_diagnostics": final_summary.get("streams", {}),
        "anomalies": anomalies,
        "log_path": str(log_path.relative_to(root)),
        "log_size_bytes": log_stat.st_size,
        "stdout_mtime": datetime.fromtimestamp(log_stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "digest_generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    # Write JSON sidecar
    digest_dir = root / "data" / "digests" / "training"
    digest_dir.mkdir(parents=True, exist_ok=True)
    json_path = digest_dir / f"{cell_id}.json"
    json_path.write_text(json.dumps(digest, indent=2))

    return digest


# ---------------------------------------------------------------------------
# Markdown narrative renderer
# ---------------------------------------------------------------------------


def _human_bytes(n: int | None) -> str:
    """Format byte count as human-readable string (KB / MB / GB)."""
    if n is None:
        return "?"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _format_lr(lr: float | None) -> str:
    """Format learning rate compactly (1e-3 form when round)."""
    if lr is None:
        return "?"
    if lr in (1e-3, 1e-4, 2e-3, 5e-4, 1e-5):
        # Use scientific notation for round numbers
        return f"{lr:g}"
    return f"{lr:g}"


def render_markdown(digest: dict | None) -> str:
    """Render a per-cell digest dict to a compact markdown narrative.

    Returns a str (~1-2 KB) suitable for direct reading by Director / mechanism_extractor.
    Returns an error stub if digest is None.
    """
    if digest is None:
        return ("# Training digest unavailable\n\n"
                "The digest could not be generated (stdout.log missing or run_index "
                "entry not found). Inspect raw artifacts directly or run "
                "`python3 tools/training_digest.py <cell_id> --regen`.\n")

    cell_id = digest.get("cell_id", "?")
    phase = digest.get("phase", "?")
    cfg = digest.get("config", {}) or {}
    outcome = digest.get("outcome", {}) or {}
    lt = digest.get("loss_trajectory", {}) or {}
    moe = digest.get("moe_balance", {}) or {}
    nan = digest.get("nan_incidents", {}) or {}
    streams = digest.get("stream_diagnostics", {}) or {}
    anomalies = digest.get("anomalies", []) or []

    # Title — phase number extracted from phase name (e.g., "phase3_factorial" -> "Phase 3")
    phase_num = re.search(r"phase(\d+)", phase)
    phase_label = f"Phase {phase_num.group(1)}" if phase_num else phase
    model = cfg.get("model", "?")
    lr_str = _format_lr(cfg.get("lr"))
    seed = cfg.get("seed", "?")
    title = f"# Cell {cell_id} — {phase_label} ({model}, lr={lr_str}, seed={seed})"

    # Outcome line
    rc = outcome.get("rc")
    atexit_str = "atexit_clean" if outcome.get("atexit_clean") else "atexit_DIRTY"
    rc_str = f"rc={rc}"
    verdict = (outcome.get("verdict_line") or "?").strip()
    pct = lt.get("convergence_pct_of_random")
    pct_str = f" ({pct}% of random)" if pct is not None else ""
    outcome_emoji = "✅" if rc == 0 and outcome.get("atexit_clean") else "❌"
    outcome_line = f"**Outcome:** {outcome_emoji} {verdict}{pct_str}; {rc_str} {atexit_str}"

    # Best PPL line
    best_ppl = outcome.get("best_ppl")
    best_step = lt.get("best_ppl_first_seen_step")
    if best_ppl is not None and best_step is not None:
        best_line = f"**Best PPL:** {best_ppl:.2f} (first seen step {best_step})"
    elif best_ppl is not None:
        best_line = f"**Best PPL:** {best_ppl:.2f}"
    else:
        best_line = "**Best PPL:** ?"

    # Wall line
    actual_h = cfg.get("actual_hours")
    expected_h = cfg.get("expected_eta_hours")
    over_pct = cfg.get("eta_overshoot_pct")
    if actual_h is not None:
        if expected_h is not None and over_pct is not None:
            sign = "+" if over_pct >= 0 else ""
            wall_line = f"**Wall:** {actual_h:.2f}h (vs ETA {expected_h}h, {sign}{over_pct}%)"
        else:
            wall_line = f"**Wall:** {actual_h:.2f}h"
    else:
        wall_line = "**Wall:** ?"

    # Loss trajectory section
    shape_class = lt.get("shape_class", "?")
    shape_evidence = lt.get("shape_class_evidence", "")
    recent_avg = outcome.get("recent_avg_lm")
    best_loss = outcome.get("best_loss")
    lt_lines = ["## Loss trajectory",
                f"- **Shape class:** {shape_class}"]
    if shape_evidence:
        lt_lines.append(f"  - {shape_evidence}")
    if recent_avg is not None and best_loss is not None:
        lt_lines.append(f"- **Final avg_lm:** {recent_avg:.4f} (recent), best {best_loss:.4f}")
    elif recent_avg is not None:
        lt_lines.append(f"- **Final avg_lm:** {recent_avg:.4f} (recent)")

    # MoE balance section
    if moe.get("applicable"):
        moe_lines = ["## MoE balance",
                     f"- **Regime:** {moe.get('regime_class', '?')}",
                     f"  - {moe.get('regime_class_evidence', '')}"]
        if moe.get("gate_hit_total") or moe.get("kill_hit_total"):
            moe_lines.append(f"- gate_hit={moe.get('gate_hit_total', 0)} · kill_hit={moe.get('kill_hit_total', 0)}")
        if moe.get("imbalance_step_ranges"):
            ranges = ", ".join(f"[{a}–{b}]" for a, b in moe["imbalance_step_ranges"])
            moe_lines.append(f"- Imbalance step ranges: {ranges}")
    else:
        moe_lines = ["## MoE balance", "n/a (dense model)"]

    # NaN incidents
    raw = nan.get("raw_count_run_index", 0)
    cum = nan.get("step_summary_cumulative", 0)
    classification = nan.get("classification", "?")
    reason = nan.get("classification_reason", "")
    nan_lines = ["## NaN incidents",
                 f"- raw_count={raw} / step_summary_cumulative={cum}",
                 f"- **Classification:** {classification}"]
    if reason:
        nan_lines.append(f"  - {reason}")

    # Stream diagnostics — single-line summary
    stream_parts = []
    for k, v in streams.items():
        if k == "total_bytes":
            continue
        name = k.replace("_bytes", "")
        stream_parts.append(f"{name}={_human_bytes(v)}")
    total = streams.get("total_bytes")
    if total is not None:
        stream_parts.append(f"TOT={_human_bytes(total)}")
    streams_line = "## Stream diagnostics\n" + " · ".join(stream_parts) if stream_parts \
        else "## Stream diagnostics\n(no stream data)"

    # Anomalies
    if anomalies:
        anom_lines = ["## Anomalies"]
        for a in anomalies:
            anom_lines.append(
                f"- **{a.get('level', '?')}** @line {a.get('line_no', '?')}: {a.get('text', '?')}"
            )
    else:
        anom_lines = ["## Anomalies", "None."]

    # Footer
    log_path = digest.get("log_path", "?")
    log_size = _human_bytes(digest.get("log_size_bytes"))
    gen_at = digest.get("digest_generated_at", "?")
    footer = f"---\n*Generated by tools/training_digest.py from {log_path} ({log_size}) on {gen_at[:10]}.*"

    sections = [
        title, "",
        outcome_line, best_line, wall_line, "",
        "\n".join(lt_lines), "",
        "\n".join(moe_lines), "",
        "\n".join(nan_lines), "",
        streams_line, "",
        "\n".join(anom_lines), "",
        footer,
    ]
    return "\n".join(sections) + "\n"


def _discover_phase(cell_id: str, root: Path) -> str | None:
    """Scan data/checkpoints/*/run_index.json for the cell. Return phase name or None."""
    ckpt_root = root / "data" / "checkpoints"
    if not ckpt_root.exists():
        return None
    for phase_dir in ckpt_root.iterdir():
        if not phase_dir.is_dir():
            continue
        ri_path = phase_dir / "run_index.json"
        if not ri_path.exists():
            continue
        try:
            ri = json.loads(ri_path.read_text())
        except json.JSONDecodeError:
            continue
        if cell_id in ri:
            return phase_dir.name
    return None


def _digest_is_fresh(json_path: Path, stdout_path: Path) -> bool:
    """Return True if existing digest is at least as new as stdout.log."""
    if not json_path.exists():
        return False
    try:
        digest = json.loads(json_path.read_text())
        gen_at = digest.get("digest_generated_at")
        if not gen_at:
            return False
        gen_ts = datetime.fromisoformat(gen_at.replace("Z", "+00:00")).timestamp()
        stdout_ts = stdout_path.stat().st_mtime
        return gen_ts >= stdout_ts
    except (json.JSONDecodeError, ValueError, OSError):
        return False


def _resolve_stdout_path(cell_id: str, phase: str, root: Path) -> Path | None:
    """Mirror the path resolution in generate_digest()."""
    # Tier 1: run_index entry's log_path
    ri_path = root / "data" / "checkpoints" / phase / "run_index.json"
    if ri_path.exists():
        try:
            ri = json.loads(ri_path.read_text())
            entry = ri.get(cell_id, {})
            if "log_path" in entry:
                p = root / entry["log_path"]
                if p.exists():
                    return p
            run_id = entry.get("run_id")
            if run_id:
                p = root / "data" / "runs" / run_id / "stdout.log"
                if p.exists():
                    return p
        except json.JSONDecodeError:
            pass
    # Tier 3: phase3_<cell_id> convention
    p = root / "data" / "runs" / f"phase3_{cell_id}" / "stdout.log"
    if p.exists():
        return p
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate training digest for a cell.")
    parser.add_argument("cell_id", nargs="?", help="Cell ID (e.g., A42)")
    parser.add_argument("--phase", help="Phase name. Auto-discovered if omitted.")
    parser.add_argument("--regen", action="store_true", help="Force regen even if digest is fresh.")
    args = parser.parse_args(argv)

    if not args.cell_id:
        parser.print_usage(sys.stderr)
        return 2

    root = REPO

    # Resolve phase
    phase = args.phase or _discover_phase(args.cell_id, root)
    if phase is None:
        print(f"error: could not auto-discover phase for cell {args.cell_id!r}; "
              f"pass --phase explicitly", file=sys.stderr)
        return 1

    # Resolve stdout path
    stdout_path = _resolve_stdout_path(args.cell_id, phase, root)
    if stdout_path is None:
        print(f"error: stdout.log not found for cell {args.cell_id!r} in phase {phase!r}",
              file=sys.stderr)
        return 1

    # Mtime staleness check
    json_path = root / "data" / "digests" / "training" / f"{args.cell_id}.json"
    if not args.regen and _digest_is_fresh(json_path, stdout_path):
        # Already fresh — skip
        return 0

    digest = generate_digest(args.cell_id, phase, repo_root=root)
    if digest is None:
        print(f"error: generate_digest returned None for {args.cell_id!r}", file=sys.stderr)
        return 1

    # Write markdown narrative alongside the JSON sidecar
    md = render_markdown(digest)
    md_path = root / "data" / "digests" / "training" / f"{args.cell_id}.md"
    md_path.write_text(md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
