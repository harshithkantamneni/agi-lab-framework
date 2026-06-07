#!/usr/bin/env python3
"""Phase-3 factorial orchestration — serial 12-run launcher under locked prereg.

Drives the locked launch order A42→D42→B42→C42→A43→D43→B43→C43→A44→D44→B44→C44
under PC-3 AMX-serialization (one run at a time; next never starts until the
previous process has fully exited via the run_long.py waiter). After run 8
(C43 the 8th in launch order) it pauses and invokes `tools/b4_t1_trigger.py`
(owned by tooling_engineer; canonical D-193 mechanical evaluator with strict-`>`
boundary semantics matching prereg §8.1) to evaluate the locked B4 T1 σ-overrun
trigger; the trigger's stdout summary + structured JSON `decision` field are
recorded in run_index.json but do NOT alter the launch order — runs 9-12
proceed unconditionally because the locked launch order is pre-committed and
T1 only affects whether the resulting N=4 was strictly necessary or excess
(per prereg §8.4 "escalation is not a re-launch but a continuation"). T1 not
fired also continues runs 9-12 by the same logic.

Contract change history:
  - D-193: original `b4_t1_handoff()` called the now-orphaned
    `tools/b4_t1_evaluator.py` with `--cell-scores <8 paths>` (informational
    contract; exit-0-always; orphan boundary semantics broken — float-precision
    boundary test failed at == threshold).
  - D-194 (P-B4-T1-CONTRACT-RECONCILE): switched to call canonical
    `tools/b4_t1_trigger.py` with `--results-json <one consolidated path>`
    (mechanical contract; exit-1-on-FATAL; prereg-binding strict-`>` semantics).
    Orphan archived to data/archives/2026-04-25/orphan_b4_t1_evaluator/.
    See data/engineering/d194_b4_t1_contract_reconcile.md for the closeout memo.

Spec source-of-truth: programs/program_2_example/phase3_p6_prereg.md
                      §2.1 (locked launch order) + §8 (B4 trigger) + §13 (LR pin).
Runbook:              programs/program_2_example/phase3_p8_launch_apparatus.md.

Idempotent: re-running the orchestration after interruption skips runs whose
journal entry records `state=completed`. Each run's own checkpoint resume is
handled by run_long.py's auto-resume + spec-sentinel; this orchestrator picks
up at the next un-completed cell.

Pre-flight (run ONCE before launching cell 1):
  1. Compute spec_invariants.yaml fingerprint (run_long.compute_spec_fingerprint).
     Compare to a lock-time fingerprint snapshot if one exists in
     data/checkpoints/phase3_factorial/lock_fingerprint.txt; if absent, snapshot
     it on this first preflight (the runbook §preflight clause).
  2. Predict per-cell config_drift outcomes against the on-disk manifest
     using the well-known semantic of src/training/config_drift.c — if any of
     the 12 cells will FATAL on `lr_base` or `weight_seed` or any of the 14
     pinned fields, BLOCK the launch with a precise diagnostic and exit
     non-zero. The runbook documents the resolution path (manifest amendment
     under unanimous PI+Director sign).
  3. For each cell's checkpoint dir, run the spec-sentinel pre-launch check;
     refuse to launch if any cell's dir is non-empty without a matching
     sentinel (P-CHECKPOINT-DIR-ARCHIVE-DISCIPLINE — emit the literal `mv`
     instruction the operator should run, then exit non-zero).

The actual launch step (per cell) delegates to run_long.py with
LAB_PROGRAM=program_2_example AND LAB_CELL=<A|B|C|D> exported,
so the binary's config_drift_assert_or_die() runs at training boot and
FATALs BEFORE step 0 on any of the 14 LR/dynamics fields not matching the
cell-keyed manifest arm. LAB_CELL was added at D-194 (§10.3 of the launch
design doc) alongside the spec_invariants.yaml 4 cell-arm extension; if
LAB_CELL is unset (legacy callers) the binary falls back to model-key arm
match (Phase-2 backward-compat). That is the "defense in depth" layer; this
orchestrator's preflight is the early-warning layer.
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Reuse the existing primitives. Do not re-implement — that is the design
# binding (run_long.py is the launcher primitive; we orchestrate it).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import run_long  # noqa: E402  pylint: disable=wrong-import-position


# --------------------------------------------------------------------------
# Locked launch matrix (prereg §2.1; do not modify)
# --------------------------------------------------------------------------

PROGRAM_NAME = "program_2_example"

# Cell identity (architecture, LR). cell_letter ∈ {A, B, C, D} per prereg
# §2.1 table; the order in this dict is the *natural* cell-order, NOT the
# launch order (LAUNCH_ORDER below carries that).
# Per-cell perf bands (post-warmup mean step_time, ms). Sourced from
# question.md §"Operational form" measured medians (D-117 amendment):
#   Dense-A: 8200-8800 ms (4 distinct medians: 8301/8650/8786/8394)
#   MoE Rev-2: 9043 ms (single median; widen ±25% since N=1)
# Bands are ±20% from midpoint to catch 2× regressions while tolerating
# normal thermal/AMX/macOS-scheduler jitter. Watcher in tools/run_long.py
# fires FATAL (rc=4) if the post-warmup mean falls outside the band.
# D-248 (id-collision rename of original D-243; commit hash 4156768
# authoritative): closes the silent-kernel-regression gap —
# could waste a 6.1d factorial. See procedures.md §"Compute commitment
# authority" Pre-launch Check #2 (post-launch step-time sentinel).
_DENSE_BAND_MS = (6800.0, 10500.0)   # midpoint 8500, ±~22%
_MOE_BAND_MS   = (7000.0, 11500.0)   # midpoint 9043, ±~28%

CELLS = {
    "A": {"model": "dense50m", "lr": 0.001, "seq_len": 512, "default_moe": False, "label": "Dense-A LR=1e-3", "perf_band_ms": _DENSE_BAND_MS},
    "B": {"model": "dense50m", "lr": 0.002, "seq_len": 512, "default_moe": False, "label": "Dense-A LR=2e-3", "perf_band_ms": _DENSE_BAND_MS},
    "C": {"model": "medium",   "lr": 0.001, "seq_len": 512, "default_moe": True,  "label": "MoE Rev-2 LR=1e-3", "perf_band_ms": _MOE_BAND_MS},
    "D": {"model": "medium",   "lr": 0.002, "seq_len": 512, "default_moe": True,  "label": "MoE Rev-2 LR=2e-3", "perf_band_ms": _MOE_BAND_MS},
}
SEEDS = (42, 43, 44)

# §2.1 LOCKED launch order. Tuples (cell, seed). 12 entries. Run 8 (1-indexed)
# = C43 — that is the B4 T1 trigger evaluation point per §8.1. (D-194 doc fix:
# prior comment said "B43" — incorrect; runtime constant B4_T1_EVAL_AFTER_RUN=8
# always pointed at LAUNCH_ORDER[7] which is ('C', 43), so behavior was correct.)
LAUNCH_ORDER = [
    ("A", 42), ("D", 42), ("B", 42), ("C", 42),
    ("A", 43), ("D", 43), ("B", 43), ("C", 43),
    ("A", 44), ("D", 44), ("B", 44), ("C", 44),
]
B4_T1_EVAL_AFTER_RUN = 8  # 1-indexed; after C43 in launch order

# Wall-budget estimates per question.md §1.7 (forwarded into prereg §2.1):
# ~12.6 h MoE + ~11.8 h Dense per run. Total: 12 × ~12 h ≈ 144-216 h.
WALL_HOURS_DENSE = 11.8
WALL_HOURS_MOE   = 12.6


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PHASE3_ROOT       = os.path.join(ROOT, "data", "checkpoints", "phase3_factorial")
RUN_INDEX_PATH    = os.path.join(PHASE3_ROOT, "run_index.json")
LOCK_FP_PATH      = os.path.join(PHASE3_ROOT, "lock_fingerprint.txt")
# D-194 P-B4-T1-CONTRACT-RECONCILE: now points at canonical b4_t1_trigger.py
# (was b4_t1_evaluator.py at D-193; orphan archived). See contract change
# history in module docstring.
B4_TRIGGER_PATH   = os.path.join(ROOT, "tools", "b4_t1_trigger.py")
# Path where the consolidated results-json (canonical input for b4_t1_trigger.py)
# is materialized just before the trigger handoff. Lives alongside the run
# index in the phase3_factorial checkpoint dir for audit colocation.
B4_TRIGGER_INPUT_PATH = os.path.join(PHASE3_ROOT, "b4_t1_input.json")


def cell_run_id(cell, seed):
    """Run-id passed to run_long.py (also used as run_dir name)."""
    return f"phase3_{cell}{seed}"


def cell_ckpt_dir(cell, seed):
    """Per-cell checkpoint dir (used by --checkpoint-dir + run_long sentinel)."""
    return os.path.join(PHASE3_ROOT, f"{cell}{seed}")


def cell_journal_path(cell, seed):
    return os.path.join(cell_ckpt_dir(cell, seed), "journal.json")


def log(msg):
    print(f"[phase3 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Pre-flight: spec_invariants.yaml fingerprint check
# --------------------------------------------------------------------------

def preflight_fingerprint(strict_lock_check):
    """Verify spec_invariants.yaml fingerprint stability.

    On the first orchestration invocation, snapshot the current fingerprint
    into LOCK_FP_PATH; on subsequent invocations, compare against the
    snapshot and refuse if drifted (manifest changed mid-experiment is a
    verdict-breaking condition per prereg §13).
    """
    fp = run_long.compute_spec_fingerprint(PROGRAM_NAME)
    log(f"spec fingerprint: {fp}")

    if not os.path.isfile(LOCK_FP_PATH):
        os.makedirs(PHASE3_ROOT, exist_ok=True)
        with open(LOCK_FP_PATH, "w") as f:
            f.write(fp + "\n")
        log(f"lock-time fingerprint snapshotted to {LOCK_FP_PATH}")
        return fp

    with open(LOCK_FP_PATH) as f:
        locked = f.read().strip()
    if locked != fp:
        log(
            "FATAL: spec_invariants.yaml has drifted from the lock-time "
            f"snapshot in {LOCK_FP_PATH}. locked='{locked}' current='{fp}'. "
            "Phase-3 prereg §13 binds the manifest at lock time; a mid-"
            "experiment manifest change is verdict-breaking."
        )
        if strict_lock_check:
            sys.exit(4)
    else:
        log(f"fingerprint matches lock-time snapshot ({locked}).")
    return fp


# --------------------------------------------------------------------------
# Pre-flight: predict per-cell config_drift outcome
# --------------------------------------------------------------------------

def _parse_manifest():
    """Parse spec_invariants.yaml for orchestration-side prediction.

    The C-side parser is the source of truth for actual gating — this is a
    PREDICTIVE preflight for early-warning. Format mirrors config_drift.c's
    parser: `arm = <name>` opens a section; `key = value` lines belong to it;
    `#` comments end-of-line.
    """
    path = os.path.join(ROOT, "programs", PROGRAM_NAME, "spec_invariants.yaml")
    arms = {}
    current = None
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if key == "arm":
                current = val
                arms[current] = {"_arm_name": current}
            elif current is not None:
                arms[current][key] = val
    return arms


def predict_cell_drift_or_block(arms):
    """Predict the binary's config_drift verdict for each of the 12 cells.

    Re-implements the comparator semantic at parser-level to flag cells that
    will FATAL pre-step-0. We surface ALL conflicts at once so Director sees
    the full diagnostic surface without iterating 12 launches.

    Returns dict {(cell, seed): [conflict_str, ...]}; empty list means a
    given cell is predicted to PASS the binary-side comparator.
    """
    blocks = {}
    for cell_letter, seed in LAUNCH_ORDER:
        cell = CELLS[cell_letter]
        # D-194 §10.3: prefer cell_id-key match (mirrors C-side LAB_CELL gate).
        # Fall back to model-key match for legacy/cell-unaware manifests
        # (preserves Phase-2 backward-compat — same fallback the binary uses
        # when LAB_CELL is unset; see config_drift.c assert decision tree).
        arm = None
        for name, fields in arms.items():
            if fields.get("cell_id") == cell_letter:
                arm = fields
                break
        if arm is None:
            for name, fields in arms.items():
                if fields.get("model") == cell["model"]:
                    arm = fields
                    break
        if arm is None:
            blocks[(cell_letter, seed)] = [
                f"no arm in manifest with cell_id='{cell_letter}' "
                f"or model='{cell['model']}'"
            ]
            continue

        conflicts = []
        # lr_base: manifest is double; runtime is scale_compute_lr_base which
        # returns args.lr_override if >0 else 0.001/0.002 keyed on --model.
        # We launch with --lr <cell.lr> so runtime lr_base = cell.lr.
        try:
            mf_lr = float(arm.get("lr_base", "nan"))
        except ValueError:
            mf_lr = float("nan")
        runtime_lr = cell["lr"]
        if abs(mf_lr - runtime_lr) > 1e-9:
            conflicts.append(
                f"lr_base manifest={mf_lr} runtime={runtime_lr} "
                f"(--model={cell['model']} cell={cell_letter})"
            )
        # weight_seed: manifest pins (has_weight_seed=true triggers compare).
        try:
            mf_seed = int(arm.get("weight_seed", "-1"))
        except ValueError:
            mf_seed = -1
        if mf_seed != -1 and mf_seed != seed:
            conflicts.append(
                f"weight_seed manifest={mf_seed} runtime={seed}"
            )
        # D-212 (seq_len-orchestrator-fix): defense-in-depth seq_len check.
        # The binary's config_drift comparator FATALs on seq_len mismatch
        # pre-step-0 (same code site as lr_base/weight_seed); we mirror that
        # at orchestrator-preflight so the next bug-of-this-class is caught
        # BEFORE 30 seconds of training boots and FATALs. Sentinel value
        # "-1" means "not pinned in manifest" (parallel to weight_seed).
        try:
            mf_seq = int(arm.get("seq_len", "-1"))
        except ValueError:
            mf_seq = -1
        runtime_seq = cell["seq_len"]
        if mf_seq != -1 and mf_seq != runtime_seq:
            conflicts.append(
                f"seq_len manifest={mf_seq} runtime={runtime_seq} "
                f"(cell={cell_letter})"
            )
        if conflicts:
            blocks[(cell_letter, seed)] = conflicts
    return blocks


def preflight_drift_prediction(strict):
    """Run the 12-cell drift prediction; if any conflicts, BLOCK with diagnostic."""
    arms = _parse_manifest()
    blocks = predict_cell_drift_or_block(arms)
    if not blocks:
        log("drift-prediction: 12/12 cells predicted to PASS the binary's "
            "config_drift comparator.")
        return True

    log("BLOCKED: drift-prediction surface (orchestrator-side, mirrors "
        "src/training/config_drift.c semantics):")
    for (cell, seed), conflicts in sorted(blocks.items()):
        for c in conflicts:
            log(f"  cell {cell}{seed}: {c}")
    log(
        "Resolution: the locked spec_invariants.yaml (arms = dense50m / medium) "
        "pins single (lr_base, weight_seed) values per arm — but the locked "
        "factorial design (prereg §2.1) sweeps both LR and seed across cells. "
        "Per prereg §13 the variation is intended to flow 'at runtime via CLI '"
        "args flowing through scale_experiment.c' but the binary-side "
        "comparator (config_drift.c:606 weight_seed; :526 lr_base) FATALs "
        "on mismatch when manifest pins the field. This is a documentary-vs-"
        "binary inconsistency surfaced at P8 preflight; resolution is a formal "
        "amendment to spec_invariants.yaml under PI+Director unanimous sign "
        "(either expand to 4 arms with arm-name discriminator, OR omit the "
        "lr_base/weight_seed fields from both arms so has_* flags are false "
        "and the comparator only checks structural fields). Until that "
        "amendment lands, factorial launch is BLOCKED at this preflight."
    )
    if strict:
        sys.exit(5)
    return False


# --------------------------------------------------------------------------
# Pre-flight: per-cell ckpt-dir spec sentinel
# --------------------------------------------------------------------------

def preflight_sentinels(strict):
    """For each of the 12 cells, run the run_long.py spec-sentinel check.

    Refuses to launch if any cell's ckpt-dir is non-empty without a matching
    sentinel — emits the literal `mv` instruction per
    P-CHECKPOINT-DIR-ARCHIVE-DISCIPLINE.
    """
    expected = run_long.compute_spec_fingerprint(PROGRAM_NAME)
    failures = []
    for cell_letter, seed in LAUNCH_ORDER:
        ckpt = cell_ckpt_dir(cell_letter, seed)
        ok, reason = run_long.verify_spec_sentinel(ckpt, PROGRAM_NAME)
        if not ok:
            failures.append((cell_letter, seed, reason))
    if not failures:
        log(f"sentinels: 12/12 cells PASS (all fresh or matching {expected}).")
        return True
    log("BLOCKED: spec-sentinel pre-launch check failed for the following cells:")
    for cell, seed, reason in failures:
        log(f"  cell {cell}{seed}: {reason}")
    if strict:
        sys.exit(6)
    return False


# --------------------------------------------------------------------------
# Per-run journaling
# --------------------------------------------------------------------------

def load_run_index():
    if not os.path.isfile(RUN_INDEX_PATH):
        return {}
    try:
        with open(RUN_INDEX_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def write_run_index(idx):
    os.makedirs(PHASE3_ROOT, exist_ok=True)
    tmp = RUN_INDEX_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=2, sort_keys=True)
    os.replace(tmp, RUN_INDEX_PATH)


def write_cell_journal(cell, seed, payload):
    path = cell_journal_path(cell, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def scan_run_log_for_signals(log_path):
    """Quick post-run scan for FATAL/ERROR/WARNING + NaN/PPL signal counts.

    Reads the per-run stdout.log written by run_long.py and tallies
    operator-readable signals. Not exhaustive — for definitive analysis the
    P9 statistical workflow reads checkpoint files directly.
    """
    counts = {"fatal": 0, "error": 0, "warning": 0, "nan": 0,
              "atexit_clean": False, "final_ppl": None, "final_loss": None}
    if not os.path.isfile(log_path):
        return counts
    try:
        with open(log_path, errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return counts
    # Window to most recent launch header (run_long appends across restarts).
    start = 0
    for i in range(len(lines) - 1, -1, -1):
        if "=== run_long launch" in lines[i]:
            start = i
            break
    for line in lines[start:]:
        u = line.upper()
        if "FATAL" in u:
            counts["fatal"] += 1
        if "ERROR" in u and "FATAL" not in u:
            counts["error"] += 1
        if "WARNING" in u:
            counts["warning"] += 1
        if "nan" in line.lower() and ("loss" in line.lower() or "step" in line.lower()):
            counts["nan"] += 1
        # Best-effort final metric extraction; exact field names depend on
        # scale_experiment.c print format. The orchestrator records what it
        # finds; downstream P9 analyzer reads checkpoint files for canonical
        # values.
        if "final ppl" in line.lower() or "final_ppl" in line.lower():
            try:
                counts["final_ppl"] = float(line.strip().split()[-1])
            except (ValueError, IndexError):
                pass
        if "final loss" in line.lower() or "final_loss" in line.lower():
            try:
                counts["final_loss"] = float(line.strip().split()[-1])
            except (ValueError, IndexError):
                pass
        if "ATEXIT" in u and ("OK" in u or "CLEAN" in u):
            counts["atexit_clean"] = True
    return counts


# --------------------------------------------------------------------------
# Single-cell launch
# --------------------------------------------------------------------------

def build_cell_cmd(cell_letter, seed):
    """Construct the scale_experiment CLI line for a given cell.

    Mirrors run-phase2-pair Makefile target conventions; --lr override + per-
    cell --weight-seed are the per-cell binding mechanism. LAB_PROGRAM is
    exported by the caller (env var, not CLI arg) so config_drift activates.
    """
    cell = CELLS[cell_letter]
    flags = [
        "build/scale_experiment",
        "--model", cell["model"],
        "--tokenizer", "data/training/tokenizer_32k.bin",
        "--backprop",
        "--stream",
        "--weight-seed", str(seed),
        "--lr", repr(cell["lr"]),
        # D-212 (seq_len-orchestrator-fix): spec_invariants.yaml pins seq_len=512
        # uniformly across all 6 arms (4 phase3 cells + 2 legacy). Without this
        # flag the C binary defaults to S=128 and config_drift_assert_or_die()
        # FATALs pre-step-0 with `seq_len expected 512, got 128` (D-212 first
        # launch attempt root cause). Carried per-cell in CELLS so a future
        # spec amendment that varies seq_len per cell is a single-site change.
        "--seq-len", str(cell["seq_len"]),
        "--steps", "5000",
        "--checkpoint-every", "500",
        "--checkpoint-dir",
        os.path.relpath(cell_ckpt_dir(cell_letter, seed), ROOT),
    ]
    if cell["default_moe"]:
        # MoE Rev-2 stack flags (mirrors run-phase2-pair MoE side; question.md
        # Operational form binding for the --default-moe + --entropy-penalty +
        # --temp-anneal triplet).
        flags.extend(["--default-moe", "--entropy-penalty", "--temp-anneal"])
    return " ".join(flags)


def launch_one_cell(cell_letter, seed, dry_run):
    """Launch a single cell via run_long.py; block until child exits.

    Does NOT return until the child process finishes (PC-3 AMX-serialization
    binding — next cell never starts before this one exits). run_long.py's
    own waiter handles SIGHUP-immunity, sentinel write, and rc translation.
    """
    run_id = cell_run_id(cell_letter, seed)
    cmd = build_cell_cmd(cell_letter, seed)
    cell = CELLS[cell_letter]
    eta_h = WALL_HOURS_MOE if cell["default_moe"] else WALL_HOURS_DENSE
    log(f"=== launching cell {cell_letter}{seed} ({cell['label']}); ETA ~{eta_h:.1f}h ===")
    log(f"  cmd: {cmd}")
    if dry_run:
        log("  (--dry-run: not invoking run_long)")
        return 0, {"dry_run": True}

    env = os.environ.copy()
    env["LAB_PROGRAM"] = PROGRAM_NAME
    # D-194 §10.3: pass cell_letter via LAB_CELL so the binary's
    # config_drift_assert_or_die() routes to the cell-keyed arm (cells A & C
    # share model=dense50m but differ in lr_base; same for B & D under
    # model=medium). Without LAB_CELL the binary would fall back to
    # model-key match and conflate the 4 cells into 2 arms.
    env["LAB_CELL"] = cell_letter

    band_min, band_max = cell["perf_band_ms"]
    runner_argv = [
        "python3", "tools/run_long.py",
        "--run-id", run_id,
        "--cmd", cmd,
        "--perf-band-min-ms", str(band_min),
        "--perf-band-max-ms", str(band_max),
        "--perf-band-warmup-steps", "10",
        "--perf-band-window-steps", "50",
    ]
    t_start = time.time()
    rc = subprocess.call(runner_argv, cwd=ROOT, env=env)
    t_end = time.time()

    log_path = os.path.join(ROOT, "data", "runs", run_id, "stdout.log")
    sig = scan_run_log_for_signals(log_path)
    return rc, {
        "rc": rc,
        "cmd": cmd,
        "run_id": run_id,
        "log_path": os.path.relpath(log_path, ROOT),
        "checkpoint_dir": os.path.relpath(cell_ckpt_dir(cell_letter, seed), ROOT),
        "t_start": t_start,
        "t_end": t_end,
        "elapsed_seconds": int(t_end - t_start),
        "elapsed_hours": round((t_end - t_start) / 3600.0, 3),
        "expected_eta_hours": eta_h,
        **sig,
    }


# --------------------------------------------------------------------------
# B4 T1 evaluator handoff (after run 8)
# --------------------------------------------------------------------------

def _build_consolidated_results_json(score_paths):
    """Read each cell's final_scores.json and consolidate into the canonical
    `b4_t1_trigger.py` input schema.

    Output schema (per `tools/b4_t1_trigger.py` `load_results_json`):
        {
          "scores": {
            "A42": {"HS": <float>, "WG": <float>, ...optional MMLU/GSM8K},
            "A43": {...},
            ... (all 8 of A42, A43, B42, B43, C42, C43, D42, D43)
          },
          "_provenance": {  # optional audit echo
            "source_paths": [...],
            "consolidated_at": "<ISO-UTC>",
          }
        }

    Returns the dict (caller writes it atomically). Missing input files are
    flagged by entry "<MISSING>" so the canonical trigger's FATAL path
    surfaces a precise diagnostic. NaN/Inf handling is delegated to the
    canonical trigger (it FATALs on those — single source of truth for
    schema validation).
    """
    scores = {}
    for (cell, seed), path in score_paths:
        cell_seed = f"{cell}{seed}"
        if not os.path.isfile(path):
            # Sentinel value the canonical trigger will FATAL on (missing
            # required keys); preserves precise per-cell diagnostic surface.
            scores[cell_seed] = {"_missing_path": path}
            continue
        try:
            with open(path) as f:
                cell_data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            scores[cell_seed] = {"_parse_error": str(exc), "_path": path}
            continue
        if not isinstance(cell_data, dict):
            scores[cell_seed] = {"_invalid_type": type(cell_data).__name__,
                                  "_path": path}
            continue
        # Pass through only the benchmark keys the canonical schema expects.
        # Required: HS, WG. Optional audit: MMLU, GSM8K.
        passthrough = {}
        for key in ("HS", "WG", "MMLU", "GSM8K"):
            if key in cell_data:
                passthrough[key] = cell_data[key]
        scores[cell_seed] = passthrough
    return {
        "scores": scores,
        "_provenance": {
            "source_paths": [p for _, p in score_paths],
            "consolidated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "consolidator": "run_phase3_factorial.b4_t1_handoff (D-194)",
        },
    }


def b4_t1_handoff(idx):
    """Invoke tooling_engineer's canonical b4_t1_trigger.py after run 8.

    Calling contract (D-194 P-B4-T1-CONTRACT-RECONCILE; supersedes the
    D-193 orphan b4_t1_evaluator.py contract):
      Input args:  --results-json <one consolidated path>
                   The orchestrator materializes the consolidated results
                   JSON at B4_TRIGGER_INPUT_PATH by reading each of the 8
                   completed cells' final_scores.json files and merging
                   into the canonical schema (see _build_consolidated_results_json).
      Stdout:      human-readable one-line summary + per-benchmark breakdown
                   (`B4 T1 TRIGGER: FIRES (max_ratio=...)` or
                   `B4 T1 TRIGGER: NO FIRE (...)`).
      JSON output: structured decision artifact at
                   data/runs/phase3_b4_t1_decision.json (default canonical path),
                   containing the `decision` field (`ESCALATE_N4` or `HOLD_N3`),
                   `trigger_fires` bool, threshold echo, all 8 input scores,
                   per-cell variance, pooled-σ per benchmark, etc.
      Exit code:   0 = evaluation completed (regardless of fire/no-fire).
                   1 = FATAL (missing cells, NaN/Inf scores, malformed input).
                   The orchestrator does NOT halt on fire (that's a launch-flow
                   decision encoded in the JSON, not the exit code) — but it
                   DOES flag rc != 0 in run_index for downstream investigation.

    The orchestrator records the trigger's stdout + decision JSON in
    run_index.json for downstream P9 analysis. Per prereg §8.4, T1-fire/
    no-fire does NOT change the launch sequence — runs 9-12 always proceed.
    The T1 outcome only annotates whether the resulting N=4 cell-size was
    strictly necessary or is excess capacity.
    """
    if not os.path.isfile(B4_TRIGGER_PATH):
        log(f"WARN: {os.path.relpath(B4_TRIGGER_PATH, ROOT)} not present — "
            "tooling_engineer's parallel deliverable. Recording 'trigger-"
            "missing' in run_index and continuing per §8.4.")
        return {
            "stdout": "trigger-missing",
            "rc": -1,
            "t_eval": time.time(),
        }

    # Build the score-path map for the 8 cells completed by run 8.
    score_paths = []
    for cell, seed in LAUNCH_ORDER[:B4_T1_EVAL_AFTER_RUN]:
        sp = os.path.join(cell_ckpt_dir(cell, seed), "final_scores.json")
        score_paths.append(((cell, seed), sp))

    # Consolidate into the canonical schema.
    consolidated = _build_consolidated_results_json(score_paths)
    os.makedirs(os.path.dirname(B4_TRIGGER_INPUT_PATH), exist_ok=True)
    tmp_path = B4_TRIGGER_INPUT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(consolidated, f, indent=2, sort_keys=True)
    os.replace(tmp_path, B4_TRIGGER_INPUT_PATH)  # atomic on POSIX

    log(f"=== B4 T1 trigger evaluation (after run {B4_T1_EVAL_AFTER_RUN}) ===")
    log(f"  trigger : {os.path.relpath(B4_TRIGGER_PATH, ROOT)}")
    log(f"  results : {os.path.relpath(B4_TRIGGER_INPUT_PATH, ROOT)} "
        f"(consolidated from {len(score_paths)} cells)")

    t_eval = time.time()
    try:
        result = subprocess.run(
            ["python3", B4_TRIGGER_PATH,
             "--results-json", B4_TRIGGER_INPUT_PATH],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        stdout_lines = (result.stdout or "").strip().splitlines()
        # Trigger emits 4-5 lines (head + 3-4 breakdown). Capture all for audit.
        log("  T1 verdict:")
        for line in stdout_lines:
            log(f"    {line}")
        return {
            "stdout": "\n".join(stdout_lines),
            "stderr": (result.stderr or "").strip(),
            "rc": result.returncode,
            "t_eval": t_eval,
            "input_path": os.path.relpath(B4_TRIGGER_INPUT_PATH, ROOT),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"  T1 trigger failed: {exc}; recording 'trigger-error' and continuing.")
        return {
            "stdout": "trigger-error",
            "stderr": str(exc),
            "rc": -2,
            "t_eval": t_eval,
        }


# --------------------------------------------------------------------------
# Top-level orchestration
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Phase-3 factorial 12-run orchestrator (locked spec)"
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Run preflights + print launch matrix; do not invoke run_long")
    ap.add_argument("--allow-drift", action="store_true",
                    help="Skip orchestrator-side drift prediction block (NOT recommended; "
                         "binary-side config_drift_assert_or_die still gates each run "
                         "pre-step-0 — but launching anyway burns ~12h per cell on what "
                         "the predictor knows will FATAL).")
    ap.add_argument("--allow-sentinel-fail", action="store_true",
                    help="Skip sentinel pre-launch block (NOT recommended; defeats "
                         "P-CHECKPOINT-DIR-ARCHIVE-DISCIPLINE).")
    ap.add_argument("--allow-fingerprint-drift", action="store_true",
                    help="Skip fingerprint-drift block (NOT recommended; manifest "
                         "drift mid-experiment is verdict-breaking per prereg §13).")
    args = ap.parse_args()

    # ---- Pre-flight ----------------------------------------------------
    log("=" * 60)
    log("Phase-3 factorial orchestration — pre-flight")
    log("=" * 60)
    fp = preflight_fingerprint(strict_lock_check=not args.allow_fingerprint_drift)
    drift_ok = preflight_drift_prediction(strict=not args.allow_drift)
    sent_ok = preflight_sentinels(strict=not args.allow_sentinel_fail)

    log(f"pre-flight verdict: fingerprint={fp} drift_ok={drift_ok} sentinels_ok={sent_ok}")

    if not (drift_ok and sent_ok):
        log("PRE-FLIGHT BLOCKED. See diagnostics above for resolution path. "
            "Exit before any cell launches.")
        if not (args.allow_drift and args.allow_sentinel_fail):
            sys.exit(7)

    # ---- ETA banner ----------------------------------------------------
    total_eta = sum(
        WALL_HOURS_MOE if CELLS[c]["default_moe"] else WALL_HOURS_DENSE
        for c, _ in LAUNCH_ORDER
    )
    log(f"expected total wall: {total_eta:.1f}h ({total_eta/24:.1f}d) "
        f"under PC-3 AMX-serial discipline")
    log(f"launch order: {' -> '.join(c+str(s) for c, s in LAUNCH_ORDER)}")
    log(f"B4 T1 evaluation after run {B4_T1_EVAL_AFTER_RUN} "
        f"({LAUNCH_ORDER[B4_T1_EVAL_AFTER_RUN-1][0]}{LAUNCH_ORDER[B4_T1_EVAL_AFTER_RUN-1][1]})")

    # ---- Serial launch loop --------------------------------------------
    idx = load_run_index()
    if "_meta" not in idx:
        idx["_meta"] = {
            "program": PROGRAM_NAME,
            "lock_fingerprint": fp,
            "launch_order": [f"{c}{s}" for c, s in LAUNCH_ORDER],
            "b4_t1_eval_after_run": B4_T1_EVAL_AFTER_RUN,
            "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        # Defense layer (a): in --dry-run mode, do NOT persist the _meta block.
        # Dry-run is fully read-only past pre-flight; the in-memory `idx` is
        # used only to drive the matrix-print loop below.
        if not args.dry_run:
            write_run_index(idx)

    cumulative_start = time.time()
    for run_idx, (cell, seed) in enumerate(LAUNCH_ORDER, start=1):
        run_key = f"{cell}{seed}"
        existing = idx.get(run_key, {})
        # Defense layer (b): a `state == completed` record from a prior dry-run
        # invocation is NOT a real completion. Real-run mode treats any
        # `dry_run: true` record as "not done" and overwrites it. This guards
        # against residue from a (legacy) dry-run that did write to disk —
        # current dry-run is read-only by layer (a) below — and against any
        # future code path that might re-introduce dry-run side effects.
        if existing.get("state") == "completed" and not existing.get("dry_run"):
            log(f"--- run {run_idx}/12 ({run_key}): already completed; skipping. ---")
            continue
        if existing.get("state") == "completed" and existing.get("dry_run"):
            log(f"--- run {run_idx}/12 ({run_key}): prior record is dry-run "
                f"residue; treating as not-done and proceeding. ---")

        # Defense layer (a): in --dry-run mode, the orchestrator is fully
        # read-only past pre-flight. Print the cell's planned cmd line via
        # launch_one_cell()'s log path, then continue without mutating the
        # run index, the per-cell journal, or any cell ckpt-dir on disk.
        # Rationale: the Makefile contract + this module's docstring promise
        # "no actual training launches; runs preflights and prints the matrix",
        # which the operator (Director) relies on when sequencing dry-run
        # before real-run. Any persistent state from dry-run poisons the
        # idempotent state machine in real-run mode.
        if args.dry_run:
            launch_one_cell(cell, seed, dry_run=True)
            log(f"--- run {run_idx}/12 ({run_key}): DRY-RUN matrix-print only; "
                f"no state written. ---")
            continue

        idx[run_key] = {
            "state": "in_progress",
            "run_index": run_idx,
            "cell": cell,
            "seed": seed,
            "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        write_run_index(idx)

        rc, journal = launch_one_cell(cell, seed, dry_run=False)

        # rc=4 from run_long.py = PERF_DRIFT (perf-band watcher killed child
        # after measuring out-of-band step time). Distinct state value so
        # the Director / future-restart logic can route differently than a
        # generic FAIL: a perf drift means apparatus is sick (kernel
        # regression, AMX contention, thermal throttle) — the launch is
        # blocked until optimization_team re-validates perf_log.md.
        if rc == 4:
            run_state = "perf_drift"
        elif rc == 0:
            run_state = "completed"
        else:
            run_state = "failed"

        idx[run_key].update({
            "state": run_state,
            "ended_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **journal,
        })
        write_cell_journal(cell, seed, idx[run_key])
        write_run_index(idx)

        # Generate the per-cell training digest (TDO Stage D3 / Task 17).
        # Non-fatal: a digester crash must not halt training.
        try:
            import subprocess as _subprocess
            _subprocess.run(
                ["python3", "tools/training_digest.py", run_key,
                 "--phase", "phase3_factorial"],
                cwd=str(ROOT),
                capture_output=True,
                check=False,
                timeout=60,
            )
        except Exception as _digest_exc:
            log(f"[digest] non-fatal: training_digest failed for {run_key}: {_digest_exc}")

        elapsed_total_h = (time.time() - cumulative_start) / 3600.0
        log(f"--- run {run_idx}/12 ({run_key}) finished rc={rc}; "
            f"cumulative elapsed: {elapsed_total_h:.2f}h ---")

        if rc == 4 and not args.dry_run:
            log(f"FAIL: cell {run_key} hit PERF_DRIFT (rc=4). Mean step time "
                f"outside the cell's perf band; watcher killed the child. "
                f"Inspect data/runs/{cell_run_id(cell, seed)}/stdout.log for "
                f"the [perf-band] verdict line. Halting factorial: "
                f"optimization_team must re-validate perf_log.md before relaunch.")
            sys.exit(rc)
        if rc != 0 and not args.dry_run:
            log(f"FAIL: cell {run_key} returned rc={rc}; halting orchestration. "
                "Inspect data/runs/{}/stdout.log; remediate; re-invoke this "
                "tool to resume from the next un-completed cell.".format(
                    cell_run_id(cell, seed)))
            sys.exit(rc)

        # B4 T1 trigger evaluation — happens between run 8 and run 9
        if run_idx == B4_T1_EVAL_AFTER_RUN and not args.dry_run:
            verdict = b4_t1_handoff(idx)
            idx["_meta"]["b4_t1_verdict"] = verdict
            write_run_index(idx)

    # Generate the phase-level training digest aggregate (TDO Stage D3 / Task 17).
    # Non-fatal.
    try:
        import subprocess as _subprocess
        _subprocess.run(
            ["python3", "tools/training_digest_aggregate.py", "phase3_factorial"],
            cwd=str(ROOT),
            capture_output=True,
            check=False,
            timeout=120,
        )
    except Exception as _agg_exc:
        log(f"[digest-aggregate] non-fatal: aggregator failed: {_agg_exc}")

    log("=" * 60)
    log(f"Phase-3 factorial orchestration COMPLETE. "
        f"Total wall: {(time.time()-cumulative_start)/3600.0:.2f}h.")
    log(f"Run index: {os.path.relpath(RUN_INDEX_PATH, ROOT)}")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
