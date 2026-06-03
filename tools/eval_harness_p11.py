"""P11 evaluation orchestrator — iterates 12 cells x 4 benchmarks via direct adapter pipeline.

Phase B implementation: calls the HSPA adapter (run_eval_in_batches) directly for
MMLU, HellaSwag, and WinoGrande. GSM8K is PARTIAL_BLOCKED per pre-reg §4.5
(generate_until not implemented).

Dataset loaders (load_hellaswag_items, load_winogrande_items) apply left-truncation
per pre-reg §4.1 so items exceeding max_seq_len tokens are truncated, not dropped.

Harness commit pin: 3fa4fd725c8a428710109f1d6c14eda37e95baea (DO NOT CHANGE)

WinoGrande uses acc metric, not acc_norm (R3 co-sign, D-319).
HellaSwag uses acc metric from direct adapter (per-choice logprobs + argmax).
GSM8K: generate_until not implemented (Phase C deferred). Records as PARTIAL_BLOCKED.

CLI:
    python3 tools/eval_harness_p11.py --arm A42 --task hellaswag --limit 20 --smoke-test
    python3 tools/eval_harness_p11.py --checkpoint-base <dir> --output-dir <dir> --tokenizer <tok> --output-json <out>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Repo root + adapter import
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import hspa_lm_eval_adapter as _adapter
from hspa_lm_eval_adapter import HSPALMEvalAdapter, run_eval_in_batches

# Module-level aliases so tests can patch "eval_harness_p11.<name>"
load_mmlu_items = _adapter.load_mmlu_items
_run_benchmark_via_adapter = _adapter.run_eval_in_batches

# ---------------------------------------------------------------------------
# Apparatus constants (hardcoded per apparatus.md §2.3 + pre-reg)
# ---------------------------------------------------------------------------

CELLS: List[str] = [
    "A42", "A43", "A44",
    "B42", "B43", "B44",
    "C42", "C43", "C44",
    "D42", "D43", "D44",
]

# Ordered benchmark list per apparatus §2.3 + pre-reg §4
BENCHMARKS: List[str] = ["mmlu", "hellaswag", "gsm8k", "winogrande"]

# lm-eval task names (harness-canonical)
HARNESS_TASKS: Dict[str, str] = {
    "mmlu": "mmlu",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    # gsm8k excluded from Phase B (generate_until not implemented)
}

SIGMA_GATE_FLOORS: Dict[str, float] = {
    "mmlu": 27.0,
    "hellaswag": 28.0,
    "gsm8k": 5.0,
    "winogrande": 54.0,
}

HARNESS_COMMIT: str = "3fa4fd725c8a428710109f1d6c14eda37e95baea"

BATCH_SIZE: int = 16
MAX_SEQ_LEN: int = 512

# Blocked result template for GSM8K (generate_until not implemented in Phase B)
GSM8K_BLOCKED_RECORD: Dict[str, Any] = {
    "acc": None,
    "acc_norm": None,
    "exact_match": None,
    "stderr": None,
    "n_items": 0,
    "sigma_gate_floor": SIGMA_GATE_FLOORS["gsm8k"],
    "sigma_gate_status": "PARTIAL_BLOCKED_NO_GENERATE",
    "harness_commit": HARNESS_COMMIT,
    "error": (
        "GSM8K requires generate_until (text generation). "
        "Phase C deferred per pre-reg §4.5 exclusion clause. "
        "PARTIAL_BLOCKED_NO_GENERATE."
    ),
}

# Alias: tests reference "GSMI8K_BLOCKED_RESULT" (the typo is the spec — DO NOT fix)
GSMI8K_BLOCKED_RESULT: Dict[str, Any] = GSM8K_BLOCKED_RECORD


# ---------------------------------------------------------------------------
# σ-gate
# ---------------------------------------------------------------------------

def compute_sigma_gate_status(benchmark: str, acc: Optional[float]) -> str:
    """Return 'PASS', 'FAIL', or 'PARTIAL_BLOCKED_NO_GENERATE'."""
    if benchmark == "gsm8k":
        return "PARTIAL_BLOCKED_NO_GENERATE"
    if acc is None:
        return "FAIL"
    floor_pct = SIGMA_GATE_FLOORS[benchmark]
    return "PASS" if (acc * 100.0) >= floor_pct else "FAIL"


# ---------------------------------------------------------------------------
# Per-run output path convention
# ---------------------------------------------------------------------------

def per_run_output_path(
    output_dir: Path,
    cell_id: str,
    benchmark: str,
    smoke: bool,
) -> Path:
    """Return the output JSON path for a single benchmark run.

    Full-run: <output_dir>/<cell_id>_<benchmark>.json
    Smoke:    <output_dir>/smoke/smoke_<cell_id>.json
    """
    if smoke:
        return Path(output_dir) / "smoke" / f"smoke_{cell_id}.json"
    return Path(output_dir) / f"{cell_id}_{benchmark}.json"


# ---------------------------------------------------------------------------
# Direct-adapter record builder
# ---------------------------------------------------------------------------

def build_record(
    cell_id: str,
    benchmark: str,
    raw_result: Dict[str, Any],
    n_items: int,
    harness_commit: str,
) -> Dict[str, Any]:
    """Build one benchmark_scores.json record from a raw adapter batch result.

    raw_result is the dict returned by run_eval_in_batches() or _fake_batch_result()
    with keys: accuracy, stderr, correct, total, wall_clock_s, rss_peak_mb, ...

    For MMLU/HellaSwag/WinoGrande: extracts accuracy as acc; acc_norm is None.
    WinoGrande uses acc only (R3 co-sign; acc_norm absent).
    HellaSwag uses acc from direct adapter (per-choice argmax accuracy).
    GSM8K: returns PARTIAL_BLOCKED_NO_GENERATE regardless of raw_result.

    sigma_gate uses acc for all non-GSM8K benchmarks.
    """
    if benchmark == "gsm8k":
        record = dict(GSM8K_BLOCKED_RECORD)
        record["cell_id"] = cell_id
        record["benchmark"] = "gsm8k"
        record["harness_commit"] = harness_commit
        return record

    acc = raw_result.get("accuracy")
    if acc is not None:
        acc = float(acc)
    stderr = raw_result.get("stderr")
    if stderr is not None:
        stderr = float(stderr)

    sigma_status = compute_sigma_gate_status(benchmark, acc)

    return {
        "cell_id": cell_id,
        "benchmark": benchmark,
        "acc": acc,
        "acc_norm": None,       # direct adapter does not compute acc_norm
        "exact_match": None,    # generate_until only (GSM8K, Phase C)
        "stderr": stderr,
        "n_items": n_items,
        "sigma_gate_floor": SIGMA_GATE_FLOORS[benchmark],
        "sigma_gate_status": sigma_status,
        "harness_commit": harness_commit,
        "wall_clock_s": round(raw_result.get("wall_clock_s", 0.0), 3),
    }


# ---------------------------------------------------------------------------
# Parse harness results dict → benchmark record (legacy simple_evaluate path)
# ---------------------------------------------------------------------------

def _coerce_numeric(val: Any) -> Optional[float]:
    # bootstrap_iters=0 makes harness emit "N/A" for stderr; treat as absent.
    if val is None or val == "N/A":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_harness_metric(
    harness_results: Dict[str, Any],
    task_name: str,
    metric_key: str,
) -> Optional[float]:
    """Extract a metric from harness results dict.

    harness_results["results"][task_name][metric_key] or None if absent.
    The harness also produces "acc,none" / "acc_norm,none" style keys in v0.4.3.
    Try both forms.
    """
    task_metrics = harness_results.get("results", {}).get(task_name, {})
    val = _coerce_numeric(task_metrics.get(metric_key))
    if val is not None:
        return val
    for k, v in task_metrics.items():
        if k.startswith(metric_key + ","):
            return _coerce_numeric(v)
    return None


def _extract_harness_stderr(
    harness_results: Dict[str, Any],
    task_name: str,
    metric_key: str,
) -> Optional[float]:
    """Extract stderr for a metric from harness results dict."""
    task_metrics = harness_results.get("results", {}).get(task_name, {})
    stderr_key = f"{metric_key}_stderr"
    val = _coerce_numeric(task_metrics.get(stderr_key))
    if val is not None:
        return val
    for k, v in task_metrics.items():
        if k.startswith(stderr_key + ","):
            return _coerce_numeric(v)
    return None


def build_record_from_harness(
    cell_id: str,
    benchmark: str,
    harness_results: Dict[str, Any],
    n_items: Optional[int],
    wall_clock_s: float,
) -> Dict[str, Any]:
    """Build one benchmark_scores.json record from simple_evaluate() output.

    For HellaSwag: extracts both acc and acc_norm (binding metric per pre-reg).
    For WinoGrande: extracts acc only (acc_norm absent per R3 co-sign).
    For MMLU: extracts acc; acc_norm not defined in YAML.
    """
    task_name = HARNESS_TASKS.get(benchmark, benchmark)

    # Primary accuracy metric
    acc = _extract_harness_metric(harness_results, task_name, "acc")
    # acc_norm: only HellaSwag YAML defines it; others will return None naturally
    acc_norm = _extract_harness_metric(harness_results, task_name, "acc_norm")
    # Prefer acc_norm stderr for HellaSwag (binding metric); fall back to acc stderr
    stderr_norm = _extract_harness_stderr(harness_results, task_name, "acc_norm")
    stderr_acc = _extract_harness_stderr(harness_results, task_name, "acc")
    stderr = stderr_norm if stderr_norm is not None else stderr_acc

    # σ-gate uses acc_norm for HellaSwag (pre-reg §4.3), acc for others
    gate_metric = acc_norm if (benchmark == "hellaswag" and acc_norm is not None) else acc
    sigma_status = compute_sigma_gate_status(benchmark, gate_metric)

    return {
        "cell_id": cell_id,
        "benchmark": benchmark,
        "acc": acc,
        "acc_norm": acc_norm,
        "exact_match": None,  # generate_until only (GSM8K, Phase C)
        "stderr": stderr,
        "n_items": n_items,
        "sigma_gate_floor": SIGMA_GATE_FLOORS[benchmark],
        "sigma_gate_status": sigma_status,
        "harness_commit": HARNESS_COMMIT,
        "wall_clock_s": round(wall_clock_s, 3),
    }


# ---------------------------------------------------------------------------
# Dataset loaders — left-truncation per pre-reg §4.1
# ---------------------------------------------------------------------------

def _load_dataset_stream(path: str, name: Optional[str] = None, split: Optional[str] = None, streaming: bool = True, **kw: Any):
    """Thin wrapper around datasets.load_dataset for easy test mocking."""
    from datasets import load_dataset  # type: ignore
    return load_dataset(path, name=name, split=split, streaming=streaming, **kw)


def _left_truncate(text: str, max_chars: int) -> str:
    """Left-truncate text to at most max_chars characters (keeps the tail).

    Pre-reg §4.1: items exceeding max_seq_len must be truncated, not dropped.
    Left-truncation preserves the question/end portion (the predictive signal).
    """
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def load_hellaswag_items(
    limit: int = 100,
    max_seq_len: int = MAX_SEQ_LEN,
) -> List[Dict[str, Any]]:
    """Return up to `limit` HellaSwag validation examples with left-truncation.

    Each item: {"prompt": str, "choices": [str, str, str, str], "label": int}
    Left-truncation applied to prompt when len(prompt) > max_seq_len chars (pre-reg §4.1).
    Items are never dropped — truncation applies, not exclusion.
    """
    docs = _load_dataset_stream("Rowan/hellaswag", split="validation", streaming=True)
    items: List[Dict[str, Any]] = []
    for doc in docs:
        if len(items) >= limit:
            break
        prompt = doc.get("ctx") or doc.get("query") or ""
        choices = doc.get("endings") or doc.get("choices") or []
        label = doc.get("label") or doc.get("gold") or 0
        if isinstance(label, str):
            try:
                label = int(label)
            except (ValueError, TypeError):
                label = 0
        # Left-truncation: keep tail if prompt exceeds max_seq_len chars
        prompt = _left_truncate(prompt, max_seq_len)
        items.append({
            "prompt": prompt,
            "choices": list(choices),
            "label": label,
        })
    return items


def load_winogrande_items(
    limit: int = 100,
    max_seq_len: int = MAX_SEQ_LEN,
) -> List[Dict[str, Any]]:
    """Return up to `limit` WinoGrande validation examples with left-truncation.

    Each item: {"prompt": str, "choices": [str, str], "label": int (0 or 1)}
    WinoGrande uses binary minimal-pair format. Left-truncation applied per §4.1.
    """
    docs = _load_dataset_stream("allenai/winogrande", name="winogrande_xl", split="validation", streaming=True)
    items: List[Dict[str, Any]] = []
    for doc in docs:
        if len(items) >= limit:
            break
        sentence = doc.get("sentence", "")
        option1 = doc.get("option1", "")
        option2 = doc.get("option2", "")
        answer = doc.get("answer", "1")
        # answer is "1" or "2" (1-indexed)
        try:
            label = int(answer) - 1  # convert to 0-indexed
        except (ValueError, TypeError):
            label = 0
        # Build prompt with blank placeholder
        prompt = sentence
        prompt = _left_truncate(prompt, max_seq_len)
        items.append({
            "prompt": prompt,
            "choices": [option1, option2],
            "label": label,
        })
    return items


# ---------------------------------------------------------------------------
# Single-arm single-benchmark runner via direct adapter
# ---------------------------------------------------------------------------

def run_benchmark(
    cell_id: str,
    benchmark: str,
    checkpoint: Path,
    tokenizer: Path,
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
    smoke: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one benchmark for one cell via direct adapter pipeline.

    Loads items with the appropriate loader, calls _run_benchmark_via_adapter,
    builds a record, writes per-run JSON to output_dir, and returns the record.

    For GSM8K: returns PARTIAL_BLOCKED_NO_GENERATE immediately (no adapter call).
    Smoke mode: applies limit to dataset loader; writes to smoke/smoke_<cell>.json.
    Full-run mode (smoke=False): no limit applied; writes to <cell>_<benchmark>.json.
    """
    output_dir = Path(output_dir)
    out_path = per_run_output_path(output_dir, cell_id, benchmark, smoke)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # GSM8K: immediately return PARTIAL record — no adapter invocation
    if benchmark == "gsm8k":
        record = build_record(
            cell_id=cell_id,
            benchmark="gsm8k",
            raw_result=GSMI8K_BLOCKED_RESULT,
            n_items=0,
            harness_commit=HARNESS_COMMIT,
        )
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    # Load items for benchmark
    loader_limit = limit  # None = full-N per apparatus G5
    if benchmark == "mmlu":
        items = load_mmlu_items(limit=loader_limit) if loader_limit is not None else load_mmlu_items(limit=14042)
    elif benchmark == "hellaswag":
        items = load_hellaswag_items(limit=loader_limit, max_seq_len=MAX_SEQ_LEN) if loader_limit is not None else load_hellaswag_items(limit=10042, max_seq_len=MAX_SEQ_LEN)
    elif benchmark == "winogrande":
        items = load_winogrande_items(limit=loader_limit, max_seq_len=MAX_SEQ_LEN) if loader_limit is not None else load_winogrande_items(limit=1267, max_seq_len=MAX_SEQ_LEN)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark!r}")

    n_items = len(items)

    # Run via direct adapter pipeline
    import tempfile
    with tempfile.TemporaryDirectory(prefix="p11_bench_") as tmpdir:
        raw_result = _run_benchmark_via_adapter(
            items=items,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            batch_size=batch_size,
            tmpdir=Path(tmpdir),
        )

    record = build_record(
        cell_id=cell_id,
        benchmark=benchmark,
        raw_result=raw_result,
        n_items=n_items,
        harness_commit=HARNESS_COMMIT,
    )
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(
        f"[P11] {cell_id}/{benchmark}: acc={record['acc']} "
        f"status={record['sigma_gate_status']} -> {out_path}",
        file=sys.stderr,
    )
    return record


# ---------------------------------------------------------------------------
# Single-arm single-benchmark runner via simple_evaluate() (legacy)
# ---------------------------------------------------------------------------

def run_benchmark_via_harness(
    cell_id: str,
    benchmark: str,
    checkpoint: Path,
    tokenizer: Path,
    limit: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    debug_token_window: bool = False,
) -> Dict[str, Any]:
    """Run one benchmark for one arm via lm_eval.evaluator.simple_evaluate().

    Returns a benchmark record dict (same schema as benchmark_scores.json).
    For GSM8K, returns the PARTIAL_BLOCKED_NO_GENERATE stub without invoking harness.
    """
    from lm_eval.evaluator import simple_evaluate  # type: ignore

    if benchmark == "gsm8k":
        rec = dict(GSM8K_BLOCKED_RECORD)
        rec["cell_id"] = cell_id
        return rec

    task_name = HARNESS_TASKS[benchmark]

    adapter = HSPALMEvalAdapter(
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        debug_token_window=debug_token_window,
    )

    t0 = time.monotonic()
    print(
        f"[P11] simple_evaluate: cell={cell_id} task={task_name} limit={limit}",
        file=sys.stderr,
    )

    harness_results = simple_evaluate(
        model=adapter,
        tasks=[task_name],
        batch_size=batch_size,
        limit=limit,
        bootstrap_iters=0,  # disable bootstrap stderr for speed (we use binomial)
        log_samples=False,
    )
    wall_clock_s = time.monotonic() - t0

    if harness_results is None:
        raise RuntimeError(
            f"simple_evaluate() returned None for {cell_id}/{benchmark}. "
            "This typically indicates a distributed-rank issue."
        )

    # Extract n_items from harness samples if available
    n_samples = harness_results.get("n-shot", {}).get(task_name)
    if n_samples is None:
        # Fallback: use limit if set, else None (unknown for full run)
        n_samples = limit

    record = build_record_from_harness(
        cell_id=cell_id,
        benchmark=benchmark,
        harness_results=harness_results,
        n_items=n_samples,
        wall_clock_s=wall_clock_s,
    )

    print(
        f"[P11] {cell_id}/{benchmark}: acc={record['acc']} "
        f"acc_norm={record['acc_norm']} status={record['sigma_gate_status']} "
        f"wall={wall_clock_s:.1f}s",
        file=sys.stderr,
    )
    return record


# ---------------------------------------------------------------------------
# Smoke runner: one arm × one (or multiple) tasks × limit items
# ---------------------------------------------------------------------------

def run_smoke(
    arm: str,
    benchmarks: List[str],
    checkpoint_base: Path,
    tokenizer: Path,
    output_dir: Path,
    limit: int,
    batch_size: int = BATCH_SIZE,
    debug_token_window: bool = False,
) -> List[Dict[str, Any]]:
    """Smoke validation: one arm, specified benchmarks, N items.

    Writes per-benchmark JSON to output_dir/smoke_<arm>_<benchmark>_post_phase_b.json.
    Asserts acc_norm non-null for HellaSwag (dispatch acceptance criterion).
    Returns list of records.
    """
    checkpoint = checkpoint_base / arm / "final.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    records: List[Dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for benchmark in benchmarks:
        record = run_benchmark_via_harness(
            cell_id=arm,
            benchmark=benchmark,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            limit=limit,
            batch_size=batch_size,
            debug_token_window=debug_token_window,
        )

        # Acceptance criterion: HellaSwag acc_norm must NOT be null
        if benchmark == "hellaswag":
            if record.get("acc_norm") is None:
                raise AssertionError(
                    f"SMOKE FAIL: acc_norm is null for {arm}/{benchmark}. "
                    "Phase B closure criterion NOT met. "
                    "Harness cannot compute acc_norm without per-choice logprobs."
                )
            print(
                f"[P11] SMOKE PASS: HellaSwag acc_norm={record['acc_norm']:.4f} "
                f"(non-null — Phase B criterion MET)",
                file=sys.stderr,
            )

        out_path = output_dir / f"smoke_{arm}_{benchmark}_post_phase_b.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"[P11] Smoke result written to {out_path}", file=sys.stderr)
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Full run: 12 arms × 3 benchmarks (GSM8K blocked)
# ---------------------------------------------------------------------------

def run_full(
    checkpoint_base: Path,
    output_dir: Path,
    tokenizer: Path,
    output_json: Path,
    batch_size: int = BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """Iterate 12 cells × 4 benchmarks sequentially.

    Returns list of 48 records. Writes benchmark_scores.json atomically.
    Pre-reg G5: no --limit flag — full-N for all benchmarks.
    GSM8K is recorded as PARTIAL_BLOCKED_NO_GENERATE (Phase C deferred).

    Per-cell checkpoint persistence: each (cell_id, benchmark) result is written
    atomically to output_dir/<cell_id>_<benchmark>.json immediately after computation.
    On restart, any cell whose artifact already exists on disk is loaded and skipped,
    limiting rerun cost after mid-run death to at most 1–2h rather than 17–18h.
    """
    records: List[Dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for cell_id in CELLS:
        checkpoint = checkpoint_base / cell_id / "final.ckpt"
        if not checkpoint.exists():
            print(
                f"[P11] WARN: checkpoint not found at {checkpoint} — skipping {cell_id}",
                file=sys.stderr,
            )
            continue

        for benchmark in BENCHMARKS:
            # --- Resume-skip: load existing per-cell artifact if present ---
            cell_path = output_dir / f"{cell_id}_{benchmark}.json"
            if cell_path.exists():
                loaded_record = json.loads(cell_path.read_text(encoding="utf-8"))
                records.append(loaded_record)
                print(
                    f"[P11] resume-skip: loaded {cell_id} × {benchmark} from {cell_path}",
                    file=sys.stderr,
                )
                continue

            record = run_benchmark_via_harness(
                cell_id=cell_id,
                benchmark=benchmark,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                limit=None,  # G5: full-N, no limit
                batch_size=batch_size,
            )
            records.append(record)

            # --- Per-cell atomic write (mirrors at-end aggregate pattern) ---
            tmp_cell_path = cell_path.with_suffix(".tmp.json")
            tmp_cell_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            tmp_cell_path.replace(cell_path)
            print(
                f"[P11] per-cell write: {cell_id} × {benchmark} → {cell_path}",
                file=sys.stderr,
            )

    # Write aggregate JSON atomically (unchanged — canonical output for downstream)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_json.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    tmp_path.replace(output_json)
    print(f"[P11] Full run complete: {len(records)} records -> {output_json}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "P11 evaluation orchestrator — Phase B: direct adapter pipeline for "
            "MMLU, HellaSwag, WinoGrande. GSM8K recorded as PARTIAL_BLOCKED."
        )
    )
    p.add_argument(
        "--arm",
        type=str,
        default=None,
        choices=CELLS,
        help="Cell ID (e.g. A42) for single-arm run or smoke test.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Direct path to a single checkpoint file (smoke test mode).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of items per benchmark (smoke test only; omit for full run).",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Run in smoke-test mode (requires --checkpoint and --task --limit).",
    )
    p.add_argument(
        "--checkpoint-base",
        type=Path,
        default=REPO_ROOT / "data/checkpoints/phase3_factorial",
        help="Base directory containing 12 cell subdirs (full-run mode).",
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=REPO_ROOT / "data/training/tokenizer_32k.bin",
        help="Path to tokenizer_32k.bin.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/eval/p11",
        help="Directory for per-run JSON outputs.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "programs/program_2_dense_vs_moe_sub100m/benchmark_scores.json",
        help="Path for aggregate benchmark_scores.json (full run only).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Alias for --output-json (legacy compat).",
    )
    p.add_argument(
        "--task",
        type=str,
        default=None,
        choices=BENCHMARKS,
        help="Single benchmark (smoke test only).",
    )
    p.add_argument(
        "--debug-token-window",
        action="store_true",
        default=False,
        help="Pass --debug-token-window to binary; emits truncation diagnostics to stderr.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.tokenizer.exists():
        print(f"[P11] BLOCK: tokenizer not found at {args.tokenizer}", file=sys.stderr)
        return 1

    if args.arm is not None and args.limit is not None:
        # Smoke test: single arm, one or all benchmarks, N items
        benchmarks = [args.task] if args.task else ["mmlu", "hellaswag", "winogrande"]
        try:
            records = run_smoke(
                arm=args.arm,
                benchmarks=benchmarks,
                checkpoint_base=args.checkpoint_base,
                tokenizer=args.tokenizer,
                output_dir=args.output_dir,
                limit=args.limit,
                debug_token_window=args.debug_token_window,
            )
        except AssertionError as exc:
            print(f"[P11] SMOKE FAIL: {exc}", file=sys.stderr)
            return 1
        except FileNotFoundError as exc:
            print(f"[P11] BLOCK: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(records, indent=2))
        return 0

    elif args.arm is not None and args.limit is None:
        # Single-arm full run
        checkpoint = args.checkpoint_base / args.arm / "final.ckpt"
        if not checkpoint.exists():
            print(f"[P11] BLOCK: checkpoint not found at {checkpoint}", file=sys.stderr)
            return 1
        benchmarks = [args.task] if args.task else ["mmlu", "hellaswag", "winogrande", "gsm8k"]
        records = []
        for bm in benchmarks:
            record = run_benchmark_via_harness(
                cell_id=args.arm,
                benchmark=bm,
                checkpoint=checkpoint,
                tokenizer=args.tokenizer,
                limit=None,
                debug_token_window=args.debug_token_window,
            )
            records.append(record)
        print(json.dumps(records, indent=2))
        return 0

    else:
        # Full 12-arm run
        if not args.checkpoint_base.exists():
            print(
                f"[P11] BLOCK: checkpoint-base not found at {args.checkpoint_base}",
                file=sys.stderr,
            )
            return 1
        run_full(
            checkpoint_base=args.checkpoint_base,
            output_dir=args.output_dir,
            tokenizer=args.tokenizer,
            output_json=args.output_json,
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
