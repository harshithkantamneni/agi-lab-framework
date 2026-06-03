"""Tests for tools/eval_harness_p11.py.

Covers:
- Cell/benchmark iteration logic (12 cells × 4 benchmarks)
- Per-run JSON schema validation (sigma_gate fields, required keys)
- Benchmark score aggregation to benchmark_scores.json (48-record schema)
- Smoke-test mode (single cell, single task, with --limit)
- Output path patterns (data/eval/p11/<cell>_<benchmark>.json, smoke/smoke_<cell>.json)
- σ-gate floor values match apparatus.md §4.3 pre-reg gates
- WinoGrande uses acc metric (not acc_norm) per R3 co-sign
- GSM8K handled as PARTIAL (generate_until not supported by binary)
- CLI argument parsing
- Left-truncation confirmed in dataset loaders
- No --limit flag in full-run invocation (apparatus G5)
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, call
import pytest

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import eval_harness_p11 as harness


# ---------------------------------------------------------------------------
# Constants — locked per apparatus.md §2.3 + pre-reg §4.3
# ---------------------------------------------------------------------------

EXPECTED_CELLS = [
    "A42", "A43", "A44",
    "B42", "B43", "B44",
    "C42", "C43", "C44",
    "D42", "D43", "D44",
]

EXPECTED_BENCHMARKS = ["mmlu", "hellaswag", "gsm8k", "winogrande"]

# σ-gate floors per apparatus.md §4.3 + pre-reg §4.3
SIGMA_GATE_FLOORS = {
    "mmlu": 27.0,
    "hellaswag": 28.0,
    "gsm8k": 5.0,
    "winogrande": 54.0,
}

# Hardcoded seeds per apparatus §2.3
EXPECTED_SEEDS = {42, 43, 44}


# ---------------------------------------------------------------------------
# Smoke-test fixture: fake a successful adapter result
# ---------------------------------------------------------------------------

def _fake_batch_result(acc: float = 0.25, n: int = 100) -> Dict[str, Any]:
    return {
        "accuracy": acc,
        "stderr": round(math.sqrt(acc * (1 - acc) / n), 4),
        "correct": int(acc * n),
        "total": n,
        "wall_clock_s": 1.2,
        "rss_peak_mb": 300.0,
        "per_batch_timings_s": [0.4, 0.4, 0.4],
        "binary_stderr": "ok",
    }


# ---------------------------------------------------------------------------
# 1. Cell and benchmark enumeration
# ---------------------------------------------------------------------------


def test_cell_list_has_12_entries():
    assert len(harness.CELLS) == 12


def test_cell_list_exact_values():
    assert harness.CELLS == EXPECTED_CELLS


def test_benchmark_list_has_4_entries():
    assert len(harness.BENCHMARKS) == 4


def test_benchmark_list_exact_values():
    assert harness.BENCHMARKS == EXPECTED_BENCHMARKS


# ---------------------------------------------------------------------------
# 2. σ-gate floors
# ---------------------------------------------------------------------------


def test_sigma_gate_floor_mmlu():
    assert harness.SIGMA_GATE_FLOORS["mmlu"] == pytest.approx(27.0)


def test_sigma_gate_floor_hellaswag():
    assert harness.SIGMA_GATE_FLOORS["hellaswag"] == pytest.approx(28.0)


def test_sigma_gate_floor_gsm8k():
    assert harness.SIGMA_GATE_FLOORS["gsm8k"] == pytest.approx(5.0)


def test_sigma_gate_floor_winogrande():
    assert harness.SIGMA_GATE_FLOORS["winogrande"] == pytest.approx(54.0)


# ---------------------------------------------------------------------------
# 3. Seeds
# ---------------------------------------------------------------------------


def test_seeds_are_42_43_44():
    # Each cell ID ends in its seed number
    seeds_from_cells = {int(c[-2:]) for c in harness.CELLS}
    assert seeds_from_cells == EXPECTED_SEEDS


# ---------------------------------------------------------------------------
# 4. Sigma-gate status computation
# ---------------------------------------------------------------------------


def test_sigma_gate_status_pass():
    # acc=30% > floor=27% → PASS for MMLU
    status = harness.compute_sigma_gate_status("mmlu", 0.30)
    assert status == "PASS"


def test_sigma_gate_status_fail():
    # acc=25% < floor=27% → FAIL for MMLU
    status = harness.compute_sigma_gate_status("mmlu", 0.25)
    assert status == "FAIL"


def test_sigma_gate_status_exactly_at_floor_is_pass():
    # acc == floor exactly → PASS (boundary: >= is PASS)
    status = harness.compute_sigma_gate_status("mmlu", 0.27)
    assert status == "PASS"


def test_sigma_gate_status_winogrande_acc_not_acc_norm():
    # WinoGrande uses acc (R3 co-sign); floor=54%
    status = harness.compute_sigma_gate_status("winogrande", 0.55)
    assert status == "PASS"
    status_fail = harness.compute_sigma_gate_status("winogrande", 0.50)
    assert status_fail == "FAIL"


# ---------------------------------------------------------------------------
# 5. Per-run record schema
# ---------------------------------------------------------------------------


def test_build_record_schema_mmlu(tmp_path: Path):
    """build_record() returns a dict with all required fields for MMLU."""
    raw = _fake_batch_result(acc=0.28, n=14042)
    record = harness.build_record(
        cell_id="A42",
        benchmark="mmlu",
        raw_result=raw,
        n_items=14042,
        harness_commit=harness.HARNESS_COMMIT,
    )
    # Required fields from apparatus.md §7 row 2 schema
    assert record["cell_id"] == "A42"
    assert record["benchmark"] == "mmlu"
    assert record["acc"] == pytest.approx(0.28)
    assert record["acc_norm"] is None          # MMLU uses acc, not acc_norm
    assert record["exact_match"] is None       # MMLU uses acc
    assert "stderr" in record
    assert record["n_items"] == 14042
    assert record["sigma_gate_floor"] == pytest.approx(27.0)
    assert record["sigma_gate_status"] == "PASS"  # 28% > 27%
    assert record["harness_commit"] == harness.HARNESS_COMMIT


def test_build_record_schema_hellaswag():
    """HellaSwag record uses acc field; acc_norm is None (adapter returns acc)."""
    raw = _fake_batch_result(acc=0.30, n=10042)
    record = harness.build_record(
        cell_id="B43",
        benchmark="hellaswag",
        raw_result=raw,
        n_items=10042,
        harness_commit=harness.HARNESS_COMMIT,
    )
    assert record["benchmark"] == "hellaswag"
    assert record["acc"] == pytest.approx(0.30)
    assert record["exact_match"] is None


def test_build_record_schema_winogrande_acc_not_acc_norm():
    """WinoGrande record uses acc (R3); acc_norm is None."""
    raw = _fake_batch_result(acc=0.55, n=1267)
    record = harness.build_record(
        cell_id="C44",
        benchmark="winogrande",
        raw_result=raw,
        n_items=1267,
        harness_commit=harness.HARNESS_COMMIT,
    )
    assert record["benchmark"] == "winogrande"
    assert record["acc"] == pytest.approx(0.55)
    assert record["acc_norm"] is None   # R3: acc_norm absent in all harness versions
    assert record["sigma_gate_status"] == "PASS"


def test_build_record_schema_gsm8k_partial():
    """GSM8K record has exact_match=None and sigma_gate_status='PARTIAL_BLOCKED_NO_GENERATE'."""
    raw = harness.GSMI8K_BLOCKED_RESULT
    record = harness.build_record(
        cell_id="D42",
        benchmark="gsm8k",
        raw_result=raw,
        n_items=0,
        harness_commit=harness.HARNESS_COMMIT,
    )
    assert record["benchmark"] == "gsm8k"
    assert record["acc"] is None
    assert record["exact_match"] is None
    # GSM8K is PARTIAL — binary has no generate mode
    assert "PARTIAL" in record["sigma_gate_status"] or "BLOCKED" in record["sigma_gate_status"]


# ---------------------------------------------------------------------------
# 6. Output path conventions
# ---------------------------------------------------------------------------


def test_per_run_output_path_full_run(tmp_path: Path):
    """Full-run path: <output_dir>/<cell>_<benchmark>.json"""
    path = harness.per_run_output_path(
        output_dir=tmp_path,
        cell_id="A42",
        benchmark="mmlu",
        smoke=False,
    )
    assert path == tmp_path / "A42_mmlu.json"


def test_per_run_output_path_smoke(tmp_path: Path):
    """Smoke path: <output_dir>/smoke/smoke_<cell>.json"""
    path = harness.per_run_output_path(
        output_dir=tmp_path,
        cell_id="A42",
        benchmark="mmlu",
        smoke=True,
    )
    assert path == tmp_path / "smoke" / "smoke_A42.json"


# ---------------------------------------------------------------------------
# 7. Aggregate JSON schema (48 records)
# ---------------------------------------------------------------------------


def _make_fake_records(cells: List[str], benchmarks: List[str]) -> List[Dict[str, Any]]:
    records = []
    for cell in cells:
        for bm in benchmarks:
            acc = 0.25 if bm != "gsm8k" else None
            n = {"mmlu": 14042, "hellaswag": 10042, "winogrande": 1267, "gsm8k": 0}[bm]
            raw = _fake_batch_result(acc=acc or 0.25, n=max(n, 1))
            if bm == "gsm8k":
                raw = harness.GSMI8K_BLOCKED_RESULT
            records.append(
                harness.build_record(
                    cell_id=cell,
                    benchmark=bm,
                    raw_result=raw,
                    n_items=n,
                    harness_commit=harness.HARNESS_COMMIT,
                )
            )
    return records


def test_aggregate_has_48_records():
    records = _make_fake_records(EXPECTED_CELLS, EXPECTED_BENCHMARKS)
    assert len(records) == 48


def test_aggregate_all_cells_present():
    records = _make_fake_records(EXPECTED_CELLS, EXPECTED_BENCHMARKS)
    cells_in_records = {r["cell_id"] for r in records}
    assert cells_in_records == set(EXPECTED_CELLS)


def test_aggregate_all_benchmarks_present():
    records = _make_fake_records(EXPECTED_CELLS, EXPECTED_BENCHMARKS)
    bms_in_records = {r["benchmark"] for r in records}
    assert bms_in_records == set(EXPECTED_BENCHMARKS)


def test_aggregate_serialises_to_valid_json():
    records = _make_fake_records(EXPECTED_CELLS, EXPECTED_BENCHMARKS)
    j = json.dumps(records)
    parsed = json.loads(j)
    assert len(parsed) == 48


# ---------------------------------------------------------------------------
# 8. Batch size constant
# ---------------------------------------------------------------------------


def test_batch_size_is_16():
    """batch_size=16 is the load-bearing throughput choice per apparatus §6."""
    assert harness.BATCH_SIZE == 16


# ---------------------------------------------------------------------------
# 9. HARNESS_COMMIT constant
# ---------------------------------------------------------------------------


def test_harness_commit_matches_pin():
    assert harness.HARNESS_COMMIT == "3fa4fd725c8a428710109f1d6c14eda37e95baea"


# ---------------------------------------------------------------------------
# 10. Dataset loaders — left-truncation invariant
# ---------------------------------------------------------------------------


def test_load_hellaswag_items_respects_limit(tmp_path: Path):
    """load_hellaswag_items returns at most `limit` items and each has prompt/choices/label."""
    # Mock the HF dataset to avoid network I/O
    fake_docs = [
        {
            "query": f"Activity: ctx {i}",
            "choices": [f"end_a_{i}", f"end_b_{i}", f"end_c_{i}", f"end_d_{i}"],
            "gold": i % 4,
        }
        for i in range(20)
    ]

    def fake_load(path, name=None, split=None, streaming=False, **kw):
        return iter(fake_docs)

    with patch("eval_harness_p11._load_dataset_stream", side_effect=fake_load):
        items = harness.load_hellaswag_items(limit=10, max_seq_len=512)

    assert len(items) <= 10
    for item in items:
        assert "prompt" in item
        assert "choices" in item
        assert "label" in item
        assert isinstance(item["choices"], list)
        assert len(item["choices"]) == 4


def test_load_winogrande_items_respects_limit():
    """load_winogrande_items returns binary-choice items within limit."""
    fake_docs = [
        {
            "sentence": f"The _ ran away. {i}",
            "option1": f"dog_{i}",
            "option2": f"cat_{i}",
            "answer": "1",
        }
        for i in range(20)
    ]

    def fake_load(path, name=None, split=None, streaming=False, **kw):
        return iter(fake_docs)

    with patch("eval_harness_p11._load_dataset_stream", side_effect=fake_load):
        items = harness.load_winogrande_items(limit=10, max_seq_len=512)

    assert len(items) <= 10
    for item in items:
        assert "prompt" in item
        assert "choices" in item
        assert len(item["choices"]) == 2   # binary minimal-pair
        assert "label" in item
        assert item["label"] in (0, 1)


def test_load_hellaswag_applies_left_truncation():
    """Items whose prompt exceeds max_seq_len are left-truncated, not dropped."""
    # A very long prompt that would exceed a tiny max_seq_len
    long_prompt = "word " * 200  # ~1000 chars
    fake_docs = [
        {
            "query": long_prompt,
            "choices": ["end_a", "end_b", "end_c", "end_d"],
            "gold": 0,
        }
    ]

    def fake_load(path, name=None, split=None, streaming=False, **kw):
        return iter(fake_docs)

    with patch("eval_harness_p11._load_dataset_stream", side_effect=fake_load):
        items = harness.load_hellaswag_items(limit=1, max_seq_len=50)

    assert len(items) == 1
    # The item must be present (not dropped) — truncation applied
    prompt_tokens_approx = len(items[0]["prompt"].split())
    # At max_seq_len=50 chars, prompt should be much shorter than 200 words
    assert prompt_tokens_approx < 200


# ---------------------------------------------------------------------------
# 11. Run single benchmark (mock adapter)
# ---------------------------------------------------------------------------


def test_run_benchmark_mmlu_returns_record(tmp_path: Path):
    """run_benchmark calls adapter and returns a valid record for MMLU."""
    ckpt = tmp_path / "A42" / "final.ckpt"
    ckpt.parent.mkdir()
    ckpt.touch()
    tok = tmp_path / "tokenizer.bin"
    tok.touch()

    fake_result = _fake_batch_result(acc=0.25, n=14042)
    with (
        patch("eval_harness_p11.load_mmlu_items", return_value=[{"prompt": "q", "choices": [" A"], "label": 0}] * 14042),
        patch("eval_harness_p11._run_benchmark_via_adapter", return_value=fake_result),
    ):
        record = harness.run_benchmark(
            cell_id="A42",
            benchmark="mmlu",
            checkpoint=ckpt,
            tokenizer=tok,
            output_dir=tmp_path,
            batch_size=16,
            smoke=False,
            limit=None,
        )

    assert record["cell_id"] == "A42"
    assert record["benchmark"] == "mmlu"
    assert record["acc"] == pytest.approx(0.25)
    assert record["sigma_gate_status"] in ("PASS", "FAIL")


def test_run_benchmark_writes_json_file(tmp_path: Path):
    """run_benchmark writes per-run JSON to output_dir."""
    ckpt = tmp_path / "A42" / "final.ckpt"
    ckpt.parent.mkdir()
    ckpt.touch()
    tok = tmp_path / "tokenizer.bin"
    tok.touch()

    fake_result = _fake_batch_result(acc=0.25, n=14042)
    with (
        patch("eval_harness_p11.load_mmlu_items", return_value=[{"prompt": "q", "choices": [" A"], "label": 0}]),
        patch("eval_harness_p11._run_benchmark_via_adapter", return_value=fake_result),
    ):
        harness.run_benchmark(
            cell_id="A42",
            benchmark="mmlu",
            checkpoint=ckpt,
            tokenizer=tok,
            output_dir=tmp_path,
            batch_size=16,
            smoke=False,
            limit=None,
        )

    expected_path = tmp_path / "A42_mmlu.json"
    assert expected_path.exists()
    data = json.loads(expected_path.read_text())
    assert data["cell_id"] == "A42"


def test_run_benchmark_gsm8k_returns_partial_record(tmp_path: Path):
    """run_benchmark for gsm8k returns a PARTIAL record (no generate mode)."""
    ckpt = tmp_path / "A42" / "final.ckpt"
    ckpt.parent.mkdir()
    ckpt.touch()
    tok = tmp_path / "tokenizer.bin"
    tok.touch()

    record = harness.run_benchmark(
        cell_id="A42",
        benchmark="gsm8k",
        checkpoint=ckpt,
        tokenizer=tok,
        output_dir=tmp_path,
        batch_size=16,
        smoke=False,
        limit=None,
    )

    assert record["benchmark"] == "gsm8k"
    assert record["acc"] is None
    assert "PARTIAL" in record["sigma_gate_status"] or "BLOCKED" in record["sigma_gate_status"]


# ---------------------------------------------------------------------------
# 12. Smoke-test mode
# ---------------------------------------------------------------------------


def test_smoke_mode_uses_limit(tmp_path: Path):
    """In smoke mode, run_benchmark passes limit to the dataset loader."""
    ckpt = tmp_path / "A42" / "final.ckpt"
    ckpt.parent.mkdir()
    ckpt.touch()
    tok = tmp_path / "tokenizer.bin"
    tok.touch()

    fake_result = _fake_batch_result(acc=0.24, n=100)
    captured_limit = []

    def fake_load_mmlu(limit=None, **kw):
        captured_limit.append(limit)
        return [{"prompt": "q", "choices": [" A"], "label": 0}] * (limit or 100)

    with (
        patch("eval_harness_p11.load_mmlu_items", side_effect=fake_load_mmlu),
        patch("eval_harness_p11._run_benchmark_via_adapter", return_value=fake_result),
    ):
        harness.run_benchmark(
            cell_id="A42",
            benchmark="mmlu",
            checkpoint=ckpt,
            tokenizer=tok,
            output_dir=tmp_path / "smoke",
            batch_size=16,
            smoke=True,
            limit=100,
        )

    assert captured_limit[0] == 100


def test_smoke_mode_writes_to_smoke_subdir(tmp_path: Path):
    """Smoke output goes to <output_dir>/smoke/smoke_<cell>.json."""
    ckpt = tmp_path / "A42" / "final.ckpt"
    ckpt.parent.mkdir()
    ckpt.touch()
    tok = tmp_path / "tokenizer.bin"
    tok.touch()

    fake_result = _fake_batch_result(acc=0.24, n=100)

    with (
        patch("eval_harness_p11.load_mmlu_items", return_value=[{"prompt": "q", "choices": [" A"], "label": 0}]),
        patch("eval_harness_p11._run_benchmark_via_adapter", return_value=fake_result),
    ):
        harness.run_benchmark(
            cell_id="A42",
            benchmark="mmlu",
            checkpoint=ckpt,
            tokenizer=tok,
            output_dir=tmp_path,
            batch_size=16,
            smoke=True,
            limit=100,
        )

    smoke_path = tmp_path / "smoke" / "smoke_A42.json"
    assert smoke_path.exists()


# ---------------------------------------------------------------------------
# 13. CLI argument parsing
# ---------------------------------------------------------------------------


def test_parse_args_full_run_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_harness_p11.py",
            "--checkpoint-base", str(tmp_path),
            "--output-dir", str(tmp_path),
            "--tokenizer", str(tmp_path / "tok.bin"),
            "--output-json", str(tmp_path / "scores.json"),
        ],
    )
    args = harness.parse_args()
    assert args.smoke_test is False
    assert args.limit is None
    assert args.task is None
    assert args.checkpoint is None


def test_parse_args_smoke_test_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_harness_p11.py",
            "--checkpoint", str(tmp_path / "A42" / "final.ckpt"),
            "--output-dir", str(tmp_path),
            "--tokenizer", str(tmp_path / "tok.bin"),
            "--task", "mmlu",
            "--limit", "100",
            "--smoke-test",
        ],
    )
    args = harness.parse_args()
    assert args.smoke_test is True
    assert args.limit == 100
    assert args.task == "mmlu"
    assert args.checkpoint is not None


def test_parse_args_no_limit_in_full_run_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Full-run invocation must have limit=None (apparatus G5: no --limit flag)."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_harness_p11.py",
            "--checkpoint-base", str(tmp_path),
            "--output-dir", str(tmp_path),
            "--tokenizer", str(tmp_path / "tok.bin"),
            "--output-json", str(tmp_path / "scores.json"),
        ],
    )
    args = harness.parse_args()
    assert args.limit is None  # G5: full-N mandatory, no --limit in full-run path


# ---------------------------------------------------------------------------
# 14. Defensive parsing — "N/A" and None values from lm-eval-harness v0.4.3
# ---------------------------------------------------------------------------


def _make_harness_results_with_na_stderr(task_name: str = "hellaswag") -> Dict[str, Any]:
    """Construct a minimal harness_results dict that mimics lm-eval v0.4.3 output
    when bootstrap_iters=0: metric values are floats but stderr values are "N/A"."""
    return {
        "results": {
            task_name: {
                "acc,none": 0.40,
                "acc_norm,none": 0.42,
                "acc_stderr,none": "N/A",
                "acc_norm_stderr,none": "N/A",
            }
        }
    }


def test_extract_harness_stderr_returns_none_for_na_string():
    """_extract_harness_stderr must return None (not raise) when value is 'N/A'.

    Regression test for D-330: lm-eval v0.4.3 returns 'N/A' as the stderr
    value when bootstrap_iters=0; the pre-fix code called float('N/A') which
    raised ValueError.
    """
    harness_results = _make_harness_results_with_na_stderr("hellaswag")
    result = harness._extract_harness_stderr(harness_results, "hellaswag", "acc_norm")
    assert result is None, (
        f"Expected None for 'N/A' stderr, got {result!r}. "
        "Unpatched code raises ValueError: could not convert string to float: 'N/A'"
    )


def test_extract_harness_metric_returns_none_for_na_string():
    """_extract_harness_metric must return None (not raise) when value is 'N/A'.

    Same defensive-parse requirement as stderr; metric values could also be
    'N/A' on degenerate tasks or partial harness runs.
    """
    harness_results: Dict[str, Any] = {
        "results": {
            "hellaswag": {
                "acc,none": "N/A",
                "acc_norm,none": 0.42,
            }
        }
    }
    result = harness._extract_harness_metric(harness_results, "hellaswag", "acc")
    assert result is None, (
        f"Expected None for 'N/A' metric, got {result!r}."
    )


def test_extract_harness_stderr_returns_none_for_none_value():
    """_extract_harness_stderr must return None when the value itself is None."""
    harness_results: Dict[str, Any] = {
        "results": {
            "hellaswag": {
                "acc_norm_stderr,none": None,
            }
        }
    }
    result = harness._extract_harness_stderr(harness_results, "hellaswag", "acc_norm")
    assert result is None


def test_extract_harness_metric_returns_none_for_none_value():
    """_extract_harness_metric must return None when the value itself is None."""
    harness_results: Dict[str, Any] = {
        "results": {
            "hellaswag": {
                "acc,none": None,
            }
        }
    }
    result = harness._extract_harness_metric(harness_results, "hellaswag", "acc")
    assert result is None


def test_build_record_from_harness_survives_na_stderr():
    """build_record_from_harness must return a valid record when stderr is 'N/A'.

    Covers the full call chain: _extract_harness_stderr is called inside
    build_record_from_harness; a ValueError propagation would crash the smoke run.
    The returned record's stderr field must be None (not raise, not 'N/A').
    """
    harness_results = _make_harness_results_with_na_stderr("hellaswag")
    record = harness.build_record_from_harness(
        cell_id="A42",
        benchmark="hellaswag",
        harness_results=harness_results,
        n_items=5,
        wall_clock_s=1.0,
    )
    assert record["acc_norm"] == pytest.approx(0.42), "acc_norm should be extracted correctly"
    assert record["stderr"] is None, (
        f"stderr must be None when harness returns 'N/A', got {record['stderr']!r}"
    )
    assert record["sigma_gate_status"] in ("PASS", "FAIL"), (
        "sigma_gate_status must still be computed when stderr is N/A"
    )


# ---------------------------------------------------------------------------
# 15. Per-cell checkpoint persistence + resume-skip (mid-run death recovery)
# ---------------------------------------------------------------------------


def _make_fake_record(cell_id: str, benchmark: str) -> Dict[str, Any]:
    """Minimal record dict matching the harness schema for two-cell fixture tests."""
    return {
        "cell_id": cell_id,
        "benchmark": benchmark,
        "acc": 0.25,
        "acc_norm": None,
        "exact_match": None,
        "stderr": 0.004,
        "n_items": 100,
        "sigma_gate_floor": SIGMA_GATE_FLOORS.get(benchmark, 0.0),
        "sigma_gate_status": "FAIL",
        "harness_commit": harness.HARNESS_COMMIT,
        "wall_clock_s": 1.0,
    }


def _make_ckpt(tmp_path: Path, cell_id: str) -> Path:
    """Create a fake checkpoint file for cell_id under tmp_path."""
    ckpt = tmp_path / "ckpts" / cell_id / "final.ckpt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.touch()
    return ckpt


def test_run_full_writes_per_cell_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_full writes a per-cell JSON immediately after each benchmark completes.

    Crucial property: per-cell artifacts must exist MID-LOOP (before the at-end
    aggregate write). This verifies that a mid-run death loses at most the in-flight
    cell, not the full 48-record run.
    """
    # Two-cell fixture: A42, A43 x ["mmlu"] only
    two_cells = ["A42", "A43"]
    monkeypatch.setattr(harness, "CELLS", two_cells)

    # Create fake checkpoints for both cells
    for cell_id in two_cells:
        _make_ckpt(tmp_path, cell_id)

    checkpoint_base = tmp_path / "ckpts"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    output_json = tmp_path / "benchmark_scores.json"
    tokenizer = tmp_path / "tok.bin"
    tokenizer.touch()

    call_count = [0]
    mid_loop_artifact_present = [False]

    def stub_run_benchmark_via_harness(
        cell_id: str,
        benchmark: str,
        checkpoint: Path,
        tokenizer: Path,
        limit=None,
        batch_size: int = harness.BATCH_SIZE,
        debug_token_window: bool = False,
    ) -> Dict[str, Any]:
        call_count[0] += 1
        # After the second call, check that the first per-cell artifact already exists
        if call_count[0] == 2:
            first_path = output_dir / f"{two_cells[0]}_mmlu.json"
            mid_loop_artifact_present[0] = first_path.exists()
        return _make_fake_record(cell_id, benchmark)

    monkeypatch.setattr(harness, "run_benchmark_via_harness", stub_run_benchmark_via_harness)
    # Override benchmark list to one entry for test speed
    monkeypatch.setattr(harness, "BENCHMARKS", ["mmlu"])

    harness.run_full(
        checkpoint_base=checkpoint_base,
        output_dir=output_dir,
        tokenizer=tokenizer,
        output_json=output_json,
    )

    # Both per-cell artifacts must exist after run_full
    assert (output_dir / "A42_mmlu.json").exists(), "per-cell artifact A42_mmlu.json missing"
    assert (output_dir / "A43_mmlu.json").exists(), "per-cell artifact A43_mmlu.json missing"

    # Critical: first artifact existed BEFORE the second stub call returned
    # (i.e., written mid-loop, not just at the end)
    assert mid_loop_artifact_present[0], (
        "A42_mmlu.json did not exist when the second benchmark was being computed. "
        "Per-cell write must happen immediately after records.append(), not at end-of-loop."
    )

    # Aggregate still exists
    assert output_json.exists(), "at-end aggregate benchmark_scores.json missing"


def test_run_full_resume_skip_loads_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_full skips cells whose per-cell artifact already exists on disk.

    Pre-placing A42_mmlu.json: the stub must NOT be called for A42/mmlu,
    but MUST be called for A43/mmlu. The final records list must contain
    both the pre-placed record (loaded from disk) and the freshly-computed one.
    """
    two_cells = ["A42", "A43"]
    monkeypatch.setattr(harness, "CELLS", two_cells)
    monkeypatch.setattr(harness, "BENCHMARKS", ["mmlu"])

    for cell_id in two_cells:
        _make_ckpt(tmp_path, cell_id)

    checkpoint_base = tmp_path / "ckpts"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    output_json = tmp_path / "benchmark_scores.json"
    tokenizer = tmp_path / "tok.bin"
    tokenizer.touch()

    # Pre-place A42_mmlu.json with a distinctive accuracy value
    pre_placed_record = _make_fake_record("A42", "mmlu")
    pre_placed_record["acc"] = 0.99  # sentinel value to confirm it was loaded, not recomputed
    pre_placed_path = output_dir / "A42_mmlu.json"
    pre_placed_path.write_text(json.dumps(pre_placed_record, indent=2), encoding="utf-8")

    calls_made: List[str] = []

    def stub_run_benchmark_via_harness(
        cell_id: str,
        benchmark: str,
        checkpoint: Path,
        tokenizer: Path,
        limit=None,
        batch_size: int = harness.BATCH_SIZE,
        debug_token_window: bool = False,
    ) -> Dict[str, Any]:
        calls_made.append(f"{cell_id}_{benchmark}")
        return _make_fake_record(cell_id, benchmark)

    monkeypatch.setattr(harness, "run_benchmark_via_harness", stub_run_benchmark_via_harness)

    records = harness.run_full(
        checkpoint_base=checkpoint_base,
        output_dir=output_dir,
        tokenizer=tokenizer,
        output_json=output_json,
    )

    # Stub must NOT have been called for A42/mmlu (skipped via resume)
    assert "A42_mmlu" not in calls_made, (
        f"run_benchmark_via_harness was called for A42/mmlu despite pre-placed artifact. "
        f"calls_made={calls_made}"
    )

    # Stub MUST have been called for A43/mmlu (no pre-placed artifact)
    assert "A43_mmlu" in calls_made, (
        f"run_benchmark_via_harness was NOT called for A43/mmlu. calls_made={calls_made}"
    )

    # Both records must be in final list
    assert len(records) == 2, f"Expected 2 records, got {len(records)}"

    # The A42 record must carry the pre-placed sentinel accuracy
    a42_record = next(r for r in records if r["cell_id"] == "A42")
    assert a42_record["acc"] == pytest.approx(0.99), (
        f"A42 record should have loaded pre-placed acc=0.99 from disk, got {a42_record['acc']}"
    )


def test_run_full_atomic_write_no_tmp_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No .tmp.json files remain in output_dir after run_full completes successfully.

    Also verifies: if a .tmp.json exists alongside a .json (leftover from a prior
    aborted write), a subsequent resume-skip run does NOT clobber the existing .json
    — the existing artifact is loaded and the cell is skipped entirely.
    """
    two_cells = ["A42", "A43"]
    monkeypatch.setattr(harness, "CELLS", two_cells)
    monkeypatch.setattr(harness, "BENCHMARKS", ["mmlu"])

    for cell_id in two_cells:
        _make_ckpt(tmp_path, cell_id)

    checkpoint_base = tmp_path / "ckpts"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    output_json = tmp_path / "benchmark_scores.json"
    tokenizer = tmp_path / "tok.bin"
    tokenizer.touch()

    # Simulate a prior aborted write: leave both a .json and a .tmp.json for A42
    prior_record = _make_fake_record("A42", "mmlu")
    prior_record["acc"] = 0.77  # sentinel: this value should survive unchanged
    final_path = output_dir / "A42_mmlu.json"
    tmp_residue_path = output_dir / "A42_mmlu.tmp.json"
    final_path.write_text(json.dumps(prior_record, indent=2), encoding="utf-8")
    tmp_residue_path.write_text('{"partial": true}', encoding="utf-8")  # corrupt/partial

    stub_calls: List[str] = []

    def stub_run_benchmark_via_harness(
        cell_id: str,
        benchmark: str,
        checkpoint: Path,
        tokenizer: Path,
        limit=None,
        batch_size: int = harness.BATCH_SIZE,
        debug_token_window: bool = False,
    ) -> Dict[str, Any]:
        stub_calls.append(f"{cell_id}_{benchmark}")
        return _make_fake_record(cell_id, benchmark)

    monkeypatch.setattr(harness, "run_benchmark_via_harness", stub_run_benchmark_via_harness)

    records = harness.run_full(
        checkpoint_base=checkpoint_base,
        output_dir=output_dir,
        tokenizer=tokenizer,
        output_json=output_json,
    )

    # No .tmp.json files created by this run should remain (freshly-computed cells only).
    # A43_mmlu.tmp.json must not exist — the atomic write replaced it with A43_mmlu.json.
    assert not (output_dir / "A43_mmlu.tmp.json").exists(), (
        "A43_mmlu.tmp.json still exists — atomic replace did not remove the .tmp.json"
    )
    # A43_mmlu.json must exist (fresh write succeeded)
    assert (output_dir / "A43_mmlu.json").exists(), "A43_mmlu.json missing after fresh computation"

    # A42 was skipped (pre-placed .json exists) — stub must NOT have been called for it
    assert "A42_mmlu" not in stub_calls, (
        f"A42/mmlu was recomputed despite existing .json artifact. stub_calls={stub_calls}"
    )

    # The existing A42_mmlu.json must NOT be clobbered — sentinel acc=0.77 preserved
    loaded = json.loads(final_path.read_text(encoding="utf-8"))
    assert loaded["acc"] == pytest.approx(0.77), (
        f"Existing A42_mmlu.json was overwritten! Expected acc=0.77, got {loaded['acc']}"
    )
