"""Tests for tools/hspa_lm_eval_adapter.py.

Covers:
- JSONL formatting from item dicts
- Batch splitting logic
- Binary result parsing (via mocked subprocess)
- Accuracy / stderr calculation
- CLI argument defaults and path validation
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess

import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path for the import
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import hspa_lm_eval_adapter as adapter


# ---------------------------------------------------------------------------
# format_as_jsonl
# ---------------------------------------------------------------------------


def test_format_as_jsonl_basic() -> None:
    items = [
        {"prompt": "What is 2+2?", "choices": [" A", " B", " C", " D"], "label": 0},
    ]
    output = adapter.format_as_jsonl(items)
    lines = [l for l in output.strip().splitlines() if l.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["prompt"] == "What is 2+2?"
    assert parsed["choices"] == [" A", " B", " C", " D"]
    assert parsed["label"] == 0


def test_format_as_jsonl_multiple_items() -> None:
    items = [
        {"prompt": "Q1?", "choices": [" A", " B", " C", " D"], "label": 1},
        {"prompt": "Q2?", "choices": [" A", " B", " C", " D"], "label": 3},
    ]
    output = adapter.format_as_jsonl(items)
    lines = [l for l in output.strip().splitlines() if l.strip()]
    assert len(lines) == 2
    p0 = json.loads(lines[0])
    p1 = json.loads(lines[1])
    assert p0["label"] == 1
    assert p1["label"] == 3


def test_format_as_jsonl_escaping() -> None:
    """Prompts with quotes and newlines must survive round-trip through json.loads."""
    items = [
        {
            "prompt": 'She said "hello"\nand left.',
            "choices": [" A", " B"],
            "label": 0,
        }
    ]
    output = adapter.format_as_jsonl(items)
    parsed = json.loads(output.strip().splitlines()[0])
    assert 'hello' in parsed["prompt"]
    assert "\n" in parsed["prompt"]


# ---------------------------------------------------------------------------
# run_eval_binary (mocked subprocess)
# ---------------------------------------------------------------------------


def _make_mock_proc(returncode: int, stdout: str, stderr: str) -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


def test_run_eval_binary_success(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "test.jsonl"
    jsonl_path.write_text('{"prompt":"Q","choices":[" A"],"label":0}\n')
    stdout_payload = json.dumps({"accuracy": 0.75, "correct": 75, "total": 100})

    with (
        patch("subprocess.run", return_value=_make_mock_proc(0, stdout_payload, "")),
        patch("resource.getrusage") as mock_ru,
    ):
        mock_ru.return_value = MagicMock(ru_maxrss=512 * 1024 * 1024)  # 512 MB in bytes
        result = adapter.run_eval_binary(
            checkpoint=tmp_path / "ckpt",
            tokenizer=tmp_path / "tok",
            jsonl_path=jsonl_path,
        )

    assert result["returncode"] == 0
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["correct"] == 75
    assert result["total"] == 100
    assert "error" not in result


def test_run_eval_binary_nonzero_rc(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "test.jsonl"
    jsonl_path.write_text("")

    with (
        patch("subprocess.run", return_value=_make_mock_proc(1, "", "fatal error")),
        patch("resource.getrusage") as mock_ru,
    ):
        mock_ru.return_value = MagicMock(ru_maxrss=0)
        result = adapter.run_eval_binary(
            checkpoint=tmp_path / "ckpt",
            tokenizer=tmp_path / "tok",
            jsonl_path=jsonl_path,
        )

    assert result["returncode"] == 1
    assert result["accuracy"] is None
    assert "error" in result


def test_run_eval_binary_bad_stdout(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "test.jsonl"
    jsonl_path.write_text("")

    with (
        patch("subprocess.run", return_value=_make_mock_proc(0, "not json", "")),
        patch("resource.getrusage") as mock_ru,
    ):
        mock_ru.return_value = MagicMock(ru_maxrss=0)
        result = adapter.run_eval_binary(
            checkpoint=tmp_path / "ckpt",
            tokenizer=tmp_path / "tok",
            jsonl_path=jsonl_path,
        )

    assert result["accuracy"] is None
    assert "error" in result


# ---------------------------------------------------------------------------
# run_eval_in_batches (mocked run_eval_binary)
# ---------------------------------------------------------------------------


def _make_items(n: int) -> list:
    return [
        {"prompt": f"Q{i}?", "choices": [" A", " B", " C", " D"], "label": i % 4}
        for i in range(n)
    ]


def test_run_eval_in_batches_aggregate(tmp_path: Path) -> None:
    """Two batches of 5 items each, both returning 4/5 correct."""
    items = _make_items(10)
    batch_result = {"accuracy": 0.8, "correct": 4, "total": 5, "stderr": "", "rss_peak_mb": 100.0, "wall_clock_s": 1.0, "returncode": 0}

    with patch.object(adapter, "run_eval_binary", return_value=batch_result):
        result = adapter.run_eval_in_batches(
            items=items,
            checkpoint=tmp_path / "ckpt",
            tokenizer=tmp_path / "tok",
            batch_size=5,
            tmpdir=tmp_path,
        )

    assert result["correct"] == 8
    assert result["total"] == 10
    assert result["accuracy"] == pytest.approx(0.8)
    # binomial stderr: sqrt(0.8*0.2/10) = sqrt(0.016) ≈ 0.1265
    expected_stderr = math.sqrt(0.8 * 0.2 / 10)
    assert result["stderr"] == pytest.approx(expected_stderr, abs=0.001)
    assert len(result["per_batch_timings_s"]) == 2
    assert "error" not in result


def test_run_eval_in_batches_single_batch(tmp_path: Path) -> None:
    items = _make_items(3)
    batch_result = {"accuracy": 1.0, "correct": 3, "total": 3, "stderr": "", "rss_peak_mb": 50.0, "wall_clock_s": 0.5, "returncode": 0}

    with patch.object(adapter, "run_eval_binary", return_value=batch_result):
        result = adapter.run_eval_in_batches(
            items=items,
            checkpoint=tmp_path / "ckpt",
            tokenizer=tmp_path / "tok",
            batch_size=10,
            tmpdir=tmp_path,
        )

    assert result["total"] == 3
    assert result["correct"] == 3
    assert len(result["per_batch_timings_s"]) == 1


def test_run_eval_in_batches_propagates_error(tmp_path: Path) -> None:
    items = _make_items(10)
    error_result = {
        "accuracy": None,
        "correct": None,
        "total": None,
        "stderr": "crash",
        "rss_peak_mb": 0.0,
        "wall_clock_s": 0.1,
        "returncode": 1,
        "error": "Binary exited with rc=1.",
    }

    with patch.object(adapter, "run_eval_binary", return_value=error_result):
        result = adapter.run_eval_in_batches(
            items=items,
            checkpoint=tmp_path / "ckpt",
            tokenizer=tmp_path / "tok",
            batch_size=5,
            tmpdir=tmp_path,
        )

    assert "error" in result
    assert result["batch_index_failed"] == 0


# ---------------------------------------------------------------------------
# format_few_shot_prefix
# ---------------------------------------------------------------------------


def test_format_few_shot_prefix_structure() -> None:
    examples = [
        {"question": "What is 1+1?", "choices": ["1", "2", "3", "4"], "answer": 1},
        {"question": "Capital of France?", "choices": ["Berlin", "Paris", "Rome", "Madrid"], "answer": 1},
    ]
    prefix = adapter._format_few_shot_prefix(examples, "math")
    assert "multiple choice questions" in prefix
    assert "math" in prefix
    assert "What is 1+1?" in prefix
    assert "Answer: B" in prefix
    assert "Capital of France?" in prefix


def test_format_query_prompt_no_answer() -> None:
    example = {"question": "What is 2+2?", "choices": ["2", "3", "4", "5"], "answer": 2}
    prompt = adapter._format_query_prompt(example)
    assert "Question: What is 2+2?" in prompt
    assert "Answer:" in prompt
    # Must NOT contain the answer letter itself
    assert prompt.strip().endswith("Answer:")


# ---------------------------------------------------------------------------
# ANSWER_LABELS and constants
# ---------------------------------------------------------------------------


def test_answer_labels_length() -> None:
    assert len(adapter.ANSWER_LABELS) == 4
    assert adapter.ANSWER_LABELS == ["A", "B", "C", "D"]


def test_max_seq_len_default() -> None:
    assert adapter.MAX_SEQ_LEN == 512


# ---------------------------------------------------------------------------
# CLI: parse_args defaults
# ---------------------------------------------------------------------------


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["hspa_lm_eval_adapter.py"])
    args = adapter.parse_args()
    assert args.limit == 100
    assert args.num_fewshot == 5
    assert args.batch_size == 25
    assert args.max_seq_len == 512
    assert args.task == "mmlu"
    assert args.output is None
    assert args.subjects is None


def test_parse_args_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hspa_lm_eval_adapter.py",
            "--limit", "50",
            "--num-fewshot", "0",
            "--batch-size", "10",
        ],
    )
    args = adapter.parse_args()
    assert args.limit == 50
    assert args.num_fewshot == 0
    assert args.batch_size == 10


# ---------------------------------------------------------------------------
# _batch_requests_under_byte_budget — unit tests (new D-344 batching helper)
# ---------------------------------------------------------------------------


def _make_loglikelihood_items(n: int, prompt_size: int = 40) -> list:
    """Return n items matching the loglikelihood() JSONL schema.

    prompt_size controls approximate per-item byte size so tests can control
    whether items cross batch boundaries.
    """
    return [
        {
            "prompt": f"Q{i}: " + ("x" * prompt_size),
            "choices": [" A"],
            "label": 0,
        }
        for i in range(n)
    ]


def _item_bytes(item: dict) -> int:
    return len((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))


class _FakeRequest:
    """Minimal stand-in for lm_eval's Instance object inside loglikelihood()."""

    def __init__(self, context: str, continuation: str) -> None:
        self.args = (context, continuation)


def _make_fake_requests(n: int, prompt_size: int = 40) -> list:
    return [
        _FakeRequest(
            context=f"ctx{i}: " + ("x" * prompt_size),
            continuation=" A",
        )
        for i in range(n)
    ]


def _make_score_per_choice_stdout(n_items: int, logprob: float = -1.5) -> str:
    """Build a minimal score-per-choice stdout with n_items rows + aggregate."""
    lines = []
    for i in range(n_items):
        lines.append(json.dumps({
            "item_id": str(i),
            "choices": [{"choice_idx": 0, "logprob": logprob, "n_tokens": 1}],
            "label": 0,
            "predicted": 0,
        }))
    lines.append(json.dumps({"accuracy": 1.0, "correct": n_items, "total": n_items}))
    return "\n".join(lines)


# --- Test 1: BATCH_MAX_BYTES constant exists and has the correct value -------


def test_batch_max_bytes_constant() -> None:
    """BATCH_MAX_BYTES must equal 12 MB (4 MB margin under binary's 16 MB cap)."""
    assert adapter.BATCH_MAX_BYTES == 12 * 1024 * 1024


# --- Test 2: Functional test — 100-item single-shot vs batched result --------


def test_loglikelihood_batched_bit_identical_to_single_shot(tmp_path: Path) -> None:
    """100-item fixture: result from a single batch == result from multi-batch.

    This is the differential regression test.  We drive loglikelihood() twice:
    once with a budget large enough to fit all 100 items in one batch, and once
    with a tiny budget that forces many batches.  Both runs mock subprocess.run
    to return deterministic per-item logprobs.  The returned list must be
    bit-identical.
    """
    N = 100
    LOGPROB = -2.3
    requests = _make_fake_requests(N, prompt_size=40)

    def fake_run(cmd, **kwargs):
        # Determine n_items from the tempfile path in cmd
        input_path = Path(cmd[cmd.index("--input") + 1])
        lines = [l for l in input_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = len(lines)
        m = MagicMock()
        m.returncode = 0
        m.stdout = _make_score_per_choice_stdout(n, LOGPROB)
        m.stderr = ""
        return m

    # Single-shot: budget big enough that all 100 items fit in one batch
    with patch("subprocess.run", side_effect=fake_run):
        single_result = adapter.HSPALMEvalAdapter(
            checkpoint=tmp_path / "ckpt",
            tokenizer=tmp_path / "tok",
        ).loglikelihood.__func__(
            adapter.HSPALMEvalAdapter(
                checkpoint=tmp_path / "ckpt",
                tokenizer=tmp_path / "tok",
            ),
            requests,
        )

    # Batched: force many tiny batches by using a very small budget per batch
    # Compute the size of one item and set budget to 2 items worth of bytes
    one_item = {
        "prompt": requests[0].args[0],
        "choices": [requests[0].args[1]],
        "label": 0,
    }
    one_item_bytes = _item_bytes(one_item)
    tiny_budget = one_item_bytes * 2  # fits exactly 2 items per batch

    batches_seen = []

    def fake_run_batched(cmd, **kwargs):
        input_path = Path(cmd[cmd.index("--input") + 1])
        lines = [l for l in input_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = len(lines)
        batches_seen.append(n)
        m = MagicMock()
        m.returncode = 0
        m.stdout = _make_score_per_choice_stdout(n, LOGPROB)
        m.stderr = ""
        return m

    with patch("subprocess.run", side_effect=fake_run_batched):
        with patch.object(adapter, "BATCH_MAX_BYTES", tiny_budget):
            batched_adapter = adapter.HSPALMEvalAdapter(
                checkpoint=tmp_path / "ckpt",
                tokenizer=tmp_path / "tok",
            )
            # Patch the constant used inside loglikelihood via the module
            import importlib
            batched_result = []
            # Invoke directly, patching at the helper level
            orig_fn = adapter._batch_requests_under_byte_budget
            def patched_batch(items, budget=adapter.BATCH_MAX_BYTES):
                return orig_fn(items, tiny_budget)
            with patch.object(adapter, "_batch_requests_under_byte_budget", side_effect=patched_batch):
                batched_result = batched_adapter.loglikelihood(requests)

    assert len(single_result) == N
    assert len(batched_result) == N
    # All logprobs must be identical (bit-for-bit equal floats from same source)
    assert single_result == batched_result, (
        "Single-shot and batched loglikelihood() returned different results"
    )


# --- Test 3: Scale test — 50k items produce ≥3 batches, correct count, order ---


def test_batch_requests_under_byte_budget_scale_50k() -> None:
    """50k synthetic items: ≥3 batches, total len == 50000, order preserved.

    Items use a small prompt (~40 chars) so generation is fast.  To guarantee
    ≥3 batches regardless of BATCH_MAX_BYTES, the budget is computed as
    roughly 1/5 of the total serialised size of all 50k items — yielding ~5
    batches in practice.  This exercises _batch_requests_under_byte_budget()
    at meaningful scale without depending on the production constant's value.

    The test verifies four invariants from the dispatch:
      1. At least 3 batches are produced.
      2. Concatenated batches contain exactly 50000 items.
      3. Concatenated order matches the original order.
      4. No batch exceeds the supplied budget.
    """
    N = 50_000
    items = [
        {"prompt": f"Q{i}: " + ("x" * 40), "choices": [" A"], "label": 0}
        for i in range(N)
    ]

    # Compute total serialised size and choose a budget that forces ~5 batches
    total_bytes = sum(_item_bytes(item) for item in items)
    budget = total_bytes // 5  # ~20% of total → approximately 5 batches

    batches = list(adapter._batch_requests_under_byte_budget(items, budget))

    # Invariant 1: at least 3 batches
    assert len(batches) >= 3, (
        f"Expected >=3 batches for {N} items at budget={budget}, got {len(batches)}"
    )

    # Invariant 2: total item count == N
    total = sum(len(b) for b in batches)
    assert total == N, f"Expected {N} items total, got {total}"

    # Invariant 3: order preserved — reconstruct and compare
    reconstructed = [item for batch in batches for item in batch]
    assert reconstructed == items, "Order not preserved across batches"

    # Invariant 4: no batch exceeds the byte budget
    for b_idx, batch in enumerate(batches):
        batch_bytes = sum(_item_bytes(item) for item in batch)
        assert batch_bytes <= budget, (
            f"Batch {b_idx} exceeds budget: {batch_bytes} > {budget}"
        )


# --- Test 4: Singleton-overflow test — oversized item yields as singleton -----


def test_batch_requests_under_byte_budget_singleton_overflow() -> None:
    """A single item whose serialised JSON exceeds BATCH_MAX_BYTES yields solo.

    The helper must NOT drop or silently skip the item.  It should log a
    WARNING and yield a singleton batch.  Downstream binary will fail loudly
    for that batch if the binary's own cap is also exceeded — that is the
    correct surfacing behaviour per the dispatch spec.
    """
    # Build one very large item: prompt just over BATCH_MAX_BYTES worth of bytes
    oversized_prompt = "x" * (adapter.BATCH_MAX_BYTES + 100)
    oversized_item = {
        "prompt": oversized_prompt,
        "choices": [" A"],
        "label": 0,
    }
    assert _item_bytes(oversized_item) > adapter.BATCH_MAX_BYTES, (
        "Test setup error: oversized item did not actually exceed the budget"
    )

    import logging
    with patch.object(adapter._log, "warning") as mock_warn:
        batches = list(
            adapter._batch_requests_under_byte_budget(
                [oversized_item], adapter.BATCH_MAX_BYTES
            )
        )

    # Must yield exactly one batch containing exactly one item
    assert len(batches) == 1, f"Expected 1 batch, got {len(batches)}"
    assert len(batches[0]) == 1, f"Expected singleton batch, got {len(batches[0])} items"
    assert batches[0][0] is oversized_item, "Batch item identity not preserved"
    # Warning must have been emitted
    assert mock_warn.called, "Expected _log.warning to be called for oversized item"
