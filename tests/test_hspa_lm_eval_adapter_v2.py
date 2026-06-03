"""Tests for the v2 HSPA lm-eval adapter (LM subclass implementation).

Covers Phase B deliverables:
(a) loglikelihood on a 2-instance fixture returns correct shape and types
(b) batch invocation produces the same logprobs as sequential per-instance
(c) JSONL parse of the score-per-choice schema is correct
(d) loglikelihood_rolling raises NotImplementedError
(e) generate_until raises NotImplementedError

Dependencies mocked: subprocess.run (binary invocation)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch, call

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import hspa_lm_eval_adapter as adapter


# ---------------------------------------------------------------------------
# Helpers: fixture factories
# ---------------------------------------------------------------------------


def _make_lm_instance(checkpoint: Path = None, tokenizer: Path = None) -> "adapter.HSPALMEvalAdapter":
    """Construct an HSPALMEvalAdapter with paths that don't need to exist for unit tests."""
    if checkpoint is None:
        checkpoint = Path("/fake/final.ckpt")
    if tokenizer is None:
        tokenizer = Path("/fake/tokenizer_32k.bin")
    return adapter.HSPALMEvalAdapter(
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        max_seq_len=512,
    )


def _make_instance(context: str, continuation: str) -> MagicMock:
    """Create a mock lm_eval Instance with .args = (context, continuation)."""
    inst = MagicMock()
    inst.args = (context, continuation)
    return inst


def _score_per_choice_line(item_id: int, logprob: float, n_tokens: int, label: int = 0) -> str:
    """Build a single score-per-choice JSONL line (one choice = one continuation)."""
    return json.dumps({
        "item_id": str(item_id),
        "choices": [{"choice_idx": 0, "logprob": logprob, "n_tokens": n_tokens}],
        "label": label,
        "predicted": 0,
    })


def _aggregate_line(correct: int, total: int) -> str:
    acc = correct / total if total > 0 else 0.0
    return json.dumps({"accuracy": round(acc, 4), "correct": correct, "total": total})


def _make_binary_stdout(items: List[Tuple[float, int]], correct: int) -> str:
    """Produce stdout mimicking --mode score-per-choice for N single-choice items."""
    lines = []
    for i, (lp, nt) in enumerate(items):
        lines.append(_score_per_choice_line(i, lp, nt, label=0))
    lines.append(_aggregate_line(correct, len(items)))
    return "\n".join(lines) + "\n"


def _make_mock_proc(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


# ---------------------------------------------------------------------------
# (a) loglikelihood returns correct shape and types on a 2-instance fixture
# ---------------------------------------------------------------------------


def test_loglikelihood_returns_list_of_tuples() -> None:
    """loglikelihood must return a list of (float, bool) tuples, one per request."""
    lm = _make_lm_instance()
    requests = [
        _make_instance("The cat sat on", " the mat"),
        _make_instance("The dog ran to", " the park"),
    ]
    stdout = _make_binary_stdout([(-5.12, 3), (-7.34, 2)], correct=1)

    with patch("subprocess.run", return_value=_make_mock_proc(stdout)):
        results = lm.loglikelihood(requests)

    assert isinstance(results, list)
    assert len(results) == 2
    for item in results:
        assert isinstance(item, tuple), f"Expected tuple, got {type(item)}"
        assert len(item) == 2
        logprob, is_greedy = item
        assert isinstance(logprob, float), f"logprob must be float, got {type(logprob)}"
        assert isinstance(is_greedy, bool), f"is_greedy must be bool, got {type(is_greedy)}"


def test_loglikelihood_returns_correct_values() -> None:
    """loglikelihood must return the logprob from the per-choice JSONL row."""
    lm = _make_lm_instance()
    requests = [
        _make_instance("The cat sat on", " the mat"),
        _make_instance("The dog ran to", " the park"),
    ]
    expected_logprobs = [-5.123456, -7.654321]
    stdout = _make_binary_stdout(
        [(expected_logprobs[0], 3), (expected_logprobs[1], 2)],
        correct=1,
    )

    with patch("subprocess.run", return_value=_make_mock_proc(stdout)):
        results = lm.loglikelihood(requests)

    assert results[0][0] == pytest.approx(expected_logprobs[0], abs=1e-5)
    assert results[1][0] == pytest.approx(expected_logprobs[1], abs=1e-5)


def test_loglikelihood_is_greedy_always_false() -> None:
    """is_greedy must be False for all results (greedy decode not implemented)."""
    lm = _make_lm_instance()
    requests = [_make_instance("prompt", " continuation")]
    stdout = _make_binary_stdout([(-3.0, 1)], correct=0)

    with patch("subprocess.run", return_value=_make_mock_proc(stdout)):
        results = lm.loglikelihood(requests)

    assert results[0][1] is False


# ---------------------------------------------------------------------------
# (b) batch invocation produces the same logprobs as sequential
# ---------------------------------------------------------------------------


def test_loglikelihood_batch_equals_sequential() -> None:
    """Batching all requests in one binary call must match sequential single calls."""
    lm = _make_lm_instance()

    req_a = _make_instance("Context A", " cont A")
    req_b = _make_instance("Context B", " cont B")

    lp_a, lp_b = -4.5, -9.1

    # Single call with both requests
    stdout_batch = _make_binary_stdout([(lp_a, 2), (lp_b, 3)], correct=1)
    with patch("subprocess.run", return_value=_make_mock_proc(stdout_batch)):
        batch_results = lm.loglikelihood([req_a, req_b])

    # Two separate calls, one request each
    stdout_a = _make_binary_stdout([(lp_a, 2)], correct=0)
    stdout_b = _make_binary_stdout([(lp_b, 3)], correct=0)
    with patch("subprocess.run", side_effect=[
        _make_mock_proc(stdout_a),
        _make_mock_proc(stdout_b),
    ]):
        result_a = lm.loglikelihood([req_a])
        result_b = lm.loglikelihood([req_b])

    assert batch_results[0][0] == pytest.approx(result_a[0][0], abs=1e-5)
    assert batch_results[1][0] == pytest.approx(result_b[0][0], abs=1e-5)


def test_loglikelihood_single_binary_call_per_batch() -> None:
    """All requests in one loglikelihood() call must be packed into ONE binary invocation."""
    lm = _make_lm_instance()
    requests = [
        _make_instance(f"Context {i}", f" cont {i}")
        for i in range(5)
    ]
    stdout = _make_binary_stdout([(-float(i), 1) for i in range(1, 6)], correct=2)

    with patch("subprocess.run", return_value=_make_mock_proc(stdout)) as mock_run:
        lm.loglikelihood(requests)

    assert mock_run.call_count == 1, (
        f"Expected 1 binary call for 5 requests, got {mock_run.call_count}"
    )


# ---------------------------------------------------------------------------
# (c) JSONL parse of score-per-choice schema
# ---------------------------------------------------------------------------


def test_parse_score_per_choice_jsonl_basic() -> None:
    """parse_score_per_choice_output must extract per-choice logprobs correctly."""
    lines = [
        json.dumps({"item_id": "0", "choices": [{"choice_idx": 0, "logprob": -3.14, "n_tokens": 2}], "label": 0, "predicted": 0}),
        json.dumps({"item_id": "1", "choices": [{"choice_idx": 0, "logprob": -1.59, "n_tokens": 1}], "label": 0, "predicted": 0}),
        json.dumps({"accuracy": 0.5, "correct": 1, "total": 2}),
    ]
    stdout = "\n".join(lines) + "\n"

    result = adapter.parse_score_per_choice_output(stdout, n_items=2)

    assert len(result) == 2
    assert result[0] == pytest.approx(-3.14, abs=1e-5)
    assert result[1] == pytest.approx(-1.59, abs=1e-5)


def test_parse_score_per_choice_jsonl_skips_aggregate_line() -> None:
    """parse_score_per_choice_output must ignore the final aggregate line."""
    lines = [
        json.dumps({"item_id": "0", "choices": [{"choice_idx": 0, "logprob": -2.0, "n_tokens": 1}], "label": 0, "predicted": 0}),
        json.dumps({"accuracy": 1.0, "correct": 1, "total": 1}),
    ]
    stdout = "\n".join(lines) + "\n"

    result = adapter.parse_score_per_choice_output(stdout, n_items=1)

    assert len(result) == 1
    assert result[0] == pytest.approx(-2.0, abs=1e-5)


def test_parse_score_per_choice_jsonl_preserves_order() -> None:
    """parse_score_per_choice_output must return logprobs in item_id order."""
    logprobs = [-1.1, -2.2, -3.3, -4.4, -5.5]
    lines = [
        json.dumps({
            "item_id": str(i),
            "choices": [{"choice_idx": 0, "logprob": lp, "n_tokens": 1}],
            "label": 0,
            "predicted": 0,
        })
        for i, lp in enumerate(logprobs)
    ]
    lines.append(json.dumps({"accuracy": 0.0, "correct": 0, "total": 5}))
    stdout = "\n".join(lines) + "\n"

    result = adapter.parse_score_per_choice_output(stdout, n_items=5)

    for i, expected in enumerate(logprobs):
        assert result[i] == pytest.approx(expected, abs=1e-5)


def test_parse_score_per_choice_missing_items_raises() -> None:
    """parse_score_per_choice_output must raise ValueError if n_items don't match output."""
    stdout = json.dumps({"accuracy": 1.0, "correct": 1, "total": 1}) + "\n"

    with pytest.raises(ValueError, match="Expected 2 per-choice rows"):
        adapter.parse_score_per_choice_output(stdout, n_items=2)


# ---------------------------------------------------------------------------
# (d) loglikelihood_rolling raises NotImplementedError
# ---------------------------------------------------------------------------


def test_loglikelihood_rolling_raises_not_implemented() -> None:
    """loglikelihood_rolling is not needed for multiple_choice tasks; must raise."""
    lm = _make_lm_instance()
    with pytest.raises(NotImplementedError, match="loglikelihood_rolling"):
        lm.loglikelihood_rolling([_make_instance("some text", " cont")])


# ---------------------------------------------------------------------------
# (e) generate_until raises NotImplementedError
# ---------------------------------------------------------------------------


def test_generate_until_raises_not_implemented() -> None:
    """generate_until is Phase C; must raise NotImplementedError for now."""
    lm = _make_lm_instance()
    with pytest.raises(NotImplementedError, match="generate_until"):
        lm.generate_until([_make_instance("Generate:", " text")])


# ---------------------------------------------------------------------------
# Binary invocation wiring: verify --mode score-per-choice flag is used
# ---------------------------------------------------------------------------


def test_loglikelihood_uses_score_per_choice_mode() -> None:
    """Binary must be called with --mode score-per-choice, not --mode score."""
    lm = _make_lm_instance()
    requests = [_make_instance("prompt", " cont")]
    stdout = _make_binary_stdout([(-1.0, 1)], correct=0)

    with patch("subprocess.run", return_value=_make_mock_proc(stdout)) as mock_run:
        lm.loglikelihood(requests)

    cmd_args = mock_run.call_args[0][0]  # positional first arg = the command list
    assert "score-per-choice" in cmd_args, (
        f"Expected '--mode score-per-choice' in command, got: {cmd_args}"
    )
    assert "score" not in [a for a in cmd_args if a not in ("score-per-choice",)], (
        "Unexpected '--mode score' still present"
    )


def test_loglikelihood_jsonl_input_has_one_choice_per_request() -> None:
    """Each loglikelihood request must be packed as one JSONL line with one choice."""
    lm = _make_lm_instance()
    context = "The quick brown fox"
    continuation = " jumps over"
    requests = [_make_instance(context, continuation)]
    stdout = _make_binary_stdout([(-2.5, 2)], correct=0)

    captured_jsonl: list = []

    def capture_run(cmd, **kwargs):
        # Find --input file in cmd and read its contents
        try:
            idx = cmd.index("--input")
            input_path = Path(cmd[idx + 1])
            captured_jsonl.append(input_path.read_text())
        except (ValueError, IndexError, FileNotFoundError):
            pass
        return _make_mock_proc(stdout)

    with patch("subprocess.run", side_effect=capture_run):
        lm.loglikelihood(requests)

    assert len(captured_jsonl) == 1, "Expected exactly one binary call"
    line = captured_jsonl[0].strip().splitlines()[0]
    parsed = json.loads(line)
    # prompt = context, choices = [continuation], label = 0
    assert parsed["prompt"] == context
    assert parsed["choices"] == [continuation]
    assert parsed["label"] == 0


# ---------------------------------------------------------------------------
# Error handling: binary non-zero exit
# ---------------------------------------------------------------------------


def test_loglikelihood_raises_on_binary_failure() -> None:
    """loglikelihood must raise RuntimeError when the binary exits non-zero."""
    lm = _make_lm_instance()
    requests = [_make_instance("prompt", " cont")]

    with patch("subprocess.run", return_value=_make_mock_proc("", returncode=1, stderr="crash")):
        with pytest.raises(RuntimeError, match="eval_model"):
            lm.loglikelihood(requests)
