"""HSPA lm-eval adapter — bridges HSPA checkpoints to lm-evaluation-harness v0.4.3.

Phase B implementation: HSPALMEvalAdapter subclasses lm_eval.api.model.LM and
implements loglikelihood(requests) using --mode score-per-choice binary output.
This enables the harness to compute acc_norm natively from per-choice logprobs,
resolving the HellaSwag methodology gap documented in phase11_apparatus_methodology_gap.md.

Harness commit pin: 3fa4fd725c8a428710109f1d6c14eda37e95baea (DO NOT CHANGE)

CLI usage (legacy smoke test — MMLU only):
    python3 tools/hspa_lm_eval_adapter.py \\
        --checkpoint data/checkpoints/phase3_factorial/A42/final.ckpt \\
        --tokenizer data/training/tokenizer_32k.bin \\
        --task mmlu \\
        --limit 100 \\
        --num-fewshot 5 \\
        --output data/eval/p11/smoke_a42.json

For full eval-p11 harness integration, use eval_harness_p11.py --arm <cell_id>.
"""

# NOTE: lm_eval import is optional at module load — tests mock the binary, not the base class.
# The harness is only needed when simple_evaluate() is called. Import lazily in HSPALMEvalAdapter.

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repository root: two levels up from tools/
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_BINARY = REPO_ROOT / "build" / "eval_model"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data/checkpoints/phase3_factorial/A42/final.ckpt"
)
DEFAULT_TOKENIZER = REPO_ROOT / "data/training/tokenizer_32k.bin"
MAX_SEQ_LEN = 512  # left-truncation limit (§5 apparatus spec)

# Byte budget for a single JSONL batch sent to the eval binary.
# The binary enforces MAX_INPUT_SIZE = 16 MB (src/eval/eval_model.c:37);
# 12 MB provides a 4 MB safety margin to account for serialization variance.
BATCH_MAX_BYTES = 12 * 1024 * 1024  # 12 MB

_log = logging.getLogger(__name__)

ANSWER_LABELS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Score-per-choice JSONL parser
# ---------------------------------------------------------------------------


def parse_score_per_choice_output(stdout: str, n_items: int) -> List[float]:
    """Parse --mode score-per-choice stdout and return per-item logprobs.

    Each per-choice line has schema:
        {"item_id": "N", "choices": [{"choice_idx": 0, "logprob": F, "n_tokens": K}],
         "label": L, "predicted": P}

    For loglikelihood requests, each item has exactly one choice (the continuation).
    The final line is an aggregate {"accuracy": X, "correct": N, "total": N} — skipped.

    Args:
        stdout: Full stdout string from the binary.
        n_items: Expected number of per-choice rows.

    Returns:
        List of logprobs (float), one per item, in item_id order.

    Raises:
        ValueError: If the number of parsed per-choice rows != n_items.
    """
    per_choice_rows: Dict[int, float] = {}

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Per-choice row: has "item_id" and "choices" keys
        if "item_id" in obj and "choices" in obj:
            item_id = int(obj["item_id"])
            choices = obj["choices"]
            # For loglikelihood, each request packs as one choice (choice_idx=0)
            if choices:
                per_choice_rows[item_id] = float(choices[0]["logprob"])
        # Aggregate line ("accuracy", "correct", "total") — skip

    if len(per_choice_rows) != n_items:
        raise ValueError(
            f"Expected {n_items} per-choice rows, got {len(per_choice_rows)}. "
            f"Stdout (first 500 chars): {stdout[:500]}"
        )

    # Return in item_id order (0, 1, 2, ...)
    return [per_choice_rows[i] for i in range(n_items)]


# ---------------------------------------------------------------------------
# Byte-budget batching helper
# ---------------------------------------------------------------------------


def _batch_requests_under_byte_budget(
    items: List[Dict[str, Any]],
    budget: int = BATCH_MAX_BYTES,
) -> Generator[List[Dict[str, Any]], None, None]:
    """Yield non-empty lists of items whose total JSONL byte size stays under budget.

    Each item is serialised as ``json.dumps(item, ensure_ascii=False) + "\\n"``
    — the same encoding used when writing the tempfile — so the byte count is
    exact.  Edge cases:

    - Empty ``items``: yields nothing.
    - A single item whose serialised form exceeds ``budget``: yields it as a
      singleton batch and emits a WARNING.  The downstream binary will fail
      loudly for that batch (which is correct — the binary's own hard-cap at
      src/eval/eval_model.c:37 is the last line of defence).

    Order is fully preserved: the i-th element of every yielded batch list
    corresponds to the same element in the original ``items`` sequence.

    Args:
        items:  Sequence of item dicts to batch.
        budget: Maximum number of bytes (inclusive) per batch.  Defaults to
                ``BATCH_MAX_BYTES``.

    Yields:
        Non-empty lists of item dicts.
    """
    if not items:
        return

    batch: List[Dict[str, Any]] = []
    batch_bytes = 0

    for item in items:
        serialised = json.dumps(item, ensure_ascii=False) + "\n"
        item_bytes = len(serialised.encode("utf-8"))

        if item_bytes > budget:
            # Oversized singleton: flush any pending batch first, then yield
            # this item alone.  Log at WARNING level — the binary will surface
            # the real error if the file truly exceeds its own cap.
            if batch:
                yield batch
                batch = []
                batch_bytes = 0
            _log.warning(
                "_batch_requests_under_byte_budget: single item exceeds budget "
                "(%d bytes > %d bytes budget); yielding as singleton batch.",
                item_bytes,
                budget,
            )
            yield [item]
            continue

        if batch and batch_bytes + item_bytes > budget:
            # Current item would push this batch over budget — emit now.
            yield batch
            batch = []
            batch_bytes = 0

        batch.append(item)
        batch_bytes += item_bytes

    if batch:
        yield batch


# ---------------------------------------------------------------------------
# LM subclass: HSPALMEvalAdapter
# ---------------------------------------------------------------------------


def _get_lm_base():
    """Return lm_eval.api.model.LM, importing lazily so unit tests don't require harness."""
    from lm_eval.api.model import LM  # type: ignore
    return LM


class _LMBase:
    """Fallback base class when lm_eval is not importable (unit-test environments)."""
    def __init__(self) -> None:
        self._rank = 0
        self._world_size = 1


def _make_hspa_base():
    """Return appropriate base class — LM if harness available, _LMBase otherwise."""
    try:
        return _get_lm_base()
    except ImportError:
        return _LMBase


class HSPALMEvalAdapter(_make_hspa_base()):
    """lm_eval.api.model.LM subclass wrapping the HSPA C binary.

    Implements loglikelihood() via --mode score-per-choice; harness computes
    acc_norm internally from the returned logprobs + n_tokens info.

    loglikelihood_rolling() and generate_until() are NotImplementedError shims:
    - HellaSwag/WinoGrande/MMLU use output_type: multiple_choice → loglikelihood path
    - GSM8K generate_until() deferred to Phase C
    """

    def __init__(
        self,
        checkpoint: Path,
        tokenizer: Path,
        max_seq_len: int = MAX_SEQ_LEN,
        debug_token_window: bool = False,
    ) -> None:
        super().__init__()
        self.checkpoint = Path(checkpoint)
        self.tokenizer = Path(tokenizer)
        self.max_seq_len = max_seq_len
        self.debug_token_window = debug_token_window

    # --- lm_eval.api.model.LM compatibility shims ---

    def loglikelihood(self, requests: List[Any]) -> List[Tuple[float, bool]]:
        """Compute loglikelihood of each (context, continuation) request.

        Batches requests into JSONL files that each stay under BATCH_MAX_BYTES
        (12 MB, 4 MB below the binary's 16 MB hard cap at eval_model.c:37).
        Invokes the binary once per batch via --mode score-per-choice and
        concatenates the per-choice logprobs in order.  A single tempfile is
        created and cleaned up per batch, so the existing try/finally cleanup
        discipline is preserved.

        Each request becomes one JSONL line:
            {"prompt": context, "choices": [continuation], "label": 0}

        Returns:
            List of (logprob: float, is_greedy: bool) tuples in the same order
            as ``requests``.  is_greedy is always False.

        Raises:
            RuntimeError: If any batch binary invocation exits non-zero.
        """
        if not requests:
            return []

        # Build item list: one dict per request, one choice (the continuation)
        items = []
        for req in requests:
            context, continuation = req.args[0], req.args[1]
            items.append({
                "prompt": context,
                "choices": [continuation],
                "label": 0,
            })

        logprobs: List[float] = []

        for batch in _batch_requests_under_byte_budget(items, BATCH_MAX_BYTES):
            tmp_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".jsonl",
                    prefix="hspa_lm_",
                    delete=False,
                    encoding="utf-8",
                ) as f:
                    tmp_path = Path(f.name)
                    for item in batch:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                cmd = [
                    str(EVAL_BINARY),
                    "--checkpoint", str(self.checkpoint),
                    "--tokenizer", str(self.tokenizer),
                    "--mode", "score-per-choice",
                    "--input", str(tmp_path),
                    "--max-seq-len", str(self.max_seq_len),
                ]
                if self.debug_token_window:
                    cmd.append("--debug-token-window")

                proc = subprocess.run(cmd, capture_output=True, text=True)

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"eval_model exited with rc={proc.returncode}. "
                        f"stderr: {proc.stderr[-2000:]}"
                    )

                batch_logprobs = parse_score_per_choice_output(
                    proc.stdout, n_items=len(batch)
                )
                logprobs.extend(batch_logprobs)
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

        return [(lp, False) for lp in logprobs]

    def loglikelihood_rolling(self, requests: List[Any]) -> List[Tuple[float]]:
        """Not implemented — HellaSwag/WinoGrande use multiple_choice, not rolling.

        Rolling loglikelihood is only needed for perplexity/LM-completion tasks.
        Per phase11_apparatus_methodology_gap.md §5 Phase B, this is a shim.
        """
        raise NotImplementedError(
            "loglikelihood_rolling is not implemented for the HSPA adapter. "
            "HellaSwag and WinoGrande use output_type: multiple_choice which "
            "routes through loglikelihood(), not loglikelihood_rolling()."
        )

    def generate_until(self, requests: List[Any]) -> List[str]:
        """Not implemented — Phase C (GSM8K-specific); deferred per pre-reg §4.5.

        GSM8K is excluded from Phase B launch per the exclusion clause.
        """
        raise NotImplementedError(
            "generate_until is not implemented (Phase C — GSM8K deferred). "
            "Per pre-reg §4.5 exclusion clause, max-arm GSM8K is PARTIAL_BLOCKED."
        )

# ---------------------------------------------------------------------------
# MMLU dataset helpers (preserved from v1 for legacy smoke CLI)
# ---------------------------------------------------------------------------

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


def _subject_few_shot_examples(
    subject: str, n_shot: int, split: str = "dev"
) -> List[Dict[str, Any]]:
    """Return up to n_shot examples from the dev split of a subject."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("cais/mmlu", subject, split=split, streaming=True)
    examples = []
    for ex in ds:
        examples.append(ex)
        if len(examples) >= n_shot:
            break
    return examples


def _format_few_shot_prefix(examples: List[Dict[str, Any]], subject: str) -> str:
    """Format few-shot examples into a prompt prefix string."""
    subject_pretty = subject.replace("_", " ")
    lines = [
        f"The following are multiple choice questions (with answers) about "
        f"{subject_pretty}.\n"
    ]
    for ex in examples:
        q = ex["question"].strip()
        choices_text = "\n".join(
            f"{ANSWER_LABELS[i]}. {ex['choices'][i]}"
            for i in range(len(ex["choices"]))
        )
        answer_letter = ANSWER_LABELS[ex["answer"]]
        lines.append(f"Question: {q}\n{choices_text}\nAnswer: {answer_letter}\n")
    return "\n".join(lines) + "\n"


def _format_query_prompt(example: Dict[str, Any]) -> str:
    """Format a single query (without answer) from an MMLU example."""
    q = example["question"].strip()
    choices_text = "\n".join(
        f"{ANSWER_LABELS[i]}. {example['choices'][i]}"
        for i in range(len(example["choices"]))
    )
    return f"Question: {q}\n{choices_text}\nAnswer:"


def load_mmlu_items(
    limit: int = 100,
    num_fewshot: int = 5,
    subjects: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return up to `limit` MMLU validation examples formatted for the binary.

    Each item is a dict:
        prompt: str     — few-shot prefix + query (without answer letter)
        choices: list   — list of single-letter strings (" A", " B", " C", " D")
        label: int      — correct choice index (0-3)
        subject: str    — MMLU subject name
    """
    from datasets import load_dataset  # type: ignore

    if subjects is None:
        subjects = MMLU_SUBJECTS

    items: List[Dict[str, Any]] = []
    per_subject = max(1, math.ceil(limit / len(subjects)))

    for subject in subjects:
        if len(items) >= limit:
            break

        if num_fewshot > 0:
            try:
                few_shot_exs = _subject_few_shot_examples(subject, num_fewshot, "dev")
            except Exception:
                few_shot_exs = []
            prefix = _format_few_shot_prefix(few_shot_exs, subject)
        else:
            prefix = ""

        try:
            ds = load_dataset(
                "cais/mmlu", subject, split="validation", streaming=True
            )
        except Exception:
            continue

        count = 0
        for ex in ds:
            if count >= per_subject or len(items) >= limit:
                break
            query_prompt = _format_query_prompt(ex)
            prompt = prefix + query_prompt
            choices = [f" {ANSWER_LABELS[i]}" for i in range(4)]
            items.append(
                {
                    "prompt": prompt,
                    "choices": choices,
                    "label": ex["answer"],
                    "subject": subject,
                }
            )
            count += 1

    return items[:limit]


# ---------------------------------------------------------------------------
# Binary invocation helpers (preserved for legacy smoke CLI / batched scoring)
# ---------------------------------------------------------------------------


def format_as_jsonl(items: List[Dict[str, Any]]) -> str:
    """Serialise a list of items to JSONL for eval_model --mode score or score-per-choice.

    Each line: {"prompt": "...", "choices": ["...", ...], "label": N}
    """
    lines = []
    for item in items:
        line = json.dumps(
            {
                "prompt": item["prompt"],
                "choices": item["choices"],
                "label": item["label"],
            },
            ensure_ascii=False,
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


def run_eval_binary(
    checkpoint: Path,
    tokenizer: Path,
    jsonl_path: Path,
    max_seq_len: int = MAX_SEQ_LEN,
    mode: str = "score",
) -> Dict[str, Any]:
    """Run build/eval_model and return parsed result dict.

    For mode='score': parses aggregate {"accuracy", "correct", "total"}.
    For mode='score-per-choice': same aggregate (last line).

    Returns:
        {
            "accuracy": float,
            "correct": int,
            "total": int,
            "stderr": str,
            "wall_clock_s": float,
            "rss_peak_mb": float,
            "returncode": int,
        }
    """
    cmd = [
        str(EVAL_BINARY),
        "--checkpoint", str(checkpoint),
        "--tokenizer", str(tokenizer),
        "--mode", mode,
        "--input", str(jsonl_path),
        "--max-seq-len", str(max_seq_len),
    ]

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_clock = time.monotonic() - t0

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    rss_peak_mb = usage.ru_maxrss / (1024 * 1024)

    result: Dict[str, Any] = {
        "stderr": proc.stderr,
        "wall_clock_s": round(wall_clock, 3),
        "rss_peak_mb": round(rss_peak_mb, 1),
        "returncode": proc.returncode,
    }

    if proc.returncode != 0:
        result["accuracy"] = None
        result["correct"] = None
        result["total"] = None
        result["error"] = f"Binary exited with rc={proc.returncode}. stderr: {proc.stderr[-2000:]}"
        return result

    # Parse the last JSON line (aggregate)
    stdout = proc.stdout.strip()
    last_line = ""
    for line in reversed(stdout.splitlines()):
        if line.strip():
            last_line = line.strip()
            break

    try:
        parsed = json.loads(last_line)
        result["accuracy"] = parsed["accuracy"]
        result["correct"] = parsed["correct"]
        result["total"] = parsed["total"]
    except (json.JSONDecodeError, KeyError) as exc:
        result["accuracy"] = None
        result["correct"] = None
        result["total"] = None
        result["error"] = f"Failed to parse binary stdout: {exc!r}. stdout: {stdout[:500]}"

    return result


def run_eval_in_batches(
    items: List[Dict[str, Any]],
    checkpoint: Path,
    tokenizer: Path,
    batch_size: int = 25,
    max_seq_len: int = MAX_SEQ_LEN,
    tmpdir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run eval_model over `items` in batches, recording per-batch timing.

    Returns aggregate result dict with per_batch_timings_s added.
    Used by the legacy MMLU smoke CLI.
    """
    if tmpdir is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="hspa_eval_"))

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    total_correct = 0
    total_items = 0
    batch_timings: List[float] = []
    all_stderr: List[str] = []
    rss_peak_mb = 0.0

    for b_idx, batch in enumerate(batches):
        jsonl_path = tmpdir / f"batch_{b_idx:04d}.jsonl"
        jsonl_path.write_text(format_as_jsonl(batch), encoding="utf-8")

        res = run_eval_binary(checkpoint, tokenizer, jsonl_path, max_seq_len)
        batch_timings.append(res["wall_clock_s"])
        all_stderr.append(f"[batch {b_idx}] {res['stderr']}")
        rss_peak_mb = max(rss_peak_mb, res["rss_peak_mb"])

        if res.get("error"):
            return {
                "error": res["error"],
                "batch_index_failed": b_idx,
                "batch_timings_s": batch_timings,
                "stderr": "\n".join(all_stderr),
            }

        total_correct += res["correct"]
        total_items += res["total"]

    acc = total_correct / total_items if total_items > 0 else 0.0
    stderr_acc = math.sqrt(acc * (1.0 - acc) / total_items) if total_items > 1 else 0.0

    return {
        "accuracy": round(acc, 4),
        "stderr": round(stderr_acc, 4),
        "correct": total_correct,
        "total": total_items,
        "wall_clock_s": round(sum(batch_timings), 3),
        "rss_peak_mb": round(rss_peak_mb, 1),
        "per_batch_timings_s": [round(t, 3) for t in batch_timings],
        "binary_stderr": "\n".join(all_stderr),
    }


# ---------------------------------------------------------------------------
# CLI entry point (legacy MMLU smoke test)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HSPA lm-eval adapter — legacy MMLU smoke test via eval_model binary."
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--task", default="mmlu", choices=["mmlu"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--subjects", nargs="+", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not EVAL_BINARY.exists():
        print(f"BLOCK: eval_model binary not found at {EVAL_BINARY}", file=sys.stderr)
        return 1
    if not args.checkpoint.exists():
        print(f"BLOCK: checkpoint not found at {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.tokenizer.exists():
        print(f"BLOCK: tokenizer not found at {args.tokenizer}", file=sys.stderr)
        return 1

    print(f"Loading {args.limit} MMLU items ({args.num_fewshot}-shot) ...", file=sys.stderr)
    t_load_start = time.monotonic()
    items = load_mmlu_items(
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        subjects=args.subjects,
    )
    t_load = time.monotonic() - t_load_start
    print(
        f"Loaded {len(items)} items from {len(set(x['subject'] for x in items))} subjects "
        f"in {t_load:.1f}s.",
        file=sys.stderr,
    )

    if len(items) == 0:
        print("BLOCK: no MMLU items loaded — dataset unavailable?", file=sys.stderr)
        return 1

    result = run_eval_in_batches(
        items=items,
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    if "error" in result:
        print(f"BLOCK: eval failed — {result['error']}", file=sys.stderr)
        print(json.dumps(result, indent=2))
        return 1

    result["metadata"] = {
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "task": args.task,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "n_subjects": len(set(x["subject"] for x in items)),
        "adapter_version": "hspa_lm_eval_adapter_v2_phase_b",
    }

    output_json = json.dumps(result, indent=2)
    print(output_json)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        print(f"Result written to {args.output}", file=sys.stderr)

    acc_pct = result["accuracy"] * 100
    print(
        f"\nSMOKE TEST SUMMARY: acc={acc_pct:.1f}% "
        f"(+/-{result['stderr']*100:.1f}%) "
        f"correct={result['correct']}/{result['total']} "
        f"wall={result['wall_clock_s']:.1f}s "
        f"RSS_peak={result['rss_peak_mb']:.0f}MB",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
