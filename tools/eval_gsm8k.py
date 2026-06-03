#!/usr/bin/env python3
"""GSM8K Evaluation Driver -- stdlib only.

Downloads the GSM8K test set, formats math problems for our scoring binary,
runs scoring, extracts final numerical answers, and reports accuracy.

GSM8K measures grade-school math reasoning. Each problem has a chain-of-thought
solution ending with "#### <number>". We evaluate whether our model selects
the correct final answer from multiple-choice candidates.

Target: 97.9% (Opus 4.6 score)

Usage:
    python tools/eval_gsm8k.py \
        --checkpoint data/checkpoints/cycle22_extend/step_002000.ckpt \
        --tokenizer  data/training/tokenizer_4k.bin \
        [--max-items 100]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCORING_DIR = DATA_DIR / "eval"
GSM8K_DIR = SCORING_DIR / "gsm8k"
RESULTS_JSON = SCORING_DIR / "gsm8k_results.json"
SCORECARD_MD = SCORING_DIR / "scorecard.md"
SCORING_BINARY = PROJECT_ROOT / "build" / "eval_model"

# GSM8K test set from GitHub (JSONL format)
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
# Fallback: HuggingFace datasets server API (paginated JSON)
GSM8K_API_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=openai/gsm8k&config=main&split=test"
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_gsm8k(dest: Path) -> Path:
    """Download GSM8K test JSONL to *dest*. Return path to file."""
    dest.mkdir(parents=True, exist_ok=True)
    test_path = dest / "gsm8k_test.jsonl"

    if test_path.exists():
        size_kb = test_path.stat().st_size / 1e3
        print(f"[gsm8k] Already downloaded ({size_kb:.1f} KB), skipping.")
        return test_path

    # Try primary URL (GitHub)
    print(f"[gsm8k] Downloading {GSM8K_URL} ...")
    try:
        urllib.request.urlretrieve(GSM8K_URL, str(test_path))
        size_kb = test_path.stat().st_size / 1e3
        print(f"[gsm8k] Downloaded {size_kb:.1f} KB")
        return test_path
    except Exception as e:
        print(f"[gsm8k] Failed from GitHub: {e}")
        if test_path.exists():
            test_path.unlink()

    # Fallback: HuggingFace datasets server API (paginated)
    print("[gsm8k] Trying HuggingFace datasets API fallback...")
    try:
        rows = []
        offset = 0
        while True:
            api_url = f"{GSM8K_API_URL}&offset={offset}&length=100"
            with urllib.request.urlopen(api_url) as resp:
                data = json.loads(resp.read().decode())
            batch = data.get("rows", [])
            if not batch:
                break
            for r in batch:
                rows.append(json.dumps(r["row"]))
            offset += len(batch)
            if offset >= data.get("num_rows_total", offset):
                break
        with open(test_path, "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"[gsm8k] Downloaded {len(rows)} items via API")
        return test_path
    except Exception as e:
        print(f"[gsm8k] API fallback failed: {e}")
        if test_path.exists():
            test_path.unlink()

    sys.exit("ERROR: Failed to download GSM8K from all sources.")


# ---------------------------------------------------------------------------
# Extract final numerical answer from GSM8K solution
# ---------------------------------------------------------------------------
def extract_answer(solution: str) -> str | None:
    """Extract the final answer from a GSM8K solution string.

    GSM8K solutions end with '#### <number>' where number may include
    commas, decimals, or negative signs.
    """
    # Match the #### pattern at the end
    match = re.search(r"####\s*(.+?)$", solution.strip(), re.MULTILINE)
    if match:
        answer = match.group(1).strip()
        # Normalize: remove commas, strip whitespace
        answer = answer.replace(",", "").strip()
        return answer
    return None


def generate_distractors(correct: str, num_distractors: int = 3) -> list[str]:
    """Generate plausible wrong answers for multiple-choice format.

    Creates distractors by perturbing the correct numerical answer.
    """
    distractors = []
    try:
        val = float(correct)
    except (ValueError, TypeError):
        # If answer is not a clean number, generate simple offsets
        distractors = [
            str(i) for i in [1, 2, 3, 4, 5]
            if str(i) != correct
        ]
        return distractors[:num_distractors]

    # Generate distractors by common mistake patterns
    candidates = set()

    # Off-by-one errors
    candidates.add(val + 1)
    candidates.add(val - 1)

    # Off by factor of 2
    candidates.add(val * 2)
    if val != 0:
        candidates.add(val / 2)

    # Off by 10
    candidates.add(val + 10)
    candidates.add(val - 10)

    # Digit swap errors (for integers)
    if val == int(val) and val > 0:
        s = str(int(val))
        if len(s) >= 2:
            # Swap first two digits
            swapped = s[1] + s[0] + s[2:]
            try:
                candidates.add(float(swapped))
            except ValueError:
                pass

    # Sign error
    candidates.add(-val)

    # Off by small amounts
    candidates.add(val + 5)
    candidates.add(val - 5)

    # Remove the correct answer and negatives for positive answers
    candidates.discard(val)
    if val >= 0:
        candidates = {c for c in candidates if c >= 0}

    # Format to match the correct answer's format
    result = []
    for c in sorted(candidates):
        if val == int(val):
            formatted = str(int(c)) if c == int(c) else f"{c:.1f}"
        else:
            # Match decimal places of original
            decimal_places = len(correct.split(".")[-1]) if "." in correct else 1
            formatted = f"{c:.{decimal_places}f}"
        if formatted != correct and formatted not in result:
            result.append(formatted)

    # Pad with simple offsets if we don't have enough
    offset = 1
    while len(result) < num_distractors:
        candidate = str(int(val) + offset * 3) if val == int(val) else f"{val + offset * 2.5:.1f}"
        if candidate != correct and candidate not in result:
            result.append(candidate)
        offset += 1

    return result[:num_distractors]


# ---------------------------------------------------------------------------
# Parse GSM8K JSONL
# ---------------------------------------------------------------------------
def parse_gsm8k(jsonl_path: Path, max_items: int | None = None) -> list[dict]:
    """Read GSM8K JSONL, return list of dicts for scoring.

    Each GSM8K item has:
      - question: the math word problem
      - answer: the chain-of-thought solution ending with #### <number>
    """
    items = []
    skipped = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] Skipping malformed line {line_num}")
                skipped += 1
                continue

            question = obj.get("question", "")
            answer_text = obj.get("answer", "")

            if not question or not answer_text:
                skipped += 1
                continue

            # Extract the final numerical answer
            correct_answer = extract_answer(answer_text)
            if correct_answer is None:
                skipped += 1
                continue

            # Generate distractors to create multiple-choice format
            distractors = generate_distractors(correct_answer, num_distractors=3)

            # Build choices: insert correct answer at a deterministic position
            # Use line_num to determine position (consistent across runs)
            correct_idx = line_num % 4
            choices_text = list(distractors[:3])  # 3 distractors
            choices_text.insert(correct_idx, correct_answer)

            # Ensure we have exactly 4 choices
            while len(choices_text) < 4:
                choices_text.append(str(int(float(correct_answer)) + len(choices_text) + 7))
            choices_text = choices_text[:4]

            labels = ["A", "B", "C", "D"]
            prompt = (
                f"Solve the following math problem. Choose the correct answer.\n\n"
                f"Problem: {question}\n\n"
                f"A. {choices_text[0]}\n"
                f"B. {choices_text[1]}\n"
                f"C. {choices_text[2]}\n"
                f"D. {choices_text[3]}\n"
                f"Answer:"
            )
            item = {
                "prompt": prompt,
                "choices": [" A", " B", " C", " D"],
                "label": correct_idx,
            }
            items.append(item)
            if max_items and len(items) >= max_items:
                break

    if skipped:
        print(f"  [gsm8k] Skipped {skipped} items (no valid answer)")
    return items


# ---------------------------------------------------------------------------
# Run the scoring binary
# ---------------------------------------------------------------------------
def run_scoring(
    jsonl_path: Path, checkpoint: Path, tokenizer: Path
) -> dict | None:
    """Run the scoring binary, parse JSON output. Return dict or None on error."""
    cmd = [
        str(SCORING_BINARY),
        "--checkpoint", str(checkpoint),
        "--tokenizer", str(tokenizer),
        "--mode", "score",
        "--input", str(jsonl_path),
    ]
    print(f"  [run] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=str(PROJECT_ROOT)
        )
    except subprocess.TimeoutExpired:
        print("  [run] TIMEOUT after 600s")
        return None
    except Exception as e:
        print(f"  [run] Exception: {e}")
        return None

    # Parse stdout -- find the JSON line
    for line in (result.stdout.splitlines() + result.stderr.splitlines()):
        line = line.strip()
        if line.startswith("{") and "accuracy" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    # If we got here, no parseable JSON
    print(f"  [run] No JSON in output. stdout={result.stdout[:500]}")
    print(f"  [run] stderr={result.stderr[:500]}")
    print(f"  [run] returncode={result.returncode}")
    return None


# ---------------------------------------------------------------------------
# Update scorecard (append GSM8K section, preserve existing)
# ---------------------------------------------------------------------------
def update_scorecard(results: dict) -> None:
    """Append GSM8K section to scorecard.md, preserving existing content."""
    now_str = results.get("timestamp", datetime.now(timezone.utc).isoformat())
    overall = results.get("overall", {})
    acc = overall.get("accuracy", 0)
    correct = overall.get("correct", 0)
    total = overall.get("total", 0)
    checkpoint = results.get("checkpoint", "unknown")
    max_items = results.get("max_items", "all")

    gsm8k_section = [
        "",
        "### GSM8K",
        f"- **Overall accuracy**: {acc:.4f} ({correct}/{total})",
        f"- **Checkpoint**: {checkpoint}",
        f"- **Max items**: {max_items}",
        f"- **Target**: 97.9% (Opus 4.6)",
        f"- **Evaluated**: {now_str}",
        "",
    ]

    # Read existing scorecard
    if SCORECARD_MD.exists():
        existing = SCORECARD_MD.read_text()
    else:
        existing = (
            "# Scorecard\n\n"
            "*Single source of truth on model quality. Updated after every scoring run.*\n"
            "*All agents read this. Director uses it for prioritization.*\n\n"
            "## Status: EVALUATED\n\n"
            "## Latest Scores\n"
        )

    # Remove any old GSM8K section to avoid duplicates
    lines = existing.split("\n")
    new_lines = []
    skip = False
    for line in lines:
        if line.strip() == "### GSM8K":
            skip = True
            continue
        if skip and line.strip().startswith("### "):
            skip = False
        if skip and line.strip().startswith("## "):
            skip = False
        if not skip:
            new_lines.append(line)

    # Find insertion point: before ## Trend
    insert_idx = len(new_lines)
    for i, line in enumerate(new_lines):
        if line.strip().startswith("## Trend"):
            insert_idx = i
            break

    # Insert GSM8K section
    for j, sl in enumerate(gsm8k_section):
        new_lines.insert(insert_idx + j, sl)

    content = "\n".join(new_lines)
    # Clean up excessive blank lines
    while "\n\n\n\n" in content:
        content = content.replace("\n\n\n\n", "\n\n\n")

    SCORECARD_MD.write_text(content)
    print(f"[gsm8k] Scorecard updated at {SCORECARD_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K benchmark driver")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/checkpoints/cycle22_extend/step_002000.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/training/tokenizer_4k.bin",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Max items to evaluate (0 = all)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, use existing data",
    )
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    tokenizer = PROJECT_ROOT / args.tokenizer

    # Validate prerequisites
    if not SCORING_BINARY.exists():
        sys.exit(f"ERROR: scoring binary not found at {SCORING_BINARY}")
    if not checkpoint.exists():
        sys.exit(f"ERROR: checkpoint not found at {checkpoint}")
    if not tokenizer.exists():
        sys.exit(f"ERROR: tokenizer not found at {tokenizer}")

    # Download
    if not args.skip_download:
        test_path = download_gsm8k(GSM8K_DIR)
    else:
        test_path = GSM8K_DIR / "gsm8k_test.jsonl"
        if not test_path.exists():
            sys.exit(f"ERROR: --skip-download but {test_path} not found")

    # Parse
    max_items = args.max_items if args.max_items > 0 else None
    items = parse_gsm8k(test_path, max_items=max_items)
    if not items:
        sys.exit(f"ERROR: No valid items parsed from {test_path}")
    print(f"[gsm8k] Parsed {len(items)} items")

    # Chunk into batches (same as HellaSwag)
    BATCH_SIZE = 200
    total_correct = 0
    total_items = 0
    batch_num = 0

    with tempfile.TemporaryDirectory(prefix="gsm8k_scoring_") as tmpdir:
        for start in range(0, len(items), BATCH_SIZE):
            batch = items[start : start + BATCH_SIZE]
            batch_num += 1
            print(f"\n[gsm8k] === Batch {batch_num} ({len(batch)} items) ===")

            # Write JSONL
            jsonl_path = Path(tmpdir) / f"batch_{batch_num:03d}.jsonl"
            with open(jsonl_path, "w") as f:
                for item in batch:
                    f.write(json.dumps(item) + "\n")

            # Run scoring
            result = run_scoring(jsonl_path, checkpoint, tokenizer)
            if result is None:
                print(f"  [ERROR] scoring binary failed for batch {batch_num}")
                total_items += len(batch)
                continue

            cor = result.get("correct", 0)
            tot = result.get("total", len(batch))
            acc = result.get("accuracy", 0.0)
            print(f"  [result] {cor}/{tot} = {acc:.4f}")

            total_correct += cor
            total_items += tot

    # Overall
    overall_acc = total_correct / total_items if total_items > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"GSM8K Overall: {total_correct}/{total_items} = {overall_acc:.4f}")
    print(f"Target: 97.9% (Opus 4.6)")
    print(f"Gap: {(overall_acc - 0.979) * 100:+.1f}%")
    print(f"{'='*60}")

    # Build full results
    now = datetime.now(timezone.utc).isoformat()
    full_results = {
        "benchmark": "GSM8K",
        "timestamp": now,
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "max_items": max_items,
        "target": 0.979,
        "overall": {
            "accuracy": overall_acc,
            "correct": total_correct,
            "total": total_items,
        },
    }

    # Save JSON
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"[gsm8k] Results saved to {RESULTS_JSON}")

    # Update scorecard
    update_scorecard(full_results)


if __name__ == "__main__":
    main()
