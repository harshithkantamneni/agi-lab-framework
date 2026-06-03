#!/usr/bin/env python3
"""HellaSwag Evaluation Driver -- stdlib only.

Downloads the HellaSwag validation set, formats it for our scoring binary,
runs scoring, and reports overall accuracy.

Usage:
    python tools/eval_hellaswag.py \
        --checkpoint data/checkpoints/cycle22_extend/step_002000.ckpt \
        --tokenizer  data/training/tokenizer_4k.bin \
        [--max-items 100]
"""

import argparse
import json
import os
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
HELLASWAG_DIR = SCORING_DIR / "hellaswag"
RESULTS_JSON = SCORING_DIR / "hellaswag_results.json"
SCORECARD_MD = SCORING_DIR / "scorecard.md"
SCORING_BINARY = PROJECT_ROOT / "build" / "eval_model"

# HellaSwag validation set (JSONL, one object per line)
HELLASWAG_URL = (
    "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/"
    "hellaswag_val.jsonl"
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_hellaswag(dest: Path) -> Path:
    """Download HellaSwag validation JSONL to *dest*. Return path to file."""
    dest.mkdir(parents=True, exist_ok=True)
    val_path = dest / "hellaswag_val.jsonl"

    if val_path.exists():
        size_mb = val_path.stat().st_size / 1e6
        print(f"[hellaswag] Already downloaded ({size_mb:.1f} MB), skipping.")
        return val_path

    print(f"[hellaswag] Downloading {HELLASWAG_URL} ...")
    try:
        urllib.request.urlretrieve(HELLASWAG_URL, str(val_path))
    except Exception as e:
        sys.exit(f"ERROR: Failed to download HellaSwag: {e}")

    size_mb = val_path.stat().st_size / 1e6
    print(f"[hellaswag] Downloaded {size_mb:.1f} MB")
    return val_path


# ---------------------------------------------------------------------------
# Parse HellaSwag JSONL
# ---------------------------------------------------------------------------
def parse_hellaswag(jsonl_path: Path, max_items: int | None = None) -> list[dict]:
    """Read HellaSwag JSONL, return list of dicts for scoring.

    Each HellaSwag item has:
      - activity_label: category (e.g. "Removing ice from car")
      - ctx_a: first sentence of context
      - ctx_b: continuation (beginning of sentence to complete)
      - ctx: full context (ctx_a + " " + ctx_b)
      - endings: list of 4 possible completions
      - label: int index of correct ending (0-3)
    """
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] Skipping malformed line {line_num}")
                continue

            ctx = obj.get("ctx", "")
            endings = obj.get("endings", [])
            label = obj.get("label", -1)

            # Validate
            if not ctx or len(endings) != 4:
                continue
            if not isinstance(label, int) or label < 0 or label > 3:
                # Some items have label as string
                try:
                    label = int(label)
                except (ValueError, TypeError):
                    continue
                if label < 0 or label > 3:
                    continue

            # Format as multiple-choice prompt
            # The context is the setup; endings are the 4 possible continuations
            prompt = (
                f"{ctx}\n"
                f"A. {endings[0]}\n"
                f"B. {endings[1]}\n"
                f"C. {endings[2]}\n"
                f"D. {endings[3]}\n"
                f"Answer:"
            )
            item = {
                "prompt": prompt,
                "choices": [" A", " B", " C", " D"],
                "label": label,
            }
            items.append(item)
            if max_items and len(items) >= max_items:
                break

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
# Update scorecard (append HellaSwag section, preserve MMLU)
# ---------------------------------------------------------------------------
def update_scorecard(results: dict) -> None:
    """Append HellaSwag section to scorecard.md, preserving existing content."""
    now_str = results.get("timestamp", datetime.now(timezone.utc).isoformat())
    overall = results.get("overall", {})
    acc = overall.get("accuracy", 0)
    correct = overall.get("correct", 0)
    total = overall.get("total", 0)
    checkpoint = results.get("checkpoint", "unknown")
    max_items = results.get("max_items", "all")

    hellaswag_section = [
        "",
        "### HellaSwag",
        f"- **Overall accuracy**: {acc:.4f} ({correct}/{total})",
        f"- **Checkpoint**: {checkpoint}",
        f"- **Max items**: {max_items}",
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

    # Remove any old HellaSwag section to avoid duplicates
    lines = existing.split("\n")
    new_lines = []
    skip = False
    for line in lines:
        if line.strip() == "### HellaSwag":
            skip = True
            continue
        if skip and line.strip().startswith("### "):
            skip = False
        if skip and line.strip().startswith("## "):
            skip = False
        if not skip:
            new_lines.append(line)

    # Find insertion point: after MMLU table (before ## Trend)
    insert_idx = len(new_lines)
    for i, line in enumerate(new_lines):
        if line.strip().startswith("## Trend"):
            insert_idx = i
            break

    # Insert HellaSwag section
    for j, hl in enumerate(hellaswag_section):
        new_lines.insert(insert_idx + j, hl)

    # Update the Trend table: add HellaSwag column if not present
    # Find the trend table and add/update the HellaSwag entry
    trend_start = None
    for i, line in enumerate(new_lines):
        if line.strip().startswith("## Trend"):
            trend_start = i
            break

    if trend_start is not None:
        # Check if there's already a HellaSwag column in the trend header
        header_idx = None
        for i in range(trend_start + 1, min(trend_start + 5, len(new_lines))):
            if "MMLU" in new_lines[i] and "HellaSwag" not in new_lines[i]:
                # Add HellaSwag column to header
                new_lines[i] = new_lines[i].rstrip(" |") + " | HellaSwag | Notes |"
                # Fix separator line too
                if i + 1 < len(new_lines) and new_lines[i + 1].startswith("|--"):
                    new_lines[i + 1] = new_lines[i + 1].rstrip(" |") + " | --------- |-------|"
                header_idx = i
                break
            elif "HellaSwag" in new_lines[i]:
                header_idx = i
                break

        # Append a trend row for this run
        date_str = now_str[:10]
        trend_row = f"| {date_str} | -- | {acc:.4f} | HellaSwag pipeline validation |"
        # Find end of trend table
        if header_idx is not None:
            last_row = header_idx + 2  # after header + separator
            for i in range(header_idx + 2, len(new_lines)):
                if new_lines[i].strip().startswith("|"):
                    last_row = i + 1
                else:
                    break
            new_lines.insert(last_row, trend_row)

    content = "\n".join(new_lines)
    # Clean up any triple+ blank lines
    while "\n\n\n\n" in content:
        content = content.replace("\n\n\n\n", "\n\n\n")

    SCORECARD_MD.write_text(content)
    print(f"[hellaswag] Scorecard updated at {SCORECARD_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="HellaSwag benchmark driver")
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
        val_path = download_hellaswag(HELLASWAG_DIR)
    else:
        val_path = HELLASWAG_DIR / "hellaswag_val.jsonl"
        if not val_path.exists():
            sys.exit(f"ERROR: --skip-download but {val_path} not found")

    # Parse
    max_items = args.max_items if args.max_items > 0 else None
    items = parse_hellaswag(val_path, max_items=max_items)
    if not items:
        sys.exit(f"ERROR: No valid items parsed from {val_path}")
    print(f"[hellaswag] Parsed {len(items)} items")

    # Chunk into batches to avoid overly long single runs
    BATCH_SIZE = 200
    total_correct = 0
    total_items = 0
    batch_num = 0

    with tempfile.TemporaryDirectory(prefix="hellaswag_scoring_") as tmpdir:
        for start in range(0, len(items), BATCH_SIZE):
            batch = items[start : start + BATCH_SIZE]
            batch_num += 1
            print(f"\n[hellaswag] === Batch {batch_num} ({len(batch)} items) ===")

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
    print(f"HellaSwag Overall: {total_correct}/{total_items} = {overall_acc:.4f}")
    print(f"{'='*60}")

    # Build full results
    now = datetime.now(timezone.utc).isoformat()
    full_results = {
        "benchmark": "HellaSwag",
        "timestamp": now,
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "max_items": max_items,
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
    print(f"[hellaswag] Results saved to {RESULTS_JSON}")

    # Update scorecard
    update_scorecard(full_results)


if __name__ == "__main__":
    main()
