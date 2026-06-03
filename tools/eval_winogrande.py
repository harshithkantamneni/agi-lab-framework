#!/usr/bin/env python3
"""WinoGrande Evaluation Driver -- stdlib only.

Downloads the WinoGrande validation set, formats coreference resolution
problems as binary choice scoring tasks, and reports accuracy.

WinoGrande tests commonsense reasoning via Winograd-schema-style fill-in-the-
blank sentences. Each item has a sentence with a blank (_) and two candidate
noun phrases. The model must pick which candidate makes the sentence coherent.

Target: 94.2% (Opus 4.6 score)

Usage:
    python tools/eval_winogrande.py \
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
WINOGRANDE_DIR = SCORING_DIR / "winogrande"
RESULTS_JSON = SCORING_DIR / "winogrande_results.json"
SCORECARD_MD = SCORING_DIR / "scorecard.md"
SCORING_BINARY = PROJECT_ROOT / "build" / "eval_model"

# HuggingFace datasets server API (paginated JSON, reliable)
WINOGRANDE_API_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=allenai/winogrande&config=winogrande_xl&split=validation"
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_winogrande(dest: Path) -> tuple[Path, "Path | None"]:
    """Download WinoGrande validation data via HuggingFace API.
    Return (data_path, None) -- labels are inline.
    """
    dest.mkdir(parents=True, exist_ok=True)
    data_path = dest / "winogrande_val.jsonl"

    if data_path.exists():
        size_kb = data_path.stat().st_size / 1e3
        print(f"[winogrande] Already downloaded ({size_kb:.1f} KB), skipping.")
        return data_path, None

    print("[winogrande] Downloading via HuggingFace datasets API...")
    try:
        rows = []
        offset = 0
        while True:
            api_url = f"{WINOGRANDE_API_URL}&offset={offset}&length=100"
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
        with open(data_path, "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"[winogrande] Downloaded {len(rows)} items")
        return data_path, None
    except Exception as e:
        print(f"[winogrande] API download failed: {e}")
        if data_path.exists():
            data_path.unlink()
        sys.exit("ERROR: Failed to download WinoGrande from all sources.")


# ---------------------------------------------------------------------------
# Parse WinoGrande JSONL
# ---------------------------------------------------------------------------
def parse_winogrande(
    data_path: Path,
    labels_path: Path | None = None,
    max_items: int | None = None,
) -> list[dict]:
    """Read WinoGrande data, return list of dicts for scoring.

    Each WinoGrande item has:
      - sentence: text with '_' placeholder (e.g. "The trophy doesn't fit in
        the brown suitcase because _ is too big.")
      - option1: first candidate (e.g. "trophy")
      - option2: second candidate (e.g. "suitcase")
      - answer: "1" or "2" (which option correctly fills the blank)

    We format as binary-choice scoring: the model sees two complete sentences
    (with option1 or option2 substituted) and picks the more likely one.
    """
    # Load labels if separate
    labels = []
    if labels_path and labels_path.exists():
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f if line.strip()]

    items = []
    skipped = 0

    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] Skipping malformed line {line_num + 1}")
                skipped += 1
                continue

            sentence = obj.get("sentence", "")
            option1 = obj.get("option1", "")
            option2 = obj.get("option2", "")

            # Get answer: inline or from labels file
            answer = obj.get("answer", "")
            if not answer and line_num < len(labels):
                answer = labels[line_num]

            # Validate
            if not sentence or not option1 or not option2:
                skipped += 1
                continue
            if "_" not in sentence:
                skipped += 1
                continue
            if answer not in ("1", "2"):
                skipped += 1
                continue

            # Build the two complete sentences
            sent_with_opt1 = sentence.replace("_", option1)
            sent_with_opt2 = sentence.replace("_", option2)

            # Format as binary choice for scoring
            # Label: 0 if option1 is correct (answer=="1"), 1 if option2
            label = 0 if answer == "1" else 1

            prompt = (
                f"Choose the word that best completes the sentence.\n\n"
                f"Sentence: {sentence}\n\n"
                f"A. {option1}\n"
                f"B. {option2}\n"
                f"Answer:"
            )
            item = {
                "prompt": prompt,
                "choices": [" A", " B"],
                "label": label,
            }
            items.append(item)
            if max_items and len(items) >= max_items:
                break

    if skipped:
        print(f"  [winogrande] Skipped {skipped} items (missing fields or invalid)")
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
# Update scorecard (append WinoGrande section, preserve existing)
# ---------------------------------------------------------------------------
def update_scorecard(results: dict) -> None:
    """Append WinoGrande section to scorecard.md, preserving existing content."""
    now_str = results.get("timestamp", datetime.now(timezone.utc).isoformat())
    overall = results.get("overall", {})
    acc = overall.get("accuracy", 0)
    correct = overall.get("correct", 0)
    total = overall.get("total", 0)
    checkpoint = results.get("checkpoint", "unknown")
    max_items = results.get("max_items", "all")

    winogrande_section = [
        "",
        "### WinoGrande",
        f"- **Overall accuracy**: {acc:.4f} ({correct}/{total})",
        f"- **Checkpoint**: {checkpoint}",
        f"- **Max items**: {max_items}",
        f"- **Target**: 94.2% (Opus 4.6)",
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

    # Remove any old WinoGrande section to avoid duplicates
    lines = existing.split("\n")
    new_lines = []
    skip = False
    for line in lines:
        if line.strip() == "### WinoGrande":
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

    # Insert WinoGrande section
    for j, sl in enumerate(winogrande_section):
        new_lines.insert(insert_idx + j, sl)

    content = "\n".join(new_lines)
    # Clean up excessive blank lines
    while "\n\n\n\n" in content:
        content = content.replace("\n\n\n\n", "\n\n\n")

    SCORECARD_MD.write_text(content)
    print(f"[winogrande] Scorecard updated at {SCORECARD_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="WinoGrande benchmark driver")
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
        data_path, labels_path = download_winogrande(WINOGRANDE_DIR)
    else:
        data_path = WINOGRANDE_DIR / "winogrande_val.jsonl"
        labels_path = WINOGRANDE_DIR / "winogrande_val_labels.lst"
        if not data_path.exists():
            sys.exit(f"ERROR: --skip-download but {data_path} not found")
        if not labels_path.exists():
            labels_path = None

    # Parse
    max_items = args.max_items if args.max_items > 0 else None
    items = parse_winogrande(data_path, labels_path, max_items=max_items)
    if not items:
        sys.exit(f"ERROR: No valid items parsed from {data_path}")
    print(f"[winogrande] Parsed {len(items)} items")

    # Chunk into batches
    BATCH_SIZE = 200
    total_correct = 0
    total_items = 0
    batch_num = 0

    with tempfile.TemporaryDirectory(prefix="winogrande_scoring_") as tmpdir:
        for start in range(0, len(items), BATCH_SIZE):
            batch = items[start : start + BATCH_SIZE]
            batch_num += 1
            print(f"\n[winogrande] === Batch {batch_num} ({len(batch)} items) ===")

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
    print(f"WinoGrande Overall: {total_correct}/{total_items} = {overall_acc:.4f}")
    print(f"Target: 94.2% (Opus 4.6)")
    print(f"Gap: {(overall_acc - 0.942) * 100:+.1f}%")
    print(f"{'='*60}")

    # Build full results
    now = datetime.now(timezone.utc).isoformat()
    full_results = {
        "benchmark": "WinoGrande",
        "timestamp": now,
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "max_items": max_items,
        "target": 0.942,
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
    print(f"[winogrande] Results saved to {RESULTS_JSON}")

    # Update scorecard
    update_scorecard(full_results)


if __name__ == "__main__":
    main()
