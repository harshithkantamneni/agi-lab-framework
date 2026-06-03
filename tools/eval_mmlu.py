#!/usr/bin/env python3
"""MMLU Evaluation Driver — stdlib only.

Downloads the MMLU test set, formats it for our scoring binary,
runs scoring, and reports per-subject + overall accuracy.

Usage:
    python tools/eval_mmlu.py \
        --checkpoint data/checkpoints/cycle22_extend/step_002000.ckpt \
        --tokenizer  data/training/tokenizer_4k.bin \
        [--subjects abstract_algebra,anatomy] \
        [--max-per-subject 5]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tarfile
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
MMLU_DIR = SCORING_DIR / "mmlu"
RESULTS_JSON = SCORING_DIR / "mmlu_results.json"
SCORECARD_MD = SCORING_DIR / "scorecard.md"
SCORING_BINARY = PROJECT_ROOT / "build" / "eval_model"

MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------
def download_mmlu(dest: Path) -> Path:
    """Download and extract the MMLU tar to *dest*. Return path to data/ root."""
    tar_path = dest / "data.tar"
    extracted_marker = dest / ".extracted"

    # If already extracted, skip
    if extracted_marker.exists():
        print("[mmlu] Already extracted, skipping download.")
        return _find_data_root(dest)

    dest.mkdir(parents=True, exist_ok=True)

    if not tar_path.exists():
        print(f"[mmlu] Downloading {MMLU_URL} ...")
        urllib.request.urlretrieve(MMLU_URL, str(tar_path))
        print(f"[mmlu] Downloaded {tar_path.stat().st_size / 1e6:.1f} MB")
    else:
        print(f"[mmlu] Tar already present ({tar_path.stat().st_size / 1e6:.1f} MB)")

    print("[mmlu] Extracting ...")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=str(dest))

    extracted_marker.touch()
    print("[mmlu] Extraction complete.")
    return _find_data_root(dest)


def _find_data_root(dest: Path) -> Path:
    """Locate the actual data/ directory inside the extraction."""
    # The tar extracts to data/ which contains subject dirs like
    # data/test/abstract_algebra_test.csv  or  data/abstract_algebra/test.csv
    candidate = dest / "data"
    if candidate.is_dir():
        # Check for "test" subdirectory (some MMLU tar layouts)
        test_sub = candidate / "test"
        if test_sub.is_dir():
            return test_sub
        # Otherwise subject dirs are immediate children
        return candidate
    # Fallback: dest itself
    return dest


# ---------------------------------------------------------------------------
# Discover subjects
# ---------------------------------------------------------------------------
def discover_subjects(data_root: Path) -> dict[str, Path]:
    """Return {subject_name: path_to_test_csv}."""
    subjects: dict[str, Path] = {}

    # Layout A:  data_root/<subject>/test.csv
    for d in sorted(data_root.iterdir()):
        if d.is_dir():
            test_csv = d / "test.csv"
            if test_csv.exists():
                subjects[d.name] = test_csv

    # Layout B:  data_root/<subject>_test.csv  (flat)
    if not subjects:
        for f in sorted(data_root.iterdir()):
            if f.is_file() and f.name.endswith("_test.csv"):
                name = f.name.replace("_test.csv", "")
                subjects[name] = f

    # Layout C: data_root is test/ and contains <subject>_test.csv
    if not subjects:
        for f in sorted(data_root.iterdir()):
            if f.is_file() and f.suffix == ".csv":
                name = f.stem
                if name.endswith("_test"):
                    name = name[: -len("_test")]
                subjects[name] = f

    return subjects


# ---------------------------------------------------------------------------
# Format a subject into JSONL
# ---------------------------------------------------------------------------
def format_subject_jsonl(csv_path: Path, max_items: int | None = None) -> list[dict]:
    """Read MMLU CSV, return list of dicts ready for JSONL."""
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            question = row[0]
            choices_text = row[1:5]  # A, B, C, D
            answer_letter = row[5].strip().upper()
            if answer_letter not in LETTER_TO_IDX:
                continue

            prompt = (
                f"Question: {question}\n"
                f"A. {choices_text[0]}\n"
                f"B. {choices_text[1]}\n"
                f"C. {choices_text[2]}\n"
                f"D. {choices_text[3]}\n"
                f"Answer:"
            )
            item = {
                "prompt": prompt,
                "choices": [" A", " B", " C", " D"],
                "label": LETTER_TO_IDX[answer_letter],
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
            cmd, capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT)
        )
    except subprocess.TimeoutExpired:
        print("  [run] TIMEOUT after 300s")
        return None
    except Exception as e:
        print(f"  [run] Exception: {e}")
        return None

    # Parse stdout — find the JSON line
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
# Write scorecard
# ---------------------------------------------------------------------------
def write_scorecard(results: dict) -> None:
    """Overwrite scorecard.md with latest results."""
    now_str = results.get("timestamp", datetime.now(timezone.utc).isoformat())
    overall = results.get("overall", {})
    subjects = results.get("subjects", {})

    lines = [
        "# Scorecard",
        "",
        "*Single source of truth on model quality. Updated after every scoring run.*",
        "*All agents read this. Director uses it for prioritization.*",
        "",
        f"## Status: EVALUATED ({now_str})",
        "",
        "## Latest Scores",
        "",
        "### MMLU",
        f"- **Overall accuracy**: {overall.get('accuracy', 0):.4f} "
        f"({overall.get('correct', 0)}/{overall.get('total', 0)})",
        f"- **Subjects tested**: {len(subjects)}",
        f"- **Checkpoint**: {results.get('checkpoint', 'unknown')}",
        f"- **Max per subject**: {results.get('max_per_subject', 'all')}",
        "",
        "| Subject | Accuracy | Correct | Total |",
        "|---------|----------|---------|-------|",
    ]
    for subj in sorted(subjects.keys()):
        s = subjects[subj]
        acc = s.get("accuracy", 0)
        cor = s.get("correct", 0)
        tot = s.get("total", 0)
        lines.append(f"| {subj} | {acc:.4f} | {cor} | {tot} |")

    lines += [
        "",
        "## Trend",
        f"| Date | MMLU | Notes |",
        f"|------|------|-------|",
        f"| {now_str[:10]} | {overall.get('accuracy', 0):.4f} | First pipeline validation (48M params, 1M tokens) |",
        "",
    ]

    SCORECARD_MD.write_text("\n".join(lines) + "\n")
    print(f"[mmlu] Scorecard written to {SCORECARD_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MMLU benchmark driver")
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
        "--subjects",
        type=str,
        default="",
        help="Comma-separated subject list (default: all)",
    )
    parser.add_argument(
        "--max-per-subject",
        type=int,
        default=0,
        help="Max questions per subject (0 = all)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, use existing data",
    )
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    tokenizer = PROJECT_ROOT / args.tokenizer

    # Validate
    if not SCORING_BINARY.exists():
        sys.exit(f"ERROR: scoring binary not found at {SCORING_BINARY}")
    if not checkpoint.exists():
        sys.exit(f"ERROR: checkpoint not found at {checkpoint}")
    if not tokenizer.exists():
        sys.exit(f"ERROR: tokenizer not found at {tokenizer}")

    # Download / extract
    data_root = download_mmlu(MMLU_DIR)
    print(f"[mmlu] Data root: {data_root}")

    # Discover subjects
    all_subjects = discover_subjects(data_root)
    if not all_subjects:
        sys.exit(f"ERROR: No subjects found under {data_root}")
    print(f"[mmlu] Found {len(all_subjects)} subjects")

    # Filter subjects
    if args.subjects:
        requested = [s.strip() for s in args.subjects.split(",")]
        filtered = {k: v for k, v in all_subjects.items() if k in requested}
        if not filtered:
            available = list(all_subjects.keys())[:20]
            sys.exit(
                f"ERROR: None of {requested} found. Available: {available}..."
            )
        all_subjects = filtered

    max_per = args.max_per_subject if args.max_per_subject > 0 else None

    # Score each subject
    subject_results: dict[str, dict] = {}
    total_correct = 0
    total_items = 0

    with tempfile.TemporaryDirectory(prefix="mmlu_scoring_") as tmpdir:
        for subj_name, csv_path in sorted(all_subjects.items()):
            print(f"\n[mmlu] === {subj_name} ===")
            items = format_subject_jsonl(csv_path, max_items=max_per)
            if not items:
                print(f"  [skip] No valid items in {csv_path}")
                continue

            # Write JSONL
            jsonl_path = Path(tmpdir) / f"{subj_name}.jsonl"
            with open(jsonl_path, "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")

            print(f"  [mmlu] {len(items)} items")

            # Run scoring
            result = run_scoring(jsonl_path, checkpoint, tokenizer)
            if result is None:
                print(f"  [ERROR] scoring binary failed for {subj_name}")
                subject_results[subj_name] = {
                    "accuracy": 0.0,
                    "correct": 0,
                    "total": len(items),
                    "error": True,
                }
                total_items += len(items)
                continue

            acc = result.get("accuracy", 0.0)
            cor = result.get("correct", 0)
            tot = result.get("total", len(items))
            print(f"  [result] {cor}/{tot} = {acc:.4f}")

            subject_results[subj_name] = {
                "accuracy": acc,
                "correct": cor,
                "total": tot,
            }
            total_correct += cor
            total_items += tot

    # Overall
    overall_acc = total_correct / total_items if total_items > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"MMLU Overall: {total_correct}/{total_items} = {overall_acc:.4f}")
    print(f"{'='*60}")

    # Build full results
    now = datetime.now(timezone.utc).isoformat()
    full_results = {
        "benchmark": "MMLU",
        "timestamp": now,
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "max_per_subject": max_per,
        "overall": {
            "accuracy": overall_acc,
            "correct": total_correct,
            "total": total_items,
        },
        "subjects": subject_results,
    }

    # Save JSON
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"[mmlu] Results saved to {RESULTS_JSON}")

    # Update scorecard
    write_scorecard(full_results)


if __name__ == "__main__":
    main()
