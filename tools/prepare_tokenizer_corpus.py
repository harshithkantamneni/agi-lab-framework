#!/usr/bin/env python3
"""prepare_tokenizer_corpus.py -- Build a balanced mixed corpus for BPE training.

Samples from text/code/math sources with a target byte ratio (default 50/30/20)
and streams the combined corpus into a single output file.  For large sources
(python_code.txt, openwebtext.txt) it uses random offset sampling of ~1 MB
chunks aligned to line boundaries to avoid loading the whole file into RAM.

Usage:
    python3 tools/prepare_tokenizer_corpus.py \\
        --text data/training/active/wikitext103_train.txt \\
        --text data/training/raw/openwebtext.txt \\
        --code data/training/raw/python_code.txt \\
        --math data/training/raw/gsm8k_train.txt \\
        --math data/training/raw/math_train.txt \\
        --text-mb 200 --code-mb 150 --math-mb 50 \\
        --out data/training/tokenizer_corpus.txt
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import BinaryIO

CHUNK = 1 << 20  # 1 MiB sampling granule


def _align_to_newline(f: BinaryIO, pos: int, file_size: int) -> int:
    """Move `pos` forward to the byte after the next '\\n', bounded by file_size."""
    if pos <= 0:
        return 0
    f.seek(pos)
    # read a small buffer and find the next newline
    buf = f.read(1024)
    idx = buf.find(b"\n")
    if idx == -1:
        return pos  # no newline found nearby, accept misaligned boundary
    return pos + idx + 1


def sample_from_file(
    path: str, target_bytes: int, out: BinaryIO, rng: random.Random
) -> int:
    """Sample ~target_bytes from `path` and write to `out`.

    For small files, copies the whole thing.  For large files, picks random
    1 MiB chunks aligned to line boundaries until the target is reached.
    Returns actual bytes written.
    """
    file_size = os.path.getsize(path)
    written = 0

    with open(path, "rb") as f:
        if file_size <= target_bytes:
            # Small file: take everything
            while True:
                block = f.read(1 << 20)
                if not block:
                    break
                out.write(block)
                written += len(block)
            if written > 0 and not block.endswith(b"\n"):
                out.write(b"\n")
                written += 1
            return written

        # Large file: random chunks
        attempts = 0
        max_attempts = max(1000, (target_bytes // CHUNK) * 4)
        while written < target_bytes and attempts < max_attempts:
            attempts += 1
            if file_size <= CHUNK:
                start = 0
            else:
                start = rng.randint(0, file_size - CHUNK - 1)
            start = _align_to_newline(f, start, file_size)
            f.seek(start)
            block = f.read(CHUNK)
            if not block:
                continue
            # Trim trailing partial line to keep boundaries clean
            nl = block.rfind(b"\n")
            if nl > 0:
                block = block[: nl + 1]
            out.write(block)
            written += len(block)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build balanced mixed corpus for BPE tokenizer training."
    )
    ap.add_argument(
        "--text", action="append", default=[], help="Text corpus file(s)"
    )
    ap.add_argument(
        "--code", action="append", default=[], help="Code corpus file(s)"
    )
    ap.add_argument(
        "--math", action="append", default=[], help="Math corpus file(s)"
    )
    ap.add_argument(
        "--text-mb", type=int, default=200, help="Total MB from text sources"
    )
    ap.add_argument(
        "--code-mb", type=int, default=150, help="Total MB from code sources"
    )
    ap.add_argument(
        "--math-mb", type=int, default=50, help="Total MB from math sources"
    )
    ap.add_argument("--out", required=True, help="Output combined corpus path")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    buckets: list[tuple[str, list[str], int]] = [
        ("text", args.text, args.text_mb * 1024 * 1024),
        ("code", args.code, args.code_mb * 1024 * 1024),
        ("math", args.math, args.math_mb * 1024 * 1024),
    ]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total_written = 0
    report: list[tuple[str, str, int]] = []

    with open(args.out, "wb") as out:
        for name, files, target in buckets:
            if not files or target <= 0:
                continue
            per_file = target // len(files)
            for p in files:
                if not os.path.exists(p):
                    print(f"[Prep] WARN: missing {p}", file=sys.stderr)
                    continue
                w = sample_from_file(p, per_file, out, rng)
                report.append((name, p, w))
                total_written += w
                print(
                    f"[Prep] {name:4s} {p}: wrote {w:,} bytes "
                    f"({w / (1024 * 1024):.1f} MB)"
                )

    print(f"\n[Prep] Combined corpus: {args.out} ({total_written:,} bytes)")
    # Print breakdown
    per_bucket: dict[str, int] = {"text": 0, "code": 0, "math": 0}
    for name, _p, w in report:
        per_bucket[name] += w
    if total_written > 0:
        for name, b in per_bucket.items():
            pct = 100.0 * b / total_written
            print(f"[Prep] {name:4s}: {b:,} bytes ({pct:.1f}%)")


if __name__ == "__main__":
    main()
