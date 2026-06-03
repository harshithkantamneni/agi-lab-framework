#!/usr/bin/env python3
"""tokenizer_compare.py -- Compare compression ratios of two BPE tokenizers.

Loads two .bin tokenizers (produced by tokenizer_save / train_tokenizer_32k.py)
and measures bytes/token on a set of corpora.  Uses the same encode algorithm
as the C implementation: apply merges in priority order over the byte sequence.

Usage:
    python3 tools/tokenizer_compare.py \\
        --a data/training/tokenizer_4k.bin \\
        --b data/training/tokenizer_32k.bin \\
        --sample-bytes 2000000 \\
        --corpus wikitext103 data/training/active/wikitext103_train.txt \\
        --corpus python_code data/training/raw/python_code.txt \\
        --corpus openwebtext data/training/raw/openwebtext.txt \\
        --corpus gsm8k data/training/raw/gsm8k_train.txt \\
        --corpus math data/training/raw/math_train.txt
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time


def load_tokenizer(path: str) -> tuple[list[bytes], list[tuple[int, int, int]]]:
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"BPE\x00", f"Bad magic in {path}: {magic!r}"
        vocab_size = struct.unpack("<i", f.read(4))[0]
        n_merges = struct.unpack("<i", f.read(4))[0]
        # skip specials
        _ = f.read(16)
        vocab: list[bytes] = []
        for _ in range(vocab_size):
            ln = struct.unpack("<i", f.read(4))[0]
            vocab.append(f.read(ln) if ln > 0 else b"")
        merges: list[tuple[int, int, int]] = []
        for _ in range(n_merges):
            left, right, merged = struct.unpack("<iii", f.read(12))
            merges.append((left, right, merged))
    return vocab, merges


def encode_bytes(
    data: bytes, merges: list[tuple[int, int, int]]
) -> list[int]:
    """Replicates tokenizer_encode(): apply merges in priority order on IDs."""
    ids = list(data)
    for left, right, merged in merges:
        out: list[int] = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == left and ids[i + 1] == right:
                out.append(merged)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids


def sample_corpus(path: str, sample_bytes: int) -> bytes:
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        if file_size <= sample_bytes:
            return f.read()
        # Take from middle for representativeness; align to newline
        start = file_size // 2 - sample_bytes // 2
        if start < 0:
            start = 0
        f.seek(start)
        # align forward to newline
        aligned = f.read(1024)
        nl = aligned.find(b"\n")
        if nl >= 0:
            f.seek(start + nl + 1)
        return f.read(sample_bytes)


def bench(
    name: str,
    data: bytes,
    vocab_a: list[bytes],
    merges_a: list[tuple[int, int, int]],
    vocab_b: list[bytes],
    merges_b: list[tuple[int, int, int]],
) -> dict[str, float]:
    t0 = time.time()
    ids_a = encode_bytes(data, merges_a)
    ta = time.time() - t0
    t0 = time.time()
    ids_b = encode_bytes(data, merges_b)
    tb = time.time() - t0

    nb = len(data)
    r_a = nb / len(ids_a) if ids_a else 0.0
    r_b = nb / len(ids_b) if ids_b else 0.0
    # round-trip check
    back_a = b"".join(vocab_a[i] for i in ids_a)
    back_b = b"".join(vocab_b[i] for i in ids_b)
    ok_a = back_a == data
    ok_b = back_b == data
    return {
        "name": name,
        "bytes": nb,
        "tokens_a": len(ids_a),
        "tokens_b": len(ids_b),
        "bpt_a": r_a,
        "bpt_b": r_b,
        "improve_pct": (r_b / r_a - 1.0) * 100.0 if r_a > 0 else 0.0,
        "time_a_s": ta,
        "time_b_s": tb,
        "roundtrip_a": ok_a,
        "roundtrip_b": ok_b,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Tokenizer A (smaller vocab)")
    ap.add_argument("--b", required=True, help="Tokenizer B (larger vocab)")
    ap.add_argument(
        "--corpus",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        default=[],
        help="Corpus to benchmark (repeatable): NAME PATH",
    )
    ap.add_argument(
        "--sample-bytes",
        type=int,
        default=2_000_000,
        help="Bytes to sample from each corpus (default 2 MB)",
    )
    args = ap.parse_args()

    print(f"Loading A: {args.a}")
    vocab_a, merges_a = load_tokenizer(args.a)
    print(f"  vocab={len(vocab_a)} merges={len(merges_a)}")
    print(f"Loading B: {args.b}")
    vocab_b, merges_b = load_tokenizer(args.b)
    print(f"  vocab={len(vocab_b)} merges={len(merges_b)}")

    print(
        f"\n{'Corpus':<14} {'bytes':>10} {'tokA':>9} {'tokB':>9} "
        f"{'B/tokA':>7} {'B/tokB':>7} {'delta%':>8} {'rtA':>4} {'rtB':>4}"
    )
    print("-" * 86)
    for name, path in args.corpus:
        if not os.path.exists(path):
            print(f"{name:<14} MISSING: {path}")
            continue
        data = sample_corpus(path, args.sample_bytes)
        r = bench(name, data, vocab_a, merges_a, vocab_b, merges_b)
        print(
            f"{r['name']:<14} {r['bytes']:>10,} {r['tokens_a']:>9,} "
            f"{r['tokens_b']:>9,} {r['bpt_a']:>7.3f} {r['bpt_b']:>7.3f} "
            f"{r['improve_pct']:>+7.1f}% "
            f"{'OK' if r['roundtrip_a'] else 'FAIL':>4} "
            f"{'OK' if r['roundtrip_b'] else 'FAIL':>4}"
        )


if __name__ == "__main__":
    main()
