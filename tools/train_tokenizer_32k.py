#!/usr/bin/env python3
"""train_tokenizer_32k.py -- Efficient BPE tokenizer trainer for 32K vocabulary.

Trains a byte-level BPE tokenizer on one or more text files, producing a binary
file in the exact format expected by src/core/tokenizer.c (tokenizer_load).

Key differences from the C trainer (train_tokenizer.c):
  - Uses word-level pre-tokenization (regex split) + word frequency counting
    so pair counting is O(sum of unique word lengths * vocab), not O(corpus * vocab).
  - Maintains a reverse index (pair -> word indices) for O(affected words) per merge.
  - Can train 32K vocab on 500MB+ text in ~30-60 minutes instead of days.

Usage:
    python3 tools/train_tokenizer_32k.py \\
        --input data/training/active/wikitext103_train.txt \\
        --output data/training/tokenizer_32k.bin \\
        --vocab-size 32768

Binary format (must match tokenizer.c):
    Header (28 bytes):
        magic:      "BPE\\0"   (4 bytes)
        vocab_size: int32      (4 bytes)
        n_merges:   int32      (4 bytes)
        pad_id:     int32      (4 bytes)
        unk_id:     int32      (4 bytes)
        bos_id:     int32      (4 bytes)
        eos_id:     int32      (4 bytes)
    Vocab (variable):
        For each entry: length (int32) + bytes (length bytes)
    Merges (n_merges * 12 bytes):
        For each: left (int32) + right (int32) + merged_id (int32)
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
import time
from collections import Counter, defaultdict
from typing import BinaryIO


# ---------------------------------------------------------------------------
# Pre-tokenization: split text into "words" (whitespace-delimited chunks).
# Each word keeps its leading space so the tokenizer is whitespace-aware.
# We use a GPT-2-style regex that splits on word boundaries.
# ---------------------------------------------------------------------------

# GPT-2 style pattern -- splits on contractions, numbers, whitespace+word, etc.
_GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def pre_tokenize(text: str) -> list[tuple[tuple[int, ...], int]]:
    """Split text into (byte_tuple, frequency) pairs using word-level counts.

    Returns a list of (word_as_byte_ids, count) where each word is a tuple of
    byte values (0-255).  Counting at the word level means we only track
    unique words, not every position in the corpus.
    """
    word_counts: Counter[tuple[int, ...]] = Counter()
    for match in _GPT2_PAT.finditer(text):
        word = match.group()
        word_bytes = tuple(word.encode("utf-8"))
        word_counts[word_bytes] += 1
    return list(word_counts.items())


# ---------------------------------------------------------------------------
# BPE Training with reverse index for efficient updates
# ---------------------------------------------------------------------------


def merge_word(
    word: tuple[int, ...], pair: tuple[int, int], new_id: int
) -> tuple[int, ...]:
    """Replace all occurrences of `pair` in `word` with `new_id`."""
    if len(word) < 2:
        return word
    new_word: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def train_bpe(
    text: str,
    max_vocab_size: int = 32768,
    min_frequency: int = 2,
    progress_interval: int = 1000,
) -> tuple[list[bytes], list[tuple[int, int, int]]]:
    """Train byte-level BPE on `text`, returning (vocab, merges).

    Uses a reverse index (pair -> set of word indices) so each merge step
    only processes words that actually contain the merged pair, not all words.

    vocab: list of byte-strings indexed by token ID.
           IDs 0-255 are single bytes. IDs 256+ are merged tokens.
    merges: list of (left_id, right_id, merged_id) triples.
    """
    max_merges = max_vocab_size - 256
    if max_merges <= 0:
        raise ValueError(f"vocab_size must be > 256, got {max_vocab_size}")

    print(f"[BPE] Pre-tokenizing text ({len(text):,} chars)...")
    t0 = time.time()
    words = pre_tokenize(text)
    print(f"[BPE] {len(words):,} unique words in {time.time() - t0:.1f}s")

    # Build initial vocab: 256 byte tokens
    vocab: list[bytes] = [bytes([i]) for i in range(256)]

    # Data structures:
    #   word_data[i] = (word_tuple, freq)   -- mutable list of words
    #   pair_counts: Counter of (left, right) -> total weighted count
    #   pair_to_words: dict of (left, right) -> set of word indices containing that pair

    word_data: list[list[int]] = [list(w) for w, _ in words]
    word_freq: list[int] = [f for _, f in words]

    print(f"[BPE] Building initial pair counts and reverse index...")
    t0 = time.time()

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)

    for wi in range(len(word_data)):
        w = word_data[wi]
        freq = word_freq[wi]
        for j in range(len(w) - 1):
            p = (w[j], w[j + 1])
            pair_counts[p] += freq
            pair_to_words[p].add(wi)

    print(
        f"[BPE] {len(pair_counts):,} unique pairs, "
        f"index built in {time.time() - t0:.1f}s"
    )

    merges: list[tuple[int, int, int]] = []
    t_start = time.time()

    for merge_idx in range(max_merges):
        if not pair_counts:
            print(f"[BPE] No more pairs to merge at step {merge_idx}")
            break

        # Find most frequent pair
        best_pair = max(pair_counts, key=lambda p: pair_counts[p])
        best_count = pair_counts[best_pair]

        if best_count < min_frequency:
            print(
                f"[BPE] Stopping: best pair count {best_count} < min_frequency {min_frequency} "
                f"at merge {merge_idx}"
            )
            break

        # Create new token
        new_id = 256 + merge_idx
        left_id, right_id = best_pair

        # Build the byte representation of the new token
        left_bytes = vocab[left_id]
        right_bytes = vocab[right_id]
        new_bytes = left_bytes + right_bytes
        vocab.append(new_bytes)

        merges.append((left_id, right_id, new_id))

        # Progress reporting
        if (merge_idx + 1) % progress_interval == 0 or merge_idx == 0:
            elapsed = time.time() - t_start
            rate = (merge_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (max_merges - merge_idx - 1) / rate if rate > 0 else 0
            n_affected = len(pair_to_words.get(best_pair, set()))
            print(
                f"[BPE] Merge {merge_idx + 1}/{max_merges}: "
                f"({left_id}, {right_id}) -> {new_id} "
                f"[count={best_count:,}, affected={n_affected:,}, "
                f"vocab={len(vocab)}, "
                f"{rate:.1f} merges/s, ETA {eta / 60:.1f}m]"
            )

        # --- Incremental pair count + reverse index update ---
        # Only process words that contain the merged pair.
        affected_words = pair_to_words.pop(best_pair, set())
        del pair_counts[best_pair]

        for wi in affected_words:
            w = word_data[wi]
            freq = word_freq[wi]

            if len(w) < 2:
                continue

            # Remove old pair counts and reverse index entries for this word
            for j in range(len(w) - 1):
                p = (w[j], w[j + 1])
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                s = pair_to_words.get(p)
                if s is not None:
                    s.discard(wi)
                    if not s:
                        pair_to_words.pop(p, None)

            # Merge the word in-place
            new_w: list[int] = []
            i = 0
            while i < len(w):
                if (
                    i < len(w) - 1
                    and w[i] == left_id
                    and w[i + 1] == right_id
                ):
                    new_w.append(new_id)
                    i += 2
                else:
                    new_w.append(w[i])
                    i += 1
            word_data[wi] = new_w

            # Add new pair counts and reverse index entries
            for j in range(len(new_w) - 1):
                p = (new_w[j], new_w[j + 1])
                pair_counts[p] = pair_counts.get(p, 0) + freq
                if p not in pair_to_words:
                    pair_to_words[p] = set()
                pair_to_words[p].add(wi)

    # Clean up zero-count entries
    pair_counts = {k: v for k, v in pair_counts.items() if v > 0}

    elapsed_total = time.time() - t_start
    print(
        f"[BPE] Training complete: {len(merges)} merges, "
        f"vocab size {len(vocab)}, {elapsed_total:.1f}s total"
    )
    return vocab, merges


# ---------------------------------------------------------------------------
# Binary format save (matching tokenizer.c exactly)
# ---------------------------------------------------------------------------


def save_tokenizer_bin(
    path: str,
    vocab: list[bytes],
    merges: list[tuple[int, int, int]],
    pad_id: int = -1,
    unk_id: int = -1,
    bos_id: int = -1,
    eos_id: int = -1,
) -> None:
    """Save tokenizer in the binary format expected by tokenizer_load() in C."""
    vocab_size = len(vocab)
    n_merges = len(merges)

    with open(path, "wb") as f:
        # Header: magic + vocab_size + n_merges + 4 special tokens
        f.write(b"BPE\x00")
        f.write(struct.pack("<i", vocab_size))
        f.write(struct.pack("<i", n_merges))
        f.write(struct.pack("<i", pad_id))
        f.write(struct.pack("<i", unk_id))
        f.write(struct.pack("<i", bos_id))
        f.write(struct.pack("<i", eos_id))

        # Vocab entries: length (int32) + bytes
        for entry in vocab:
            f.write(struct.pack("<i", len(entry)))
            f.write(entry)

        # Merge rules: left (int32) + right (int32) + merged_id (int32)
        for left, right, merged in merges:
            f.write(struct.pack("<iii", left, right, merged))

    file_size = os.path.getsize(path)
    print(f"[Save] Written {path} ({file_size:,} bytes)")
    print(f"[Save] vocab_size={vocab_size}, n_merges={n_merges}")


# ---------------------------------------------------------------------------
# Verification: load back and test encode/decode
# ---------------------------------------------------------------------------


def load_tokenizer_bin(
    path: str,
) -> tuple[list[bytes], list[tuple[int, int, int]], dict[str, int]]:
    """Load tokenizer from binary file (for verification)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"BPE\x00", f"Bad magic: {magic!r}"

        vocab_size = struct.unpack("<i", f.read(4))[0]
        n_merges = struct.unpack("<i", f.read(4))[0]
        pad_id = struct.unpack("<i", f.read(4))[0]
        unk_id = struct.unpack("<i", f.read(4))[0]
        bos_id = struct.unpack("<i", f.read(4))[0]
        eos_id = struct.unpack("<i", f.read(4))[0]

        vocab: list[bytes] = []
        for _ in range(vocab_size):
            length = struct.unpack("<i", f.read(4))[0]
            data = f.read(length) if length > 0 else b""
            vocab.append(data)

        merges: list[tuple[int, int, int]] = []
        for _ in range(n_merges):
            left, right, merged = struct.unpack("<iii", f.read(12))
            merges.append((left, right, merged))

    specials = {
        "pad_id": pad_id,
        "unk_id": unk_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
    }
    return vocab, merges, specials


def encode(
    text: str, vocab: list[bytes], merges: list[tuple[int, int, int]]
) -> list[int]:
    """Encode text to token IDs using the BPE merge rules."""
    ids = list(text.encode("utf-8"))
    for left, right, merged in merges:
        new_ids: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == left and ids[i + 1] == right:
                new_ids.append(merged)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids
    return ids


def decode(ids: list[int], vocab: list[bytes]) -> str:
    """Decode token IDs back to text."""
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")


def verify_tokenizer(path: str, test_strings: list[str] | None = None) -> bool:
    """Load tokenizer and verify encode-decode round-trip."""
    print(f"\n[Verify] Loading {path}...")
    vocab, merges, specials = load_tokenizer_bin(path)
    print(f"[Verify] vocab_size={len(vocab)}, n_merges={len(merges)}")
    print(f"[Verify] specials={specials}")

    if test_strings is None:
        test_strings = [
            "The quick brown fox jumps over the lazy dog.",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "Solve for x: 3x^2 + 5x - 2 = 0",
            "Hello, world!",
            "Machine learning and artificial intelligence are transforming the world.",
            "The integral of x^2 dx = x^3/3 + C",
            "import numpy as np\nresult = np.array([1, 2, 3])",
        ]

    all_ok = True
    for s in test_strings:
        ids = encode(s, vocab, merges)
        decoded = decode(ids, vocab)
        ok = decoded == s
        status = "OK" if ok else "FAIL"
        compression = len(s.encode("utf-8")) / len(ids) if ids else 0
        print(
            f"  [{status}] \"{s[:50]}{'...' if len(s) > 50 else ''}\" "
            f"-> {len(ids)} tokens ({compression:.2f} bytes/tok)"
        )
        if not ok:
            print(f"         decoded: \"{decoded[:80]}\"")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Data preparation: extract text from various formats
# ---------------------------------------------------------------------------


def load_text_files(paths: list[str], max_bytes: int = 0) -> str:
    """Load and concatenate text from multiple files.

    Supports:
      - .txt: raw text
      - .jsonl: extracts all string values (question, answer, text fields)
    """
    chunks: list[str] = []
    total_bytes = 0

    for path in paths:
        if not os.path.exists(path):
            print(f"[Data] WARNING: {path} not found, skipping")
            continue

        file_size = os.path.getsize(path)
        print(f"[Data] Loading {path} ({file_size:,} bytes)...")

        if path.endswith(".jsonl"):
            import json

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        for key in (
                            "text",
                            "question",
                            "answer",
                            "problem",
                            "solution",
                            "content",
                        ):
                            if key in obj and isinstance(obj[key], str):
                                chunks.append(obj[key])
                                total_bytes += len(obj[key].encode("utf-8"))
                    except json.JSONDecodeError:
                        continue
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
                chunks.append(data)
                total_bytes += len(data.encode("utf-8"))

        if max_bytes > 0 and total_bytes >= max_bytes:
            print(f"[Data] Reached max_bytes={max_bytes:,}, stopping")
            break

    combined = "\n".join(chunks)
    print(f"[Data] Total: {len(combined):,} chars from {len(paths)} file(s)")
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 32K BPE tokenizer for the AGI project."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input text file(s) for training (.txt or .jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/training/tokenizer_32k.bin",
        help="Output tokenizer binary path",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32768,
        help="Target vocabulary size (default: 32768)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum pair frequency to merge (default: 2)",
    )
    parser.add_argument(
        "--max-text-mb",
        type=int,
        default=0,
        help="Max MB of text to use (0 = unlimited).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing tokenizer file",
    )
    parser.add_argument(
        "--sample-mb",
        type=int,
        default=0,
        help="Sample this many MB from the loaded text (0 = use all). "
        "Reduces training time for very large corpora.",
    )

    args = parser.parse_args()

    if args.verify_only:
        ok = verify_tokenizer(args.output)
        sys.exit(0 if ok else 1)

    # Load training text
    max_bytes = args.max_text_mb * 1024 * 1024 if args.max_text_mb > 0 else 0
    text = load_text_files(args.input, max_bytes=max_bytes)

    if not text:
        print("ERROR: No text loaded. Check input paths.", file=sys.stderr)
        sys.exit(1)

    # Optional sampling for very large corpora
    if args.sample_mb > 0:
        sample_chars = args.sample_mb * 1024 * 1024
        if len(text) > sample_chars:
            print(f"[Data] Sampling {args.sample_mb}MB from {len(text):,} chars...")
            text = text[:sample_chars]

    # Train BPE
    vocab, merges = train_bpe(
        text,
        max_vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    # Save in C-compatible binary format
    save_tokenizer_bin(args.output, vocab, merges)

    # Verify
    ok = verify_tokenizer(args.output)
    if not ok:
        print("\nWARNING: Verification failed!", file=sys.stderr)
        sys.exit(1)

    print("\nSUCCESS: Tokenizer trained and verified.")


if __name__ == "__main__":
    main()
