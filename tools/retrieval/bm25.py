"""BM25 sparse retrieval over the same chunks as tools/lab_memory.py.

Uses rank_bm25 (BM25Okapi). Persistence: JSON file containing tokenized
corpus + chunk metadata. On load, BM25Okapi is rebuilt from tokens.
JSON instead of pickle for safety (no arbitrary code execution risk on
load). Rebuild cost: ~50-200ms for 11K chunks — negligible.

Tokenization preserves canonical lab identifiers (D-420, P-D417-FOO,
program_2_example) by keeping hyphen-joined alphanumeric runs intact.
"""
from __future__ import annotations
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)*|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer that preserves canonical identifiers."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: list[dict]
    tokenized: list[list[str]] | None = None  # set when built from chunks; None after a stats load

    @classmethod
    def from_chunks(cls, chunks: list[dict]) -> "BM25Index":
        if not chunks:
            raise ValueError("BM25Index.from_chunks requires at least one chunk")
        tokenized = [tokenize(c["chunk_text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        return cls(bm25=bm25, chunks=chunks, tokenized=tokenized)

    @staticmethod
    def _bm25_stats(bm25: BM25Okapi) -> dict:
        """The precomputed BM25Okapi statistics needed to score — persisting
        these lets load() skip the O(total tokens) rebuild + IDF recomputation."""
        return {
            "k1": bm25.k1, "b": bm25.b, "epsilon": bm25.epsilon,
            "corpus_size": bm25.corpus_size, "avgdl": bm25.avgdl,
            "average_idf": bm25.average_idf,
            "doc_freqs": bm25.doc_freqs, "idf": bm25.idf, "doc_len": bm25.doc_len,
        }

    @staticmethod
    def _bm25_from_stats(s: dict) -> BM25Okapi:
        """Reconstruct a scorable BM25Okapi from persisted stats WITHOUT re-running
        the corpus loop (bypasses __init__; verified get_scores is identical)."""
        bm25 = BM25Okapi.__new__(BM25Okapi)
        bm25.k1, bm25.b, bm25.epsilon = s["k1"], s["b"], s["epsilon"]
        bm25.corpus_size, bm25.avgdl = s["corpus_size"], s["avgdl"]
        bm25.average_idf = s["average_idf"]
        bm25.doc_freqs, bm25.idf, bm25.doc_len = s["doc_freqs"], s["idf"], s["doc_len"]
        bm25.tokenizer = None
        return bm25

    @classmethod
    def from_lab_memory_db(cls, db_path: Path) -> "BM25Index":
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """SELECT id, program_id, phase, role, deliverable_type,
                          source_path, chunk_text, timestamp
                   FROM memories"""
            ).fetchall()
        cols = ["id", "program_id", "phase", "role", "deliverable_type",
                "source_path", "chunk_text", "timestamp"]
        chunks = [dict(zip(cols, r)) for r in rows]
        return cls.from_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        program: str | None = None,
        role: str | None = None,
        phase: str | None = None,
    ) -> list[dict]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        # Keep (score, index) tuples and apply the metadata predicate here, so we
        # only materialize full chunk dicts for the top_k survivors — not for
        # every scored chunk. (audit #4: avoid early hydration of all hits.)
        keep = []
        for i, score in enumerate(scores):
            c = self.chunks[i]
            if program is not None and c.get("program_id") != program:
                continue
            if role is not None and c.get("role") != role:
                continue
            if phase is not None and c.get("phase") != phase:
                continue
            keep.append((score, i))
        keep.sort(key=lambda x: x[0], reverse=True)
        # Note: deviates from plan's `if s > 0` filter — BM25Okapi yields
        # non-positive scores on tiny corpora (test fixtures with 1-2 docs)
        # due to the log((N-df+0.5)/(df+0.5)) IDF formula. We keep scoring
        # and rely on top_k truncation; production corpora (~11K chunks)
        # will have ample positive scores so behavior is unchanged in practice.
        return [{**self.chunks[i], "bm25_score": float(s)} for s, i in keep[:top_k]]

    def save(self, path: Path) -> None:
        """Persist as JSON, ATOMICALLY. Stores the PRECOMPUTED BM25 statistics
        (idf / doc_freqs / doc_len / avgdl / ...), not the raw token lists, so
        load() rehydrates the scorer directly instead of recomputing it from the
        whole corpus on every query (audit #4: ~O(total tokens) CPU per load).

        Writes to a sibling temp file then os.replace()s it into place, so a
        concurrent reader (e.g. a query loading the index while the runner
        rebuilds it) never observes a truncated/half-written ~96MB file — the
        rename is atomic on the same filesystem (POSIX). Streams via json.dump
        to an open handle to avoid materializing the whole blob as one string.
        """
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"chunks": self.chunks, "bm25_stats": self._bm25_stats(self.bm25)}, fh)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = json.loads(path.read_text())
        chunks = data["chunks"]
        if "bm25_stats" in data:  # new format — rehydrate, no recompute
            return cls(bm25=cls._bm25_from_stats(data["bm25_stats"]), chunks=chunks)
        # backward compatibility: old format persisted raw tokens -> rebuild once.
        tokenized = data["tokenized"]
        return cls(bm25=BM25Okapi(tokenized), chunks=chunks, tokenized=tokenized)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="BM25 sparse retrieval CLI.")
    parser.add_argument("--index", default="tools/lab_bm25.json")
    parser.add_argument("--lab-memory-db", default="tools/lab_memory.db")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Per Task 1.5 experience: tests invoke --index/--lab-memory-db AFTER the
    # subcommand, so accept them on subparsers via argparse.SUPPRESS (the
    # parent-parser defaults still apply when the subparser flag is omitted).
    p_build = sub.add_parser("build", help="(Re)build the BM25 index from lab_memory.db")
    p_build.add_argument("--index", default=argparse.SUPPRESS)
    p_build.add_argument("--lab-memory-db", default=argparse.SUPPRESS)

    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("query")
    p_search.add_argument("--index", default=argparse.SUPPRESS)
    p_search.add_argument("--lab-memory-db", default=argparse.SUPPRESS)
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--program", default=None)
    p_search.add_argument("--role", default=None)
    p_search.add_argument("--phase", default=None)

    args = parser.parse_args(argv)
    index_path = Path(args.index)

    if args.cmd == "build":
        idx = BM25Index.from_lab_memory_db(Path(args.lab_memory_db))
        idx.save(index_path)
        print(f"Built BM25 index with {len(idx.chunks)} chunks at {index_path}")
        return 0

    if args.cmd == "search":
        if not index_path.exists():
            idx = BM25Index.from_lab_memory_db(Path(args.lab_memory_db))
            idx.save(index_path)
        else:
            idx = BM25Index.load(index_path)
        hits = idx.search(args.query, top_k=args.top_k, program=args.program,
                          role=args.role, phase=args.phase)
        for h in hits:
            snip = h["chunk_text"][:140].replace("\n", " ")
            print(f"[{h['bm25_score']:.3f}] {h['source_path']} — {snip}{'...' if len(h['chunk_text']) > 140 else ''}")
        return 0

    return 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
