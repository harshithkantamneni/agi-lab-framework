"""L3: RRF fusion of BM25 (sparse) + dense (vector) retrieval.

Reciprocal Rank Fusion is score-free: it ranks each retriever's hits, then
sums 1/(k+rank) across retrievers per document. Robust to score scale
differences between dense (cosine distance) and sparse (BM25). k=60 is the
literature default.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Iterable


def reciprocal_rank_fusion(
    rankings: Iterable[list[Any]],
    k: int = 60,
) -> list[tuple[Any, float]]:
    """Fuse multiple rankings via RRF. Each ranking is a list of doc-IDs in
    descending preference order. Returns list of (doc_id, fused_score) sorted
    by score descending.
    """
    scores: dict[Any, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_search(
    query: str,
    bm25_index_path: Path,
    dense_search_fn: Callable[[str, int], list[dict]],
    top_k: int = 20,
    fetch_per_retriever: int = 50,
    rrf_k: int = 60,
    program: str | None = None,
    role: str | None = None,
    phase: str | None = None,
    bm25_index: object | None = None,
) -> list[dict]:
    """Run both retrievers, fuse via RRF, return top_k merged chunks.

    `bm25_index`, if given, is a pre-loaded BM25Index (used by the persistent
    retrieval server to avoid re-parsing the ~96MB index per query). Otherwise
    the index is loaded from `bm25_index_path`.

    Robustness: a missing OR corrupt/truncated BM25 index (e.g. read while the
    runner is mid-rebuild) degrades to dense-only rather than crashing the
    query path.
    """
    from tools.retrieval.bm25 import BM25Index
    bm25_hits: list[dict] = []
    bm25 = bm25_index
    if bm25 is None and bm25_index_path is not None and Path(bm25_index_path).exists():
        try:
            bm25 = BM25Index.load(bm25_index_path)
        except (ValueError, OSError, KeyError) as e:
            # json.JSONDecodeError subclasses ValueError. Truncated/corrupt index
            # (or an unreadable file): degrade to dense-only, never crash.
            import sys
            sys.stderr.write(f"hybrid_search: BM25 index unusable ({e}); dense-only.\n")
            bm25 = None
    if bm25 is not None:
        try:
            bm25_hits = bm25.search(query, top_k=fetch_per_retriever,
                                    program=program, role=role, phase=phase)
        except Exception as e:  # never let a retriever fault kill the query
            import sys
            sys.stderr.write(f"hybrid_search: BM25 search failed ({e}); dense-only.\n")
            bm25_hits = []
    dense_hits = dense_search_fn(query, fetch_per_retriever)

    bm25_ids = [h["id"] for h in bm25_hits]
    dense_ids = [h["id"] for h in dense_hits]

    fused = reciprocal_rank_fusion([bm25_ids, dense_ids], k=rrf_k)

    by_id_dense = {h["id"]: h for h in dense_hits}
    by_id_bm25 = {h["id"]: h for h in bm25_hits}

    results = []
    for doc_id, rrf_score in fused[:top_k]:
        rec = by_id_dense.get(doc_id) or by_id_bm25.get(doc_id) or {"id": doc_id}
        results.append({
            **rec,
            "rrf_score": rrf_score,
            "from_dense": doc_id in by_id_dense,
            "from_bm25": doc_id in by_id_bm25,
        })
    return results
