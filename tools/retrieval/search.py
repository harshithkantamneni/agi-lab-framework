"""Top-level orchestrator for the 4-layer retrieval system.

Pipeline:
    1. Extract canonical seed tokens from query (D-N, P-*)
    2. Run PPR on the token graph; produce related-node set
    3. Run BM25 search and dense search in parallel
    4. RRF fusion
    5. Cross-encoder rerank top-30 to top-K
    6. Return ranked chunks with full metadata
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

from tools.retrieval.graph import (
    TokenGraph, personalized_pagerank,
    extract_decision_tokens, extract_carry_forward_tokens,
)
from tools.retrieval.bm25 import BM25Index
from tools.retrieval.hybrid import hybrid_search
from tools.retrieval.rerank import CrossEncoderReranker


def _extract_seed_tokens(query: str) -> set[str]:
    """Pull canonical tokens from the query for graph PPR seeding."""
    decisions = extract_decision_tokens(query)
    carry_forwards = extract_carry_forward_tokens(query)
    return decisions | carry_forwards


def search(
    query: str,
    repo_root: Path,
    top_k: int = 5,
    fetch_per_retriever: int = 50,
    rerank_pool: int = 30,
    use_graph: bool = True,
    use_rerank: bool = True,
    program: str | None = None,
    role: str | None = None,
    phase: str | None = None,
    reranker: "CrossEncoderReranker | None" = None,
    lab_memory: object | None = None,
    bm25_index: object | None = None,
) -> list[dict]:
    """Run the full 4-layer retrieval pipeline.

    The `reranker`, `lab_memory`, and `bm25_index` singletons may be injected by
    the persistent retrieval server (tools/retrieval/server.py) so the heavy
    models + parsed index are loaded ONCE and reused across queries. When left
    None they are constructed locally, preserving the original in-process
    behavior (used by the CLI fallback path and tests).
    """
    # ---- L0: Token graph seed expansion ----
    graph_related: set[str] = set()
    if use_graph:
        seeds = _extract_seed_tokens(query)
        if seeds:
            graph_db = repo_root / "tools/lab_graph.db"
            if graph_db.exists():
                try:
                    g = TokenGraph(graph_db)
                    seed_weights = {s: 1.0 / len(seeds) for s in seeds}
                    ranking = personalized_pagerank(g, seed_weights, top_k=20)
                    graph_related = {node_id for node_id, _ in ranking}
                except Exception as e:  # graph is an enhancement, never fatal
                    import sys as _sys
                    _sys.stderr.write(f"search: graph layer skipped ({e}).\n")

    # ---- L1 + L2: dense + BM25 ----
    bm25_path = repo_root / "tools/lab_bm25.json"
    lab_memory_db = repo_root / "tools/lab_memory.db"

    # Build (or reuse) the dense embedder once — not per dense_search call.
    lm = lab_memory
    if lm is None:
        import sys as _sys
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from tools.lab_memory import LabMemory
        lm = LabMemory(str(lab_memory_db))

    def dense_search(q: str, k: int) -> list[dict]:
        try:
            hits = lm.search(q, program_id=program, role=role, phase=phase, top_k=k)
        except Exception as e:  # dense fault must not kill the query — RRF uses bm25 only
            import sys as _sys
            _sys.stderr.write(f"search: dense layer failed ({e}); bm25-only.\n")
            return []
        return [
            {"id": h.id, "program_id": h.program_id, "phase": h.phase, "role": h.role,
             "source_path": h.source_path, "chunk_text": h.chunk_text, "distance": h.distance}
            for h in hits
        ]

    fused = hybrid_search(
        query=query,
        bm25_index_path=bm25_path,
        dense_search_fn=dense_search,
        top_k=rerank_pool if use_rerank else top_k,
        fetch_per_retriever=fetch_per_retriever,
        program=program, role=role, phase=phase,
        bm25_index=bm25_index,
    )

    # Graph boost annotation (does not re-rank — keeps RRF authoritative)
    if graph_related:
        for chunk in fused:
            file_node = f"file:{chunk.get('source_path', '')}"
            chunk["graph_boost"] = file_node in graph_related

    # ---- L4: rerank ----
    if use_rerank:
        rr = reranker if reranker is not None else CrossEncoderReranker()
        try:
            final = rr.rerank(query, fused, top_k=top_k)
        except Exception as e:  # rerank fault degrades to RRF order, never fatal
            import sys as _sys
            _sys.stderr.write(f"search: rerank failed ({e}); RRF order.\n")
            final = fused[:top_k]
    else:
        final = fused[:top_k]

    return final


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="4-layer hybrid retrieval CLI.")
    parser.add_argument("query")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--program", default=None)
    parser.add_argument("--role", default=None)
    parser.add_argument("--phase", default=None)
    args = parser.parse_args(argv)

    results = search(
        query=args.query,
        repo_root=Path(args.repo),
        top_k=args.top_k,
        use_graph=not args.no_graph,
        use_rerank=not args.no_rerank,
        program=args.program,
        role=args.role,
        phase=args.phase,
    )

    for r in results:
        score = r.get("rerank_score") or r.get("rrf_score") or 0.0
        boost = " +graph" if r.get("graph_boost") else ""
        sources = ("D" if r.get("from_dense") else " ") + ("B" if r.get("from_bm25") else " ")
        snip = r.get("chunk_text", "")[:160].replace("\n", " ")
        print(f"[{score:.3f}][{sources}{boost}] {r.get('source_path', '?')} — {snip}{'...' if len(r.get('chunk_text', '')) > 160 else ''}")
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
