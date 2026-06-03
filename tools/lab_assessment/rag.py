from __future__ import annotations
import math
import sqlite3
from pathlib import Path
from tools.lab_assessment.types import DimensionResult
from tools.lab_assessment.sources import Sources


def recall_at_k(ranked_ids: list[list], truth_ids: list, k: int) -> float:
    hits = sum(1 for ranked, t in zip(ranked_ids, truth_ids) if t in ranked[:k])
    return hits / len(truth_ids) if truth_ids else 0.0


def mrr(ranked_ids: list[list], truth_ids: list) -> float:
    total = 0.0
    for ranked, t in zip(ranked_ids, truth_ids):
        if t in ranked:
            total += 1.0 / (ranked.index(t) + 1)
    return total / len(truth_ids) if truth_ids else 0.0


def ndcg_at_k(ranked_ids: list[list], truth_ids: list, k: int) -> float:
    """Binary relevance, exactly one relevant item per query -> IDCG = 1, so nDCG = DCG."""
    total = 0.0
    for ranked, t in zip(ranked_ids, truth_ids):
        for i, rid in enumerate(ranked[:k]):
            if rid == t:
                total += 1.0 / math.log2(i + 2)
                break
    return total / len(truth_ids) if truth_ids else 0.0


def build_known_item_queries(chunks: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Deterministic sample: each query is the first ~12 words of a chunk; truth is its id."""
    import random
    rng = random.Random(seed)
    pool = [c for c in chunks if c.get("chunk_text", "").strip()]
    sample = pool if len(pool) <= n else rng.sample(pool, n)
    out = []
    for c in sample:
        words = c["chunk_text"].split()[:12]
        out.append({"query": " ".join(words), "truth_id": c["id"]})
    return out


_UNAVAILABLE_HEALTH = {
    "chunks": "unavailable",
    "pct_metadata_tagged": "unavailable",
    "program_tagged_pct": "unavailable",
    "role_tagged_pct": "unavailable",
    "phase_tagged_pct": "unavailable",
    "deliverable_tagged_pct": "unavailable",
}


def _index_health(repo: Path, db: Path | None = None) -> dict:
    """Chunk count + metadata-tagged coverage from the live lab_memory.db (if present).

    Reports both the topline pct_metadata_tagged (program OR role OR phase non-empty)
    and per-column coverage so composition is visible (phase is the noisiest column).
    Pass an explicit ``db`` path to point at a non-default database (used by tests);
    otherwise it defaults to ``repo/'tools/lab_memory.db'``.
    """
    if db is None:
        db = repo / "tools/lab_memory.db"
    if not db.exists():
        return dict(_UNAVAILABLE_HEALTH)
    try:
        con = sqlite3.connect(db)
        total = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        tagged = con.execute(
            "SELECT COUNT(*) FROM memories WHERE program_id != '' OR role != '' OR phase != ''"
        ).fetchone()[0]

        def _col_pct(col: str) -> float:
            n = con.execute(
                f"SELECT COUNT(*) FROM memories WHERE {col} != ''"
            ).fetchone()[0]
            return round(100 * n / total, 1) if total else 0.0

        program_pct = _col_pct("program_id")
        role_pct = _col_pct("role")
        phase_pct = _col_pct("phase")
        deliverable_pct = _col_pct("deliverable_type")
        con.close()
        pct = round(100 * tagged / total, 1) if total else 0.0
        return {
            "chunks": total,
            "pct_metadata_tagged": pct,
            "program_tagged_pct": program_pct,
            "role_tagged_pct": role_pct,
            "phase_tagged_pct": phase_pct,
            "deliverable_tagged_pct": deliverable_pct,
        }
    except sqlite3.Error:
        return dict(_UNAVAILABLE_HEALTH)


def compute(src: Sources) -> DimensionResult:
    repo = src.root
    metrics, caveats = {}, []
    metrics.update(_index_health(repo))

    # Retrieval quality is measured against the LIVE index + models. It is heavy
    # (loads the embedder/reranker), so it is OPT-IN via compute_retrieval_quality()
    # called by the CLI on the real repo; the unit tests cover the metric math above.
    metrics["recall_at_5"] = "unavailable"
    metrics["mrr"] = "unavailable"
    caveats.append("Retrieval-quality (recall@k/MRR) computed on-demand on the real index; "
                   "see compute_retrieval_quality(). Known-item eval != hand-labeled relevance.")

    tagged = metrics.get("pct_metadata_tagged")
    breakdown = (f"program {metrics.get('program_tagged_pct')}%/"
                 f"role {metrics.get('role_tagged_pct')}%/"
                 f"phase {metrics.get('phase_tagged_pct')}% tagged")
    if metrics["chunks"] == "unavailable":
        level, rationale = "N/A", "no live retrieval index found"
    elif isinstance(tagged, (int, float)) and tagged < 10:
        level = "Developing"
        rationale = (f"{metrics['chunks']} chunks indexed but only {tagged}% metadata-tagged — "
                     f"scoped retrieval (--program/--role/--phase) is largely unpopulated (P-D620); "
                     f"{breakdown}")
    else:
        level, rationale = "Solid", f"{metrics['chunks']} chunks, {tagged}% tagged; {breakdown}"

    return DimensionResult(
        dimension="rag", metrics=metrics, verdict_level=level,
        verdict_rationale=rationale,
        relative_to="IR norms (recall@k/MRR) + the lab's own metadata-coverage goal",
        caveats=caveats)


def compute_retrieval_quality(repo: Path, n: int = 30, k: int = 5, seed: int = 0) -> dict:
    """Opt-in heavy path: sample known items from lab_memory.db, run search(), measure
    recall@k + MRR + nDCG@k. Loads real models — only call on the real repo, not in unit tests."""
    import sys
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    db = repo / "tools/lab_memory.db"
    if not db.exists():
        return {"recall_at_k": "unavailable", "mrr": "unavailable"}
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    chunks = [dict(r) for r in con.execute(
        "SELECT id, chunk_text FROM memories ORDER BY id LIMIT 5000")]
    con.close()
    queries = build_known_item_queries(chunks, n=n, seed=seed)
    # NOTE: confirm the real search() signature in tools/retrieval/search.py before
    # relying on these kwargs; adapt if the param names differ. Do NOT guess — read it.
    from tools.retrieval.search import search
    ranked, truth = [], []
    for q in queries:
        hits = search(q["query"], repo_root=repo, top_k=k, use_rerank=True)
        ranked.append([h.get("id") for h in hits])
        truth.append(q["truth_id"])
    return {f"recall_at_{k}": round(recall_at_k(ranked, truth, k), 3),
            "mrr": round(mrr(ranked, truth), 3),
            f"ndcg_at_{k}": round(ndcg_at_k(ranked, truth, k), 3),
            "n_queries": len(queries)}
