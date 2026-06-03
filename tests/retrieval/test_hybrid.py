def test_rrf_rewards_consensus():
    """A doc retrieved by BOTH retrievers must beat a doc retrieved by only one
    at a comparable rank — that is the entire point of fusion (was untested).
    (Note: by convexity of 1/(k+r), RRF does NOT reward equal-middling ranks over
    spread ranks; the real consensus property is membership in more lists.)"""
    from tools.retrieval.hybrid import reciprocal_rank_fusion
    # 'a' is in both lists (rank 1 & 1); 'b' only in list1, 'c' only in list2.
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], k=60)
    score = dict(fused)
    assert score["a"] > score["b"], f"consensus doc a should win: {fused}"
    assert score["a"] > score["c"], f"consensus doc a should win: {fused}"


def test_hybrid_degrades_to_dense_when_bm25_corrupt(tmp_path):
    """A truncated/half-written BM25 index (e.g. read during a runner rebuild)
    must NOT crash the query path — hybrid_search degrades to dense-only."""
    from tools.retrieval.hybrid import hybrid_search
    corrupt = tmp_path / "bm25.json"
    corrupt.write_text('{"chunks": [{"id": 1, "source_path": "a.md", "chunk_text": "alp')  # truncated

    def stub_dense_search(query, top_k):
        return [{"id": 7, "source_path": "z.md", "chunk_text": "dense only", "distance": 0.2}]

    results = hybrid_search(
        query="anything",
        bm25_index_path=corrupt,
        dense_search_fn=stub_dense_search,
        top_k=5,
    )
    assert [r["id"] for r in results] == [7], "should degrade to dense-only, not crash"


def test_rrf_fuses_two_rankings():
    from tools.retrieval.hybrid import reciprocal_rank_fusion
    dense_ranking = ["doc1", "doc2", "doc3", "doc4"]
    bm25_ranking = ["doc4", "doc3", "doc1", "doc2"]
    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=60)
    ids = [d for d, _ in fused]
    assert set(ids) == {"doc1", "doc2", "doc3", "doc4"}
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_handles_disjoint_rankings():
    from tools.retrieval.hybrid import reciprocal_rank_fusion
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "d"]], k=60)
    ids = [d for d, _ in fused]
    assert set(ids) == {"a", "b", "c", "d"}


def test_hybrid_search_returns_unioned_hits(tmp_path):
    from tools.retrieval.hybrid import hybrid_search
    from tools.retrieval.bm25 import BM25Index
    chunks = [
        {"id": 1, "source_path": "a.md", "chunk_text": "D-420 closed Program 2"},
        {"id": 2, "source_path": "b.md", "chunk_text": "router entropy collapse"},
        {"id": 3, "source_path": "c.md", "chunk_text": "dense MoE comparison"},
    ]
    bm25 = BM25Index.from_chunks(chunks)
    bm25_path = tmp_path / "bm25.json"
    bm25.save(bm25_path)

    def stub_dense_search(query, top_k):
        return [
            {"id": 2, "source_path": "b.md", "chunk_text": "router entropy collapse", "distance": 0.1},
            {"id": 1, "source_path": "a.md", "chunk_text": "D-420 closed Program 2", "distance": 0.3},
        ]

    results = hybrid_search(
        query="D-420",
        bm25_index_path=bm25_path,
        dense_search_fn=stub_dense_search,
        top_k=5,
    )
    assert len(results) > 0
    top_ids = [r["id"] for r in results[:2]]
    assert 1 in top_ids
