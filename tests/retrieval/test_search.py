def test_search_returns_ranked_chunks_end_to_end():
    from tools.retrieval.search import search
    from pathlib import Path
    REPO = Path(__file__).resolve().parent.parent.parent

    if not (REPO / "tools/lab_memory.db").exists():
        import pytest
        pytest.skip("lab_memory.db missing; integration test skipped")

    results = search(
        query="Program 2 D-421 closure_memo",
        repo_root=REPO,
        top_k=5,
        use_graph=True,
        use_rerank=True,
    )
    assert isinstance(results, list)
    if results:
        assert "chunk_text" in results[0]
        assert "rerank_score" in results[0] or "rrf_score" in results[0]


def test_search_works_without_rerank():
    from tools.retrieval.search import search
    from pathlib import Path
    REPO = Path(__file__).resolve().parent.parent.parent
    if not (REPO / "tools/lab_memory.db").exists():
        import pytest
        pytest.skip("lab_memory.db missing")
    results = search(
        query="router entropy",
        repo_root=REPO,
        top_k=3,
        use_graph=False,
        use_rerank=False,
    )
    assert isinstance(results, list)


def test_search_extracts_seeds_from_query():
    from tools.retrieval.search import _extract_seed_tokens
    seeds = _extract_seed_tokens("Tell me about D-420 and P-D417-DEDUP")
    assert "D-420" in seeds
    assert "P-D417-DEDUP" in seeds
