def test_reranker_lazy_loads_model():
    from tools.retrieval.rerank import CrossEncoderReranker
    r = CrossEncoderReranker()
    assert r._model is None


def test_reranker_orders_pairs_by_relevance():
    from tools.retrieval.rerank import CrossEncoderReranker
    r = CrossEncoderReranker()
    query = "Program 2 closed with verdict O4"
    candidates = [
        {"id": 1, "chunk_text": "Random unrelated text about cats and dogs."},
        {"id": 2, "chunk_text": "Program 2 dense vs MoE closed at D-421 with verdict O4 'below the measurement floor'."},
        {"id": 3, "chunk_text": "Program 1 envelope paper on Opus-4.7 18GB infeasibility."},
    ]
    reranked = r.rerank(query, candidates, top_k=3)
    top_ids = [c["id"] for c in reranked[:2]]
    assert 2 in top_ids, f"reranker missed relevant chunk; order: {[c['id'] for c in reranked]}"


def test_reranker_truncates_top_k():
    from tools.retrieval.rerank import CrossEncoderReranker
    r = CrossEncoderReranker()
    candidates = [{"id": i, "chunk_text": f"chunk {i}"} for i in range(10)]
    reranked = r.rerank("query", candidates, top_k=3)
    assert len(reranked) == 3
