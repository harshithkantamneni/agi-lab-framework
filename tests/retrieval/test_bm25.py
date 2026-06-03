from pathlib import Path


def test_bm25_save_is_atomic_no_tmp_leftover(tmp_path):
    """save() must land the index via an atomic rename and leave no .tmp file,
    so a concurrent reader never observes a partially-written 96MB JSON."""
    from tools.retrieval.bm25 import BM25Index
    idx = BM25Index.from_chunks([{"id": 1, "source_path": "a.md", "chunk_text": "alpha beta"}])
    out = tmp_path / "lab_bm25.json"
    idx.save(out)
    assert out.exists()
    leftovers = list(tmp_path.glob("*.tmp"))
    assert not leftovers, f"temp files left behind: {leftovers}"
    # round-trips
    reloaded = BM25Index.load(out)
    assert reloaded.search("alpha", top_k=1)[0]["id"] == 1


def test_bm25_search_respects_program_filter():
    """Metadata filtering must actually scope results (was untested)."""
    from tools.retrieval.bm25 import BM25Index
    chunks = [
        {"id": 1, "source_path": "a.md", "program_id": "P1", "chunk_text": "router entropy collapse"},
        {"id": 2, "source_path": "b.md", "program_id": "P2", "chunk_text": "router entropy collapse"},
    ]
    idx = BM25Index.from_chunks(chunks)
    hits = idx.search("router entropy", top_k=10, program="P1")
    assert hits, "expected a P1 hit"
    assert all(h.get("program_id") == "P1" for h in hits)
    assert all(h["id"] != 2 for h in hits)


def test_bm25_index_build_and_query(tmp_path):
    from tools.retrieval.bm25 import BM25Index
    chunks = [
        {"id": 1, "source_path": "a.md", "chunk_text": "D-420 closed Program 2 phase 3"},
        {"id": 2, "source_path": "b.md", "chunk_text": "router entropy collapse mechanism"},
        {"id": 3, "source_path": "c.md", "chunk_text": "P-D417 carry forward dedup gap"},
        {"id": 4, "source_path": "d.md", "chunk_text": "dense vs MoE comparison at sub-100M"},
    ]
    idx = BM25Index.from_chunks(chunks)
    hits = idx.search("D-420 Program 2", top_k=2)
    assert len(hits) > 0
    assert hits[0]["id"] == 1
    hits = idx.search("P-D417 dedup", top_k=2)
    assert hits[0]["id"] == 3


def test_bm25_persists_and_reloads_via_json(tmp_path):
    """BM25 state is persisted as JSON (precomputed stats — no pickle anywhere).
    On load, the scorer is rehydrated from stats without recomputing the corpus."""
    from tools.retrieval.bm25 import BM25Index
    chunks = [{"id": 1, "source_path": "a.md", "chunk_text": "alpha beta gamma"}]
    idx_path = tmp_path / "bm25.json"
    idx = BM25Index.from_chunks(chunks)
    idx.save(idx_path)
    assert idx_path.exists()
    # Verify the file is JSON (not pickle) and holds precomputed stats.
    import json
    data = json.loads(idx_path.read_text())
    assert "chunks" in data
    assert "bm25_stats" in data and "tokenized" not in data
    idx2 = BM25Index.load(idx_path)
    hits = idx2.search("alpha", top_k=1)
    assert hits[0]["id"] == 1


def test_bm25_stats_persistence_preserves_scores(tmp_path):
    """load() rehydrates from stats (no recompute: tokenized is None) and returns
    identical scores/order to the freshly-built index. (audit #4)"""
    from tools.retrieval.bm25 import BM25Index
    chunks = [
        {"id": 1, "source_path": "a.md", "chunk_text": "router entropy collapse"},
        {"id": 2, "source_path": "b.md", "chunk_text": "dense vs MoE comparison"},
        {"id": 3, "source_path": "c.md", "chunk_text": "router entropy at step 500"},
    ]
    idx = BM25Index.from_chunks(chunks)
    expected = idx.search("router entropy", top_k=3)
    out = tmp_path / "bm25.json"
    idx.save(out)
    reloaded = BM25Index.load(out)
    assert reloaded.tokenized is None, "rehydrated from stats, not re-tokenized"
    got = reloaded.search("router entropy", top_k=3)
    assert [h["id"] for h in got] == [h["id"] for h in expected]
    assert all(abs(a["bm25_score"] - b["bm25_score"]) < 1e-9 for a, b in zip(got, expected))


def test_bm25_load_backward_compat_old_tokenized_format(tmp_path):
    """An old-format index file (raw 'tokenized') must still load (rebuild once)."""
    import json
    from tools.retrieval.bm25 import BM25Index, tokenize
    chunks = [{"id": 1, "source_path": "a.md", "chunk_text": "alpha beta gamma"}]
    old = tmp_path / "old.json"
    old.write_text(json.dumps(
        {"chunks": chunks, "tokenized": [tokenize(c["chunk_text"]) for c in chunks]}))
    idx = BM25Index.load(old)
    assert idx.search("alpha", top_k=1)[0]["id"] == 1


def test_bm25_build_from_lab_memory_db(tmp_path):
    from tools.retrieval.bm25 import BM25Index
    import sqlite3
    db = tmp_path / "lab_memory.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY,
                program_id TEXT DEFAULT '',
                phase TEXT DEFAULT '',
                role TEXT DEFAULT '',
                deliverable_type TEXT DEFAULT '',
                source_path TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            );
            INSERT INTO memories (source_path, chunk_text, timestamp) VALUES
                ('a.md', 'alpha beta gamma', 1),
                ('b.md', 'delta epsilon zeta', 2);
        """)
    idx = BM25Index.from_lab_memory_db(db)
    hits = idx.search("alpha", top_k=2)
    assert any(h["chunk_text"] == "alpha beta gamma" for h in hits)


def test_bm25_cli_build_and_search(tmp_path):
    import subprocess
    REPO = Path(__file__).resolve().parent.parent.parent
    db = REPO / "tools/lab_memory.db"
    if not db.exists():
        import pytest
        pytest.skip("lab_memory.db not present; skipping CLI integration")
    idx_path = tmp_path / "bm25_test.json"
    out = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-m", "tools.retrieval.bm25", "build",
         "--index", str(idx_path), "--lab-memory-db", str(db)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert out.returncode == 0, f"build failed: {out.stderr}"
    assert idx_path.exists()

    out = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-m", "tools.retrieval.bm25", "search",
         "router entropy", "--index", str(idx_path), "--top-k", "3"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert out.returncode == 0, f"search failed: {out.stderr[-500:]}"
    assert len(out.stdout) > 0
