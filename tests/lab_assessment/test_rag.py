import math
import sqlite3
from pathlib import Path
from tools.lab_assessment.rag import (
    recall_at_k, mrr, ndcg_at_k, build_known_item_queries, _index_health,
)


def test_recall_at_k():
    ranked = [[3, 1, 2], [5, 4]]
    truth = [1, 9]   # q0 truth=1 at rank2 (in top3); q1 truth=9 absent
    assert recall_at_k(ranked, truth, k=3) == 0.5
    assert recall_at_k(ranked, truth, k=1) == 0.0


def test_mrr():
    ranked = [[3, 1, 2], [9, 4]]
    truth = [1, 9]   # q0: 1 at rank2 -> 1/2; q1: 9 at rank1 -> 1/1
    assert abs(mrr(ranked, truth) - (0.5 + 1.0) / 2) < 1e-9


def test_ndcg_at_k():
    # single relevant item per query -> IDCG = 1, so nDCG = DCG
    ranked = [[3, 1, 2], [9, 4]]
    truth = [1, 9]   # q0: 1 at index1 -> 1/log2(3); q1: 9 at index0 -> 1/log2(2)=1
    expected = ((1 / math.log2(3)) + 1.0) / 2
    assert abs(ndcg_at_k(ranked, truth, k=3) - expected) < 1e-9


def test_build_known_item_queries():
    chunks = [{"id": 1, "chunk_text": "D-420 closed Program 2 phase 3 with verdict O4"},
              {"id": 2, "chunk_text": "router entropy collapse mechanism note"}]
    qs = build_known_item_queries(chunks, n=2, seed=0)
    assert len(qs) == 2
    assert all("query" in q and "truth_id" in q for q in qs)
    assert {q["truth_id"] for q in qs} == {1, 2}


def _make_memories_db(path: Path, rows: list[dict]) -> None:
    """Build a minimal lab_memory.db with the live `memories` schema and given rows."""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE memories ("
        "id INTEGER, program_id TEXT, phase TEXT, role TEXT, "
        "deliverable_type TEXT, source_path TEXT, chunk_text TEXT, timestamp INTEGER)"
    )
    for i, r in enumerate(rows):
        con.execute(
            "INSERT INTO memories "
            "(id, program_id, phase, role, deliverable_type, source_path, chunk_text, timestamp) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (i,
             r.get("program_id", ""), r.get("phase", ""), r.get("role", ""),
             r.get("deliverable_type", ""), "src.md", "text", 0),
        )
    con.commit()
    con.close()


def test_index_health_per_column_pcts(tmp_path):
    # 4 rows of mixed tagging:
    #   program_id: rows 0,1,2  -> 3/4 = 75.0
    #   role:       rows 0,3    -> 2/4 = 50.0
    #   phase:      row  0      -> 1/4 = 25.0  (noisiest column)
    #   deliverable_type: rows 0,1,2,3 -> 4/4 = 100.0
    # pct_metadata_tagged = program OR role OR phase non-empty -> rows 0,1,2,3 = 100.0
    rows = [
        {"program_id": "P1", "role": "pi",   "phase": "p3", "deliverable_type": "note"},
        {"program_id": "P1", "role": "",     "phase": "",   "deliverable_type": "note"},
        {"program_id": "P2", "role": "",     "phase": "",   "deliverable_type": "note"},
        {"program_id": "",   "role": "eval", "phase": "",   "deliverable_type": "note"},
    ]
    db = tmp_path / "lab_memory.db"
    _make_memories_db(db, rows)

    h = _index_health(tmp_path, db=db)
    assert h["chunks"] == 4
    assert h["program_tagged_pct"] == 75.0
    assert h["role_tagged_pct"] == 50.0
    assert h["phase_tagged_pct"] == 25.0
    assert h["deliverable_tagged_pct"] == 100.0
    # topline = program OR role OR phase (deliverable excluded from the topline)
    assert h["pct_metadata_tagged"] == 100.0


def test_index_health_topline_excludes_deliverable(tmp_path):
    # Only deliverable_type set -> topline (program/role/phase) must be 0, not 100.
    rows = [
        {"program_id": "", "role": "", "phase": "", "deliverable_type": "note"},
        {"program_id": "", "role": "", "phase": "", "deliverable_type": "note"},
    ]
    db = tmp_path / "lab_memory.db"
    _make_memories_db(db, rows)

    h = _index_health(tmp_path, db=db)
    assert h["chunks"] == 2
    assert h["pct_metadata_tagged"] == 0.0
    assert h["deliverable_tagged_pct"] == 100.0
    assert h["program_tagged_pct"] == 0.0


def test_index_health_unavailable_when_db_missing(tmp_path):
    h = _index_health(tmp_path, db=tmp_path / "nope.db")
    for key in ("chunks", "pct_metadata_tagged", "program_tagged_pct",
                "role_tagged_pct", "phase_tagged_pct", "deliverable_tagged_pct"):
        assert h[key] == "unavailable"


def test_index_health_default_db_path_uses_repo(tmp_path):
    # When no explicit db is passed, it must resolve to repo/'tools/lab_memory.db'.
    (tmp_path / "tools").mkdir()
    db = tmp_path / "tools" / "lab_memory.db"
    _make_memories_db(db, [{"program_id": "P1"}])
    h = _index_health(tmp_path)
    assert h["chunks"] == 1
    assert h["program_tagged_pct"] == 100.0
