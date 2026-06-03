"""Tests for tools/lab_memory.py -- local semantic memory for the AGI lab."""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

# Make tools/ importable
sys.path.insert(0, str(Path(__file__).parent))

import lab_memory


def test_init_creates_db_with_schema():
    """LabMemory.init() creates a SQLite DB with the expected schema."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        lm = lab_memory.LabMemory(db_path)
        lm.init()

        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)

        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "memories" in tables, f"expected 'memories' table, got {tables}"
        assert "memories_vec" in tables, f"expected 'memories_vec' table, got {tables}"

        cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
        required_cols = {
            "id", "program_id", "phase", "role",
            "deliverable_type", "source_path", "chunk_text", "timestamp"
        }
        assert required_cols.issubset(cols), f"missing cols: {required_cols - cols}"
        conn.close()


def test_ingest_chunks_and_embeds_file():
    """ingest() reads a file, chunks it, embeds, and persists."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        source_path = os.path.join(tmp, "doc.md")
        # Write a ~2000-char doc — should chunk into 2-3 chunks
        with open(source_path, "w") as f:
            f.write("# Doc\n\n" + ("Sentence about router entropy. " * 40) + "\n\n")
            f.write("Second paragraph about backprop validation. " * 30 + "\n\n")
            f.write("Third paragraph about scaling laws. " * 30 + "\n")

        lm = lab_memory.LabMemory(db_path)
        lm.init()
        n = lm.ingest(
            source_path,
            program_id="program_test",
            phase="P2",
            role="literature_hunter",
            deliverable_type="prior_work",
        )
        assert n >= 2, f"expected at least 2 chunks, got {n}"

        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        rows = conn.execute(
            "SELECT program_id, phase, role, deliverable_type FROM memories"
        ).fetchall()
        assert len(rows) == n
        assert all(r[0] == "program_test" for r in rows)
        assert all(r[1] == "P2" for r in rows)
        vec_count = conn.execute(
            "SELECT COUNT(*) FROM memories_vec"
        ).fetchone()[0]
        assert vec_count == n, f"vec count {vec_count} != memory count {n}"
        conn.close()


def test_search_returns_relevant_chunks():
    """search() returns chunks ranked by cosine similarity."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        # Create three distinct topic docs
        topics = {
            "doc_routing.md": "Mixture of experts routing collapses when entropy "
                              "drops below a threshold. Load balancing via EMA bias "
                              "updates is one remedy. " * 5,
            "doc_backprop.md": "Backpropagation is the chain rule applied to computational "
                               "graphs. Gradient magnitudes affect convergence. " * 5,
            "doc_metal.md": "Metal Shading Language compiles to the Apple GPU. "
                            "Threadgroups of 32 threads map to a SIMD group. " * 5,
        }
        for name, body in topics.items():
            with open(os.path.join(tmp, name), "w") as f:
                f.write(body)

        lm = lab_memory.LabMemory(db_path)
        lm.init()
        for name in topics:
            lm.ingest(
                os.path.join(tmp, name),
                program_id="test",
                phase="P2",
                role="paper_digester",
                deliverable_type="digest",
            )

        hits = lm.search("mixture of experts router balance", top_k=2)
        assert len(hits) == 2
        # Top hit should be the routing doc
        assert "routing" in hits[0].source_path.lower() or "doc_routing" in hits[0].source_path
        assert isinstance(hits[0].distance, float)
        assert hits[0].distance <= hits[1].distance  # ascending distance


def test_search_filters_by_program():
    """search() with program_id filter returns only that program's chunks."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        # Two docs, different programs, same topic
        for prog in ("prog_a", "prog_b"):
            path = os.path.join(tmp, f"{prog}.md")
            with open(path, "w") as f:
                f.write(f"Program {prog} notes on entropy collapse. " * 20)
            lm = lab_memory.LabMemory(db_path)
            lm.init()
            lm.ingest(path, program_id=prog, phase="P1", role="pi",
                      deliverable_type="notes")

        lm = lab_memory.LabMemory(db_path)
        hits = lm.search("entropy collapse", program_id="prog_a", top_k=5)
        assert len(hits) >= 1
        assert all(h.program_id == "prog_a" for h in hits)


def test_list_returns_metadata_without_vectors():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        path = os.path.join(tmp, "doc.md")
        with open(path, "w") as f:
            f.write("Lab memory test content. " * 20)

        lm = lab_memory.LabMemory(db_path)
        lm.init()
        lm.ingest(path, program_id="p1", role="math_theorist", phase="P3",
                  deliverable_type="frame")

        rows = lm.list(program_id="p1")
        assert len(rows) >= 1
        assert rows[0]["program_id"] == "p1"
        assert rows[0]["role"] == "math_theorist"
        assert "chunk_text" in rows[0]


def test_get_returns_single_record():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        path = os.path.join(tmp, "doc.md")
        with open(path, "w") as f:
            f.write("Content for get test. " * 20)
        lm = lab_memory.LabMemory(db_path)
        lm.init()
        lm.ingest(path, program_id="p1")
        rows = lm.list()
        rid = rows[0]["id"]
        rec = lm.get(rid)
        assert rec is not None
        assert rec["id"] == rid


def test_delete_removes_record_and_vector():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        path = os.path.join(tmp, "doc.md")
        with open(path, "w") as f:
            f.write("Content for delete test. " * 20)
        lm = lab_memory.LabMemory(db_path)
        lm.init()
        lm.ingest(path, program_id="p1")
        rid = lm.list()[0]["id"]

        assert lm.delete(rid) is True
        assert lm.get(rid) is None
        # Ensure search skips deleted row
        hits = lm.search("delete test", top_k=5)
        assert all(h.id != rid for h in hits)
