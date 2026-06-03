"""Tests for tools.retrieval.backfill_metadata against a TINY temp sqlite db.

NEVER points at the real tools/lab_memory.db. Builds a temp `memories` table
matching the live schema, inserts a few rows across 2 source_paths, runs the
backfill in --apply, and asserts file-level tagging, row-count invariance, and
idempotency.
"""
from __future__ import annotations

import sqlite3

import pytest

from tools.retrieval.backfill_metadata import apply_backfill

_SCHEMA = """
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    program_id TEXT NOT NULL DEFAULT '',
    phase TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT '',
    deliverable_type TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    timestamp INTEGER NOT NULL
);
"""


def _make_db(tmp_path):
    db = tmp_path / "tiny.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_SCHEMA)
    rows = [
        # program_2 file -> should get program_id='program_2'
        ("programs/program_2_dense_vs_moe_sub100m/a.md",
         "dense vs moe sub-100m baseline notes", 1),
        # a multi-program memory FILE: two chunks that TOGETHER name two
        # programs -> file-level guard -> program_id='' on BOTH rows.
        ("data/memories/log.md",
         "we worked on program_1_opus47_on_18gb today", 2),
        ("data/memories/log.md",
         "and also pushed program_2_dense_vs_moe_sub100m forward", 2),
        # director agent memory -> role='director'
        ("data/agents/director/semantic.md",
         "director standing orders and routing policy", 3),
    ]
    conn.executemany(
        "INSERT INTO memories (source_path, chunk_text, timestamp) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


def _fetch(db):
    conn = sqlite3.connect(str(db))
    out = conn.execute(
        "SELECT id, source_path, program_id, role, phase, deliverable_type "
        "FROM memories ORDER BY id"
    ).fetchall()
    conn.close()
    return out


def test_backfill_apply_file_level(tmp_path):
    db = _make_db(tmp_path)
    before = _fetch(db)
    assert len(before) == 4

    apply_backfill(str(db), apply=True)

    after = _fetch(db)
    # Row COUNT unchanged (metadata-only UPDATE; no INSERT/DELETE).
    assert len(after) == 4
    # ids unchanged.
    assert [r[0] for r in before] == [r[0] for r in after]

    by_path = {}
    for _id, sp, prog, role, phase, deliv in after:
        by_path.setdefault(sp, []).append((prog, role))

    # program_2 file row got program_id='program_2'.
    p2 = by_path["programs/program_2_dense_vs_moe_sub100m/a.md"]
    assert all(prog == "program_2" for prog, _role in p2)

    # multi-program memory file rows got program_id='' (file-level guard).
    logrows = by_path["data/memories/log.md"]
    assert len(logrows) == 2
    assert all(prog == "" for prog, _role in logrows)

    # director file got role='director'.
    dirrows = by_path["data/agents/director/semantic.md"]
    assert all(role == "director" for _prog, role in dirrows)


def test_backfill_apply_idempotent(tmp_path):
    db = _make_db(tmp_path)
    apply_backfill(str(db), apply=True)
    first = _fetch(db)
    # Running --apply again must not error and must produce identical values.
    apply_backfill(str(db), apply=True)
    second = _fetch(db)
    assert first == second
    assert len(second) == 4


def test_backfill_dry_run_writes_nothing(tmp_path):
    db = _make_db(tmp_path)
    before = _fetch(db)
    apply_backfill(str(db), apply=False)  # default mode
    after = _fetch(db)
    assert before == after
    # Nothing tagged because we wrote nothing.
    assert all(prog == "" and role == "" for _id, _sp, prog, role, _ph, _d in after)
