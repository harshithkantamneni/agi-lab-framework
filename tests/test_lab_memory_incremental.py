"""Tests for --incremental mode on tools/lab_memory.py LabMemory.ingest_incremental()."""
from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_last_ingest_timestamp_empty_db_returns_zero(tmp_path):
    from tools.lab_memory import LabMemory
    db = tmp_path / "lm.db"
    lm = LabMemory(str(db))
    lm.init()
    assert lm._last_ingest_timestamp() == 0


def test_last_ingest_timestamp_returns_max_after_ingest(tmp_path):
    from tools.lab_memory import LabMemory
    db = tmp_path / "lm.db"
    lm = LabMemory(str(db))
    lm.init()
    f = tmp_path / "note.md"
    f.write_text("hello world")
    before = int(time.time())
    lm.ingest(str(f))
    after = int(time.time())
    ts = lm._last_ingest_timestamp()
    assert before <= ts <= after


def test_incremental_ingest_skips_unchanged_files(tmp_path):
    from tools.lab_memory import LabMemory
    db = tmp_path / "lm.db"
    lm = LabMemory(str(db))
    lm.init()
    root = tmp_path / "tree"
    root.mkdir()
    f = root / "a.md"
    f.write_text("some text")
    # First run: ingests 1 file
    count1 = lm.ingest_incremental([root])
    assert count1 == 1
    # Second run with no changes: ingests 0
    count2 = lm.ingest_incremental([root])
    assert count2 == 0


def test_incremental_ingest_picks_up_modified_file(tmp_path):
    from tools.lab_memory import LabMemory
    db = tmp_path / "lm.db"
    lm = LabMemory(str(db))
    lm.init()
    root = tmp_path / "tree"
    root.mkdir()
    f = root / "a.md"
    f.write_text("v1")
    lm.ingest_incremental([root])  # ingests v1
    time.sleep(1.1)  # mtime resolution safety (>= 1s)
    f.write_text("v2 modified")
    count = lm.ingest_incremental([root])
    assert count == 1  # picks up the modified file
