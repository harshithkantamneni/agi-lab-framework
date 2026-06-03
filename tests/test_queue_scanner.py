"""tests/test_queue_scanner.py — tests for tools/queue_scanner.py (Stage Q2).

All tests use ``tmp_path`` for fully isolated state via the ``repo_root``
parameter accepted by scan() and each individual detector.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Make sure the repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.queue_scanner import (
    _detect_cell_events,
    _detect_consolidator_run,
    _detect_diagnostic_files,
    _detect_heartbeat,
    _detect_operator_nudge,
    _detect_phase_advances,
    scan,
)
from tools.work_queue import peek, summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_current_md(root: Path, program: str) -> None:
    """Create data/memories/current.md with the given active program name."""
    d = root / "data" / "memories"
    d.mkdir(parents=True, exist_ok=True)
    (d / "current.md").write_text(
        f"## Current Program: `{program}` — some description\n",
        encoding="utf-8",
    )


def _write_close_memo(root: Path, program: str, phase_n: int) -> Path:
    """Create programs/<program>/phase<N>_close_memo.md."""
    d = root / "programs" / program
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"phase{phase_n}_close_memo.md"
    p.write_text(f"# Phase {phase_n} close\n", encoding="utf-8")
    return p


def _write_user_notes(root: Path, content: str = "A nudge.") -> Path:
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "user_notes.md"
    p.write_text(content, encoding="utf-8")
    return p


def _write_run_index(root: Path, cells: dict) -> Path:
    d = root / "data" / "checkpoints" / "phase3_factorial"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "run_index.json"
    p.write_text(json.dumps(cells, indent=2), encoding="utf-8")
    return p


def _write_diagnostic(root: Path, name: str) -> Path:
    d = root / "data" / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text("# diag\n", encoding="utf-8")
    return p


def _last_seen(root: Path) -> dict:
    p = root / "data" / "work_queue" / "last_seen.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Helpers: suppress synthetic detectors so pre-Q5 tests keep exact counts
# ---------------------------------------------------------------------------

def _suppress_synthetic_detectors(root: Path) -> None:
    """Write a recent consolidator.last_run so synthetic detectors stay silent.

    Without this, _detect_consolidator_run fires on every fresh tmp_path
    (no file = last_run=0 = always stale). Tests that assert exact item
    counts must call this at the start of their setup.
    """
    d = root / "data" / "infra"
    d.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    (d / "consolidator.last_run").write_text(f"{ts} 0\n", encoding="utf-8")
    # No session_logs dir → _detect_heartbeat returns [] (cold start path).


# ---------------------------------------------------------------------------
# 1. Fresh state → no items
# ---------------------------------------------------------------------------

def test_scan_empty_state_produces_no_items(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    ids = scan(repo_root=tmp_path)
    assert ids == []


# ---------------------------------------------------------------------------
# 2. phase_advance detected from close memo
# ---------------------------------------------------------------------------

def test_phase_advance_detected_from_close_memo(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    _write_current_md(tmp_path, "prog_test")
    _write_close_memo(tmp_path, "prog_test", 9)

    ids = scan(repo_root=tmp_path)

    assert len(ids) == 1
    # Verify the enqueued item has the right type + phases
    item = peek(repo_root=tmp_path)
    assert item is not None
    assert item["type"] == "phase_advance"
    assert item["payload"]["from_phase"] == "P9"
    assert item["payload"]["to_phase"] == "P10"


# ---------------------------------------------------------------------------
# 3. phase_advance idempotent (second scan = 0)
# ---------------------------------------------------------------------------

def test_phase_advance_idempotent(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    _write_current_md(tmp_path, "prog_test")
    _write_close_memo(tmp_path, "prog_test", 9)

    ids1 = scan(repo_root=tmp_path)
    assert len(ids1) == 1

    ids2 = scan(repo_root=tmp_path)
    assert ids2 == []


# ---------------------------------------------------------------------------
# 4. phase_advance skips when P14 already closed
# ---------------------------------------------------------------------------

def test_phase_advance_skips_when_p14_already_closed(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    _write_current_md(tmp_path, "prog_test")
    _write_close_memo(tmp_path, "prog_test", 14)

    ids = scan(repo_root=tmp_path)
    assert ids == []


# ---------------------------------------------------------------------------
# 5. operator_nudge on user_notes change
# ---------------------------------------------------------------------------

def test_operator_nudge_on_user_notes_change(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    _write_user_notes(tmp_path, "Please investigate issue X.")

    ids = scan(repo_root=tmp_path)

    assert len(ids) == 1
    item = peek(repo_root=tmp_path)
    assert item is not None
    assert item["type"] == "operator_nudge"
    assert item["priority"] == "urgent"
    assert "Please investigate" in item["payload"]["context"]


# ---------------------------------------------------------------------------
# 6. operator_nudge idempotent until mtime changes
# ---------------------------------------------------------------------------

def test_operator_nudge_idempotent_until_mtime_changes(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    p = _write_user_notes(tmp_path, "First content.")

    ids1 = scan(repo_root=tmp_path)
    assert len(ids1) == 1

    # Second scan with same mtime → no new item
    ids2 = scan(repo_root=tmp_path)
    assert ids2 == []

    # Touch the file (change mtime) — ensure mtime actually changes
    time.sleep(0.01)
    p.write_text("Updated content.", encoding="utf-8")
    # Force a distinct mtime by touching again after a brief gap
    time.sleep(0.05)
    p.write_text("Updated content.", encoding="utf-8")

    ids3 = scan(repo_root=tmp_path)
    assert len(ids3) == 1
    item = peek(repo_root=tmp_path)
    # The new nudge item should have been enqueued (its ID is mtime-based)
    assert item is not None
    assert item["type"] == "operator_nudge"


# ---------------------------------------------------------------------------
# 7. cell_complete detected
# ---------------------------------------------------------------------------

def test_cell_complete_detected(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    _write_current_md(tmp_path, "prog_test")
    _write_run_index(tmp_path, {
        "A42": {
            "state": "completed",
            "rc": 0,
            "fatal": 0,
            "error": 0,
        }
    })

    ids = scan(repo_root=tmp_path)
    assert len(ids) == 1
    item = peek(repo_root=tmp_path)
    assert item is not None
    assert item["type"] == "cell_complete"
    assert item["priority"] == "normal"
    assert item["payload"]["cell"] == "A42"


# ---------------------------------------------------------------------------
# 8. cell_failed detected at urgent priority
# ---------------------------------------------------------------------------

def test_cell_failed_detected_urgent(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    _write_current_md(tmp_path, "prog_test")
    _write_run_index(tmp_path, {
        "B42": {
            "state": "failed",
            "rc": 1,
            "fatal": 0,
            "error": 0,
        }
    })

    ids = scan(repo_root=tmp_path)
    assert len(ids) == 1
    item = peek(repo_root=tmp_path)
    assert item is not None
    assert item["type"] == "cell_failed"
    assert item["priority"] == "urgent"
    assert item["payload"]["cell"] == "B42"


# ---------------------------------------------------------------------------
# 9. cells_queued tracked idempotent (second scan = 0 new)
# ---------------------------------------------------------------------------

def test_cells_queued_tracked_idempotent(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    _write_current_md(tmp_path, "prog_test")
    _write_run_index(tmp_path, {
        "A42": {"state": "completed", "rc": 0, "fatal": 0, "error": 0},
    })

    ids1 = scan(repo_root=tmp_path)
    assert len(ids1) == 1

    ids2 = scan(repo_root=tmp_path)
    assert ids2 == []


# ---------------------------------------------------------------------------
# 10. diagnostic_files detected
# ---------------------------------------------------------------------------

def test_diagnostic_files_detected(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    _write_current_md(tmp_path, "prog_test")
    _write_diagnostic(tmp_path, "foo.md")

    ids = scan(repo_root=tmp_path)
    assert len(ids) == 1
    item = peek(repo_root=tmp_path)
    assert item is not None
    assert item["type"] == "diagnostic_review"
    assert "foo.md" in item["payload"]["file"]


# ---------------------------------------------------------------------------
# 11. diagnostic_files idempotent per path (second scan = 0)
# ---------------------------------------------------------------------------

def test_diagnostic_files_idempotent_per_path(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    _write_current_md(tmp_path, "prog_test")
    _write_diagnostic(tmp_path, "bar.md")

    ids1 = scan(repo_root=tmp_path)
    assert len(ids1) == 1

    ids2 = scan(repo_root=tmp_path)
    assert ids2 == []


# ---------------------------------------------------------------------------
# 12. last_seen.json written after scan
# ---------------------------------------------------------------------------

def test_scan_writes_last_seen_json(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    _write_user_notes(tmp_path)

    scan(repo_root=tmp_path)

    ls = _last_seen(tmp_path)
    assert "last_scan_at" in ls
    assert "user_notes_mtime" in ls


# ---------------------------------------------------------------------------
# 13. Multiple event types simultaneously
# ---------------------------------------------------------------------------

def test_scan_combines_multiple_event_types(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    program = "prog_multi"
    _write_current_md(tmp_path, program)
    _write_close_memo(tmp_path, program, 5)
    _write_user_notes(tmp_path, "Nudge content.")
    _write_run_index(tmp_path, {
        "C42": {"state": "completed", "rc": 0, "fatal": 0, "error": 0},
    })

    ids = scan(repo_root=tmp_path)

    # Should have 3 items: phase_advance + operator_nudge + cell_complete
    assert len(ids) == 3

    s = summary(repo_root=tmp_path)
    assert s["pending"] == 3
    types_present = set(s["by_type"].keys())
    assert "phase_advance" in types_present
    assert "operator_nudge" in types_present
    assert "cell_complete" in types_present


# ---------------------------------------------------------------------------
# 14. main() CLI runs scan and prints summary
# ---------------------------------------------------------------------------

def test_main_cli_runs_scan_and_prints_summary(tmp_path):
    """Run queue_scanner as a subprocess with a --repo-root override."""
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "tools" / "queue_scanner.py"),
            "--repo-root", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    # Fresh state → 0 new items message
    assert "0 new items" in result.stdout or "queue_scanner:" in result.stdout


# ---------------------------------------------------------------------------
# Extra: cell with fatal>0 → cell_failed
# ---------------------------------------------------------------------------

def test_cell_failed_on_fatal_flag(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    _write_current_md(tmp_path, "prog_test")
    _write_run_index(tmp_path, {
        "C99": {
            "state": "completed",  # state says completed but fatal fired
            "rc": 0,
            "fatal": 1,
            "error": 0,
        }
    })

    ids = scan(repo_root=tmp_path)
    assert len(ids) == 1
    item = peek(repo_root=tmp_path)
    assert item["type"] == "cell_failed"
    assert item["priority"] == "urgent"


# ---------------------------------------------------------------------------
# Extra: _meta keys in run_index.json are ignored
# ---------------------------------------------------------------------------

def test_cell_events_ignores_meta_keys(tmp_path):
    _write_current_md(tmp_path, "prog_test")
    _write_run_index(tmp_path, {
        "_meta": {"schema": "v1"},
        "A42": {"state": "completed", "rc": 0, "fatal": 0, "error": 0},
    })

    items = _detect_cell_events(repo_root=tmp_path)
    cells = [i["payload"]["cell"] for i in items]
    assert "_meta" not in cells
    assert "A42" in cells


# ---------------------------------------------------------------------------
# Extra: phase_advance picks max N when multiple close memos exist
# ---------------------------------------------------------------------------

def test_phase_advance_uses_highest_closed_phase(tmp_path):
    _suppress_synthetic_detectors(tmp_path)
    _write_current_md(tmp_path, "prog_test")
    _write_close_memo(tmp_path, "prog_test", 3)
    _write_close_memo(tmp_path, "prog_test", 7)
    _write_close_memo(tmp_path, "prog_test", 5)

    ids = scan(repo_root=tmp_path)
    assert len(ids) == 1
    item = peek(repo_root=tmp_path)
    assert item["payload"]["from_phase"] == "P7"
    assert item["payload"]["to_phase"] == "P8"


# ---------------------------------------------------------------------------
# Extra: _active_program regex anchored to start-of-line
# ---------------------------------------------------------------------------

def test_active_program_regex_skips_code_block_example(tmp_path):
    """Code-block fragment with active_program: must not win over bare-prose value.

    If current.md has a markdown code block containing 'active_program: example'
    BEFORE the real prose 'active_program: real_program', the regex must return
    the real prose value, not the code-block example.
    """
    from tools.queue_scanner import _active_program

    d = tmp_path / "data" / "memories"
    d.mkdir(parents=True, exist_ok=True)
    (d / "current.md").write_text(
        "Some intro text.\n"
        "\n"
        "```yaml\n"
        "# Example — do not copy\n"
        "active_program: example_code_block_value\n"
        "```\n"
        "\n"
        "The real setting:\n"
        "active_program: real_program_name\n",
        encoding="utf-8",
    )

    result = _active_program(tmp_path)
    assert result == "real_program_name", (
        f"expected 'real_program_name' but got {result!r}. "
        "Regex is picking code-block example instead of bare-prose line."
    )


# ---------------------------------------------------------------------------
# Extra: detector functions individually return empty lists gracefully
# ---------------------------------------------------------------------------

def test_detectors_return_empty_on_missing_files(tmp_path):
    # No files at all — each detector should return []
    assert _detect_phase_advances(repo_root=tmp_path) == []
    assert _detect_operator_nudge(repo_root=tmp_path) == []
    assert _detect_cell_events(repo_root=tmp_path) == []
    assert _detect_diagnostic_files(repo_root=tmp_path) == []


# ===========================================================================
# Q5: synthetic-item detectors — consolidator_run
# ===========================================================================

def _write_consolidator_last_run(root: Path, epoch: float | None = None) -> Path:
    """Create data/infra/consolidator.last_run with given epoch (default=now)."""
    d = root / "data" / "infra"
    d.mkdir(parents=True, exist_ok=True)
    ts = int(epoch if epoch is not None else time.time())
    p = d / "consolidator.last_run"
    p.write_text(f"{ts} 42\n", encoding="utf-8")
    return p


def _write_session_log(root: Path, mtime_offset_hours: float = 0.0) -> Path:
    """Create data/infra/session_logs/session_test.log with given mtime offset."""
    d = root / "data" / "infra" / "session_logs"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "session_test.log"
    p.write_text("session log\n", encoding="utf-8")
    if mtime_offset_hours != 0.0:
        new_mtime = time.time() - (mtime_offset_hours * 3600)
        import os
        os.utime(str(p), (new_mtime, new_mtime))
    return p


# ---------------------------------------------------------------------------
# 15. consolidator_run detected when no last_run file
# ---------------------------------------------------------------------------

def test_consolidator_run_detected_when_no_last_run_file(tmp_path):
    """No consolidator.last_run → last_run=0 → enqueue consolidator_run."""
    items = _detect_consolidator_run(tmp_path)
    assert len(items) == 1
    assert items[0]["type"] == "consolidator_run"
    assert items[0]["priority"] == "low"
    assert items[0]["payload"]["hours_since_last_run"] >= 6.0


# ---------------------------------------------------------------------------
# 16. consolidator_run detected when stale (7h ago)
# ---------------------------------------------------------------------------

def test_consolidator_run_detected_when_stale(tmp_path):
    """consolidator.last_run written 7h ago → enqueue consolidator_run."""
    stale_epoch = time.time() - (7 * 3600)
    _write_consolidator_last_run(tmp_path, epoch=stale_epoch)

    items = _detect_consolidator_run(tmp_path)
    assert len(items) == 1
    assert items[0]["type"] == "consolidator_run"
    assert items[0]["payload"]["hours_since_last_run"] >= 6.0


# ---------------------------------------------------------------------------
# 17. consolidator_run silent when recent
# ---------------------------------------------------------------------------

def test_consolidator_run_silent_when_recent(tmp_path):
    """consolidator.last_run written just now → no item."""
    _write_consolidator_last_run(tmp_path, epoch=time.time())

    items = _detect_consolidator_run(tmp_path)
    assert items == []


# ---------------------------------------------------------------------------
# 18. consolidator_run idempotent within 6h bucket
# ---------------------------------------------------------------------------

def test_consolidator_run_idempotent_within_6h_bucket(tmp_path):
    """Two back-to-back scans with stale last_run → only 1 item enqueued (dedup)."""
    stale_epoch = time.time() - (8 * 3600)
    _write_consolidator_last_run(tmp_path, epoch=stale_epoch)

    ids1 = scan(repo_root=tmp_path)
    consolidator_ids1 = [i for i in ids1]  # could include other items too
    # Count consolidator_run items enqueued in first scan
    from tools.work_queue import peek
    # Re-scan immediately (same bucket window)
    ids2 = scan(repo_root=tmp_path)
    # Second scan should produce 0 new IDs (enqueue deduplicates by ID)
    assert ids2 == []


# ===========================================================================
# Q5: synthetic-item detectors — heartbeat
# ===========================================================================

# ---------------------------------------------------------------------------
# 19. heartbeat returns empty when no session_logs dir (cold start)
# ---------------------------------------------------------------------------

def test_heartbeat_detected_when_no_recent_session(tmp_path):
    """Empty session_logs/ dir (no files) → 0 items (cold start handled elsewhere)."""
    log_dir = tmp_path / "data" / "infra" / "session_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    items = _detect_heartbeat(tmp_path)
    assert items == []


# ---------------------------------------------------------------------------
# 20. heartbeat detected when session log mtime 5h ago
# ---------------------------------------------------------------------------

def test_heartbeat_detected_when_session_old(tmp_path):
    """Newest session log is 5h old → enqueue heartbeat item."""
    _write_session_log(tmp_path, mtime_offset_hours=5.0)

    items = _detect_heartbeat(tmp_path)
    assert len(items) == 1
    assert items[0]["type"] == "heartbeat"
    assert items[0]["priority"] == "low"
    assert items[0]["payload"]["hours_since_last_session"] >= 4.0


# ---------------------------------------------------------------------------
# 21. heartbeat silent when session log mtime 1h ago
# ---------------------------------------------------------------------------

def test_heartbeat_silent_when_session_recent(tmp_path):
    """Newest session log is 1h old → no heartbeat item."""
    _write_session_log(tmp_path, mtime_offset_hours=1.0)

    items = _detect_heartbeat(tmp_path)
    assert items == []


# ---------------------------------------------------------------------------
# 22. heartbeat idempotent within 4h bucket
# ---------------------------------------------------------------------------

def test_heartbeat_idempotent_within_4h_bucket(tmp_path):
    """Two back-to-back scans with old session log → only 1 heartbeat enqueued."""
    _write_session_log(tmp_path, mtime_offset_hours=5.0)

    ids1 = scan(repo_root=tmp_path)
    # There should be at least one new item (the heartbeat)
    heartbeat_present = len(ids1) >= 1

    ids2 = scan(repo_root=tmp_path)
    # Second scan: same bucket window → 0 new items
    assert ids2 == []
