"""Tests for the orphan_deliverable detector in queue_scanner.py."""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _setup_qrepo(tmp_path):
    """Set up a tmp repo with the queue dir, agents/_shared/handlers.md, and an empty log."""
    (tmp_path / "data/work_queue/completed").mkdir(parents=True)
    (tmp_path / "data/work_queue/failed").mkdir(parents=True)
    (tmp_path / "data/agents/_shared").mkdir(parents=True)
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("# Log\n\n---\n\n")
    # Copy real handler schema
    src = REPO / "data/agents/_shared/work_queue_handlers.md"
    (tmp_path / "data/agents/_shared/work_queue_handlers.md").write_text(src.read_text())
    return tmp_path


def _write_claim(tmp_path, claim: dict):
    p = tmp_path / "data/work_queue/claimed.jsonl"
    with p.open("a") as f:
        f.write(json.dumps(claim) + "\n")


def _backdate(path: Path, age_seconds: float):
    """Force file mtime to be `age_seconds` ago."""
    when = time.time() - age_seconds
    os.utime(path, (when, when))


def test_orphan_emits_recovery_when_old_claim_has_deliverable(tmp_path):
    from tools.queue_scanner import _detect_orphan_deliverable
    _setup_qrepo(tmp_path)

    # Create a claim that's 45 min old
    claim = {
        "id": "wq-orphan01", "type": "phase_advance", "priority": "normal",
        "program": "test_program", "payload": {"program": "test_program", "to_phase": "P10"},
        "created_at": "2026-05-05T03:00:00Z", "created_by": "test",
        "claimed_at": "2026-05-05T03:00:00Z", "claimed_by": "director_test",
        "completed_at": None, "outcome": None,
    }
    _write_claim(tmp_path, claim)
    # Backdate the claimed.jsonl mtime to 45 min ago so the >30m guard trips
    _backdate(tmp_path / "data/work_queue/claimed.jsonl", 45 * 60)

    # Create matching deliverable; touch to make mtime "now"
    deliv = tmp_path / "programs/test_program/phase10_mech.md"
    deliv.parent.mkdir(parents=True)
    deliv.write_text("# Mechanism report\n" + "x" * 500)
    deliv.touch()

    items = _detect_orphan_deliverable(repo_root=tmp_path)
    assert len(items) == 1
    item = items[0]
    assert item["type"] == "session_recovery_review"
    assert item["payload"]["orphan_item_id"] == "wq-orphan01"
    assert "phase10_mech.md" in item["payload"]["deliverable"]


def test_orphan_no_emit_when_claim_is_young(tmp_path):
    from tools.queue_scanner import _detect_orphan_deliverable
    _setup_qrepo(tmp_path)

    claim = {
        "id": "wq-young01", "type": "phase_advance", "priority": "normal",
        "program": "test_program", "payload": {"program": "test_program", "to_phase": "P10"},
        "created_at": "2026-05-05T03:00:00Z", "created_by": "test",
        "claimed_at": "2026-05-05T03:00:00Z", "claimed_by": "director_test",
        "completed_at": None, "outcome": None,
    }
    _write_claim(tmp_path, claim)
    # Don't backdate — claimed.jsonl mtime is "now"

    deliv = tmp_path / "programs/test_program/phase10_mech.md"
    deliv.parent.mkdir(parents=True)
    deliv.write_text("# x\n" + "y" * 200)

    items = _detect_orphan_deliverable(repo_root=tmp_path)
    assert items == []


def test_orphan_no_emit_when_deliverable_missing(tmp_path):
    from tools.queue_scanner import _detect_orphan_deliverable
    _setup_qrepo(tmp_path)

    claim = {
        "id": "wq-nodeliv", "type": "phase_advance", "priority": "normal",
        "program": "test_program", "payload": {"program": "test_program", "to_phase": "P10"},
        "created_at": "2026-05-05T03:00:00Z", "created_by": "test",
        "claimed_at": "2026-05-05T03:00:00Z", "claimed_by": "director_test",
        "completed_at": None, "outcome": None,
    }
    _write_claim(tmp_path, claim)
    _backdate(tmp_path / "data/work_queue/claimed.jsonl", 45 * 60)

    # No deliverable file
    items = _detect_orphan_deliverable(repo_root=tmp_path)
    assert items == []


def test_orphan_no_emit_when_telemetry_shows_completion(tmp_path):
    """If queue_telemetry.jsonl already has a complete event for the item, skip."""
    from tools.queue_scanner import _detect_orphan_deliverable
    _setup_qrepo(tmp_path)

    claim = {
        "id": "wq-completed01", "type": "phase_advance", "priority": "normal",
        "program": "test_program", "payload": {"program": "test_program", "to_phase": "P10"},
        "created_at": "2026-05-05T03:00:00Z", "created_by": "test",
        "claimed_at": "2026-05-05T03:00:00Z", "claimed_by": "director_test",
        "completed_at": None, "outcome": None,
    }
    _write_claim(tmp_path, claim)
    _backdate(tmp_path / "data/work_queue/claimed.jsonl", 45 * 60)

    # Write a complete event in telemetry
    telem = tmp_path / "data/work_queue/queue_telemetry.jsonl"
    telem.write_text(json.dumps({
        "ts": "2026-05-05T03:30:00Z",
        "action": "complete",
        "item_id": "wq-completed01",
        "item_type": "phase_advance",
    }) + "\n")

    deliv = tmp_path / "programs/test_program/phase10_mech.md"
    deliv.parent.mkdir(parents=True)
    deliv.write_text("# x\n" + "y" * 200)

    items = _detect_orphan_deliverable(repo_root=tmp_path)
    assert items == []  # already completed in telemetry → skip


def test_orphan_skips_handler_opt_out(tmp_path):
    """cell_failed has null pattern — orphan detector should skip it (operator must triage)."""
    from tools.queue_scanner import _detect_orphan_deliverable
    _setup_qrepo(tmp_path)

    claim = {
        "id": "wq-cellfail01", "type": "cell_failed", "priority": "urgent",
        "program": "test", "payload": {"cell_id": "A42"},
        "created_at": "2026-05-05T03:00:00Z", "created_by": "test",
        "claimed_at": "2026-05-05T03:00:00Z", "claimed_by": "director_test",
        "completed_at": None, "outcome": None,
    }
    _write_claim(tmp_path, claim)
    _backdate(tmp_path / "data/work_queue/claimed.jsonl", 45 * 60)

    items = _detect_orphan_deliverable(repo_root=tmp_path)
    assert items == []


def test_orphan_idempotent_id_for_same_orphan(tmp_path):
    """Two scans of the same orphan should produce the same item ID (so dedup works)."""
    from tools.queue_scanner import _detect_orphan_deliverable
    from tools.work_queue import compute_id
    _setup_qrepo(tmp_path)

    claim = {
        "id": "wq-orphan02", "type": "phase_advance", "priority": "normal",
        "program": "test_program", "payload": {"program": "test_program", "to_phase": "P10"},
        "created_at": "2026-05-05T03:00:00Z", "created_by": "test",
        "claimed_at": "2026-05-05T03:00:00Z", "claimed_by": "director_test",
        "completed_at": None, "outcome": None,
    }
    _write_claim(tmp_path, claim)
    _backdate(tmp_path / "data/work_queue/claimed.jsonl", 45 * 60)

    deliv = tmp_path / "programs/test_program/phase10_mech.md"
    deliv.parent.mkdir(parents=True)
    deliv.write_text("# x\n" + "y" * 200)

    items_a = _detect_orphan_deliverable(repo_root=tmp_path)
    items_b = _detect_orphan_deliverable(repo_root=tmp_path)
    assert len(items_a) == 1 and len(items_b) == 1
    id_a = compute_id(items_a[0]["type"], items_a[0].get("program"), items_a[0].get("payload", {}))
    id_b = compute_id(items_b[0]["type"], items_b[0].get("program"), items_b[0].get("payload", {}))
    assert id_a == id_b


def test_scan_enqueues_orphan_recovery_e2e(tmp_path):
    """Full scan() pipeline picks up orphan_deliverable items and enqueues them."""
    from tools.queue_scanner import scan
    from tools.work_queue import summary as wq_summary
    import tools.work_queue as wq
    _setup_qrepo(tmp_path)

    # Set up a claim that's 45 min old + matching deliverable
    claim = {
        "id": "wq-e2e01", "type": "phase_advance", "priority": "normal",
        "program": "test_program", "payload": {"program": "test_program", "to_phase": "P10"},
        "created_at": "2026-05-05T03:00:00Z", "created_by": "test",
        "claimed_at": "2026-05-05T03:00:00Z", "claimed_by": "director_test",
        "completed_at": None, "outcome": None,
    }
    _write_claim(tmp_path, claim)
    _backdate(tmp_path / "data/work_queue/claimed.jsonl", 45 * 60)

    deliv = tmp_path / "programs/test_program/phase10_mech.md"
    deliv.parent.mkdir(parents=True)
    deliv.write_text("# Mechanism\n" + "x" * 500)
    deliv.touch()

    # Patch work_queue module-level _REPO so enqueue writes to tmp_path
    import tools.work_queue
    orig_repo = tools.work_queue._REPO
    tools.work_queue._REPO = tmp_path
    try:
        new_ids = scan(repo_root=tmp_path)
    finally:
        tools.work_queue._REPO = orig_repo

    # At least one new item should have been enqueued; one should be session_recovery_review
    pending_path = tmp_path / "data/work_queue/pending.jsonl"
    assert pending_path.exists()
    pending_types = []
    for line in pending_path.read_text().splitlines():
        if line.strip():
            pending_types.append(json.loads(line)["type"])
    assert "session_recovery_review" in pending_types


def test_scan_orphan_dedup_across_runs(tmp_path):
    """Running scan() twice does not double-enqueue the same orphan."""
    from tools.queue_scanner import scan
    _setup_qrepo(tmp_path)

    claim = {
        "id": "wq-e2e02", "type": "phase_advance", "priority": "normal",
        "program": "test_program", "payload": {"program": "test_program", "to_phase": "P10"},
        "created_at": "2026-05-05T03:00:00Z", "created_by": "test",
        "claimed_at": "2026-05-05T03:00:00Z", "claimed_by": "director_test",
        "completed_at": None, "outcome": None,
    }
    _write_claim(tmp_path, claim)
    _backdate(tmp_path / "data/work_queue/claimed.jsonl", 45 * 60)

    deliv = tmp_path / "programs/test_program/phase10_mech.md"
    deliv.parent.mkdir(parents=True)
    deliv.write_text("# Mechanism\n" + "x" * 500)
    deliv.touch()

    import tools.work_queue
    orig_repo = tools.work_queue._REPO
    tools.work_queue._REPO = tmp_path
    try:
        scan(repo_root=tmp_path)
        # The first scan enqueues; the second scan should NOT add another orphan recovery
        # NOTE: subsequent scans may re-touch claimed.jsonl mtime; we re-backdate before
        # the second run to keep the >30m guard tripping
        _backdate(tmp_path / "data/work_queue/claimed.jsonl", 45 * 60)
        scan(repo_root=tmp_path)
    finally:
        tools.work_queue._REPO = orig_repo

    pending_path = tmp_path / "data/work_queue/pending.jsonl"
    recovery_count = 0
    for line in pending_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("type") == "session_recovery_review" \
               and r.get("payload", {}).get("orphan_item_id") == "wq-e2e01" or \
               r.get("payload", {}).get("orphan_item_id") == "wq-e2e02":
                recovery_count += 1
    # Both wq-e2e01 (from previous test) and wq-e2e02 may match by ID;
    # but for THIS specific orphan, only one recovery item should exist.
    # Filter by orphan_item_id == wq-e2e02:
    e2e02_count = 0
    for line in pending_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if (r.get("type") == "session_recovery_review"
                    and r.get("payload", {}).get("orphan_item_id") == "wq-e2e02"):
                e2e02_count += 1
    assert e2e02_count == 1, f"expected 1 recovery item for wq-e2e02, got {e2e02_count}"
