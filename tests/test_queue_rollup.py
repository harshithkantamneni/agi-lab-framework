"""tests/test_queue_rollup.py — tests for tools/queue_rollup.py (Stage Q6).

All tests use ``tmp_path`` for fully isolated state via monkeypatch of
the REPO constant at module level.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Make sure the repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import queue_rollup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_telemetry(root: Path, records: list[dict]) -> Path:
    """Write queue_telemetry.jsonl with the given records."""
    d = root / "data" / "work_queue"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "queue_telemetry.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return p


def _read_rollup(root: Path) -> str:
    """Read the output rollup markdown."""
    p = root / "data" / "infra" / "queue_rollup.md"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rollup_handles_missing_telemetry(capsys, tmp_path, monkeypatch):
    """Test that missing telemetry file prints 'no telemetry yet' without crashing."""
    monkeypatch.setattr(queue_rollup, "REPO", tmp_path)
    queue_rollup.main()
    captured = capsys.readouterr()
    assert "no telemetry yet" in captured.out


def test_rollup_writes_markdown_with_header(tmp_path, monkeypatch):
    """Test that minimal telemetry produces markdown output with correct header."""
    monkeypatch.setattr(queue_rollup, "REPO", tmp_path)

    # Write minimal telemetry (just one enqueue event)
    _write_telemetry(
        tmp_path,
        [
            {
                "ts": "2026-05-04T22:30:00Z",
                "action": "enqueue",
                "item_id": "wq-test-001",
                "item_type": "phase_advance",
            }
        ],
    )

    queue_rollup.main()
    output = _read_rollup(tmp_path)

    assert "# Work queue rollup" in output
    assert "Generated:" in output
    assert "By item type" in output


def test_rollup_per_type_counts_correct(tmp_path, monkeypatch):
    """Test that per-type counts are correct: 3 enqueued, 2 completed for type A."""
    monkeypatch.setattr(queue_rollup, "REPO", tmp_path)

    # 3 enqueue events for type A, 2 complete events for type A
    _write_telemetry(
        tmp_path,
        [
            {
                "ts": "2026-05-04T22:00:00Z",
                "action": "enqueue",
                "item_id": "wq-a-001",
                "item_type": "phase_advance",
            },
            {
                "ts": "2026-05-04T22:05:00Z",
                "action": "enqueue",
                "item_id": "wq-a-002",
                "item_type": "phase_advance",
            },
            {
                "ts": "2026-05-04T22:10:00Z",
                "action": "enqueue",
                "item_id": "wq-a-003",
                "item_type": "phase_advance",
            },
            {
                "ts": "2026-05-04T22:15:00Z",
                "action": "claim",
                "item_id": "wq-a-001",
                "item_type": "phase_advance",
                "claimer": "director",
            },
            {
                "ts": "2026-05-04T22:20:00Z",
                "action": "complete",
                "item_id": "wq-a-001",
                "item_type": "phase_advance",
            },
            {
                "ts": "2026-05-04T22:25:00Z",
                "action": "claim",
                "item_id": "wq-a-002",
                "item_type": "phase_advance",
                "claimer": "director",
            },
            {
                "ts": "2026-05-04T22:30:00Z",
                "action": "complete",
                "item_id": "wq-a-002",
                "item_type": "phase_advance",
            },
        ],
    )

    queue_rollup.main()
    output = _read_rollup(tmp_path)

    # Check that the phase_advance row has correct counts
    assert "phase_advance" in output
    # Should have: enqueued=3, completed=2, failed=0
    lines = output.split("\n")
    phase_line = [l for l in lines if "phase_advance" in l and "|" in l]
    assert len(phase_line) == 1
    phase_line = phase_line[0]
    assert "| 3 |" in phase_line  # 3 enqueued
    assert "| 2 |" in phase_line  # 2 completed


def test_rollup_computes_claim_complete_latency(tmp_path, monkeypatch):
    """Test that median claim→complete latency is computed correctly (10s apart)."""
    monkeypatch.setattr(queue_rollup, "REPO", tmp_path)

    # Create a claim at T0 and complete at T0+10s
    _write_telemetry(
        tmp_path,
        [
            {
                "ts": "2026-05-04T22:00:00Z",
                "action": "enqueue",
                "item_id": "wq-lat-001",
                "item_type": "cell_complete",
            },
            {
                "ts": "2026-05-04T22:00:00Z",
                "action": "claim",
                "item_id": "wq-lat-001",
                "item_type": "cell_complete",
                "claimer": "director",
            },
            {
                "ts": "2026-05-04T22:00:10Z",
                "action": "complete",
                "item_id": "wq-lat-001",
                "item_type": "cell_complete",
            },
        ],
    )

    queue_rollup.main()
    output = _read_rollup(tmp_path)

    # Check that cell_complete row shows ~10s latency
    lines = output.split("\n")
    cell_line = [l for l in lines if "cell_complete" in l and "|" in l]
    assert len(cell_line) == 1
    # Should contain "10" (or close to it) for the latency
    assert "10" in cell_line[0]


def test_rollup_flags_high_fail_rate(tmp_path, monkeypatch):
    """Test fail-rate flag: 1 completed + 4 failed → 'FAIL RATE HIGH'."""
    monkeypatch.setattr(queue_rollup, "REPO", tmp_path)

    # 1 complete + 4 fail = 80% fail rate > 20% threshold
    _write_telemetry(
        tmp_path,
        [
            {
                "ts": "2026-05-04T22:00:00Z",
                "action": "enqueue",
                "item_id": "wq-fail-001",
                "item_type": "operator_nudge",
            },
            {
                "ts": "2026-05-04T22:01:00Z",
                "action": "enqueue",
                "item_id": "wq-fail-002",
                "item_type": "operator_nudge",
            },
            {
                "ts": "2026-05-04T22:02:00Z",
                "action": "enqueue",
                "item_id": "wq-fail-003",
                "item_type": "operator_nudge",
            },
            {
                "ts": "2026-05-04T22:03:00Z",
                "action": "enqueue",
                "item_id": "wq-fail-004",
                "item_type": "operator_nudge",
            },
            {
                "ts": "2026-05-04T22:04:00Z",
                "action": "enqueue",
                "item_id": "wq-fail-005",
                "item_type": "operator_nudge",
            },
            {
                "ts": "2026-05-04T22:05:00Z",
                "action": "claim",
                "item_id": "wq-fail-001",
                "item_type": "operator_nudge",
                "claimer": "director",
            },
            {
                "ts": "2026-05-04T22:06:00Z",
                "action": "complete",
                "item_id": "wq-fail-001",
                "item_type": "operator_nudge",
            },
            {
                "ts": "2026-05-04T22:07:00Z",
                "action": "claim",
                "item_id": "wq-fail-002",
                "item_type": "operator_nudge",
                "claimer": "director",
            },
            {
                "ts": "2026-05-04T22:08:00Z",
                "action": "fail",
                "item_id": "wq-fail-002",
                "item_type": "operator_nudge",
            },
            {
                "ts": "2026-05-04T22:09:00Z",
                "action": "claim",
                "item_id": "wq-fail-003",
                "item_type": "operator_nudge",
                "claimer": "director",
            },
            {
                "ts": "2026-05-04T22:10:00Z",
                "action": "fail",
                "item_id": "wq-fail-003",
                "item_type": "operator_nudge",
            },
            {
                "ts": "2026-05-04T22:11:00Z",
                "action": "claim",
                "item_id": "wq-fail-004",
                "item_type": "operator_nudge",
                "claimer": "director",
            },
            {
                "ts": "2026-05-04T22:12:00Z",
                "action": "fail",
                "item_id": "wq-fail-004",
                "item_type": "operator_nudge",
            },
            {
                "ts": "2026-05-04T22:13:00Z",
                "action": "claim",
                "item_id": "wq-fail-005",
                "item_type": "operator_nudge",
                "claimer": "director",
            },
            {
                "ts": "2026-05-04T22:14:00Z",
                "action": "fail",
                "item_id": "wq-fail-005",
                "item_type": "operator_nudge",
            },
        ],
    )

    queue_rollup.main()
    output = _read_rollup(tmp_path)

    # Check that operator_nudge row is flagged
    lines = output.split("\n")
    nudge_line = [l for l in lines if "operator_nudge" in l and "|" in l]
    assert len(nudge_line) == 1
    assert "FAIL RATE HIGH" in nudge_line[0]
