"""tests/test_work_queue.py — comprehensive tests for tools/work_queue.py (Stage Q1).

All tests use ``tmp_path`` for isolated state via the ``repo_root`` parameter.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.work_queue import (
    claim,
    complete,
    compute_id,
    enqueue,
    fail,
    peek,
    reclaim_stale,
    summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_ITEM = {
    "type": "phase_advance",
    "priority": "normal",
    "program": "program_test",
    "payload": {"from_phase": "P1", "to_phase": "P2"},
    "created_by": "test_suite",
}


def _make(overrides: dict | None = None) -> dict:
    item = dict(_BASE_ITEM)
    if overrides:
        item.update(overrides)
    return item


def _queue_dir(tmp_path: Path) -> Path:
    return tmp_path / "data" / "work_queue"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# 1. enqueue creates item with required fields
# ---------------------------------------------------------------------------


def test_enqueue_creates_item_with_required_fields(tmp_path: Path) -> None:
    """enqueue minimal item; peek returns it with id + created_at populated."""
    item_id = enqueue(_make(), repo_root=tmp_path)
    assert item_id.startswith("wq-")

    result = peek(repo_root=tmp_path)
    assert result is not None
    assert result["id"] == item_id
    assert result["type"] == "phase_advance"
    assert result["priority"] == "normal"
    assert result["created_at"] is not None
    assert result["claimed_at"] is None
    assert result["claimed_by"] is None
    assert result["completed_at"] is None
    assert result["outcome"] is None


# ---------------------------------------------------------------------------
# 2. enqueue is idempotent on same payload
# ---------------------------------------------------------------------------


def test_enqueue_idempotent_on_same_payload(tmp_path: Path) -> None:
    """Enqueueing the same item twice results in only one entry in pending."""
    id1 = enqueue(_make(), repo_root=tmp_path)
    id2 = enqueue(_make(), repo_root=tmp_path)
    assert id1 == id2

    pending = _read_jsonl(_queue_dir(tmp_path) / "pending.jsonl")
    assert len(pending) == 1


# ---------------------------------------------------------------------------
# 3. peek respects priority order
# ---------------------------------------------------------------------------


def test_peek_respects_priority_order(tmp_path: Path) -> None:
    """Enqueueing urgent + normal + low items; peek returns urgent first."""
    enqueue(_make({"priority": "low", "payload": {"k": "low"}}), repo_root=tmp_path)
    enqueue(_make({"priority": "normal", "payload": {"k": "normal"}}), repo_root=tmp_path)
    enqueue(_make({"priority": "urgent", "payload": {"k": "urgent"}}), repo_root=tmp_path)

    result = peek(repo_root=tmp_path)
    assert result is not None
    assert result["priority"] == "urgent"


# ---------------------------------------------------------------------------
# 4. peek FIFO within same priority
# ---------------------------------------------------------------------------


def test_peek_fifo_within_priority(tmp_path: Path) -> None:
    """Two normal items; peek returns the one with the earlier created_at."""
    older_ts = "2026-05-04T10:00:00Z"
    newer_ts = "2026-05-04T11:00:00Z"
    enqueue(
        _make({"payload": {"k": "newer"}, "created_at": newer_ts}),
        repo_root=tmp_path,
    )
    enqueue(
        _make({"payload": {"k": "older"}, "created_at": older_ts}),
        repo_root=tmp_path,
    )

    result = peek(repo_root=tmp_path)
    assert result is not None
    assert result["created_at"] == older_ts


# ---------------------------------------------------------------------------
# 5. claim moves pending item to claimed
# ---------------------------------------------------------------------------


def test_claim_moves_pending_to_claimed(tmp_path: Path) -> None:
    """enqueue, claim; peek returns None; claimed file has the item."""
    item_id = enqueue(_make(), repo_root=tmp_path)
    claimed_item = claim(item_id, claimer="director", repo_root=tmp_path)

    assert claimed_item is not None
    assert claimed_item["id"] == item_id
    assert claimed_item["claimed_by"] == "director"
    assert claimed_item["claimed_at"] is not None

    # pending should be empty
    assert peek(repo_root=tmp_path) is None

    # claimed file should have one item
    claimed_list = _read_jsonl(_queue_dir(tmp_path) / "claimed.jsonl")
    assert len(claimed_list) == 1
    assert claimed_list[0]["id"] == item_id


# ---------------------------------------------------------------------------
# 6. claim one-at-a-time constraint
# ---------------------------------------------------------------------------


def test_claim_one_at_a_time_constraint(tmp_path: Path) -> None:
    """Claiming a second item while one is already claimed raises ValueError."""
    id1 = enqueue(_make({"payload": {"k": "first"}}), repo_root=tmp_path)
    id2 = enqueue(_make({"payload": {"k": "second"}}), repo_root=tmp_path)

    claim(id1, claimer="director", repo_root=tmp_path)

    with pytest.raises(ValueError, match="claimed.jsonl already has an item"):
        claim(id2, claimer="director", repo_root=tmp_path)


# ---------------------------------------------------------------------------
# 7. complete moves claimed to completed/<date>.jsonl
# ---------------------------------------------------------------------------


def test_complete_moves_claimed_to_completed_dated_file(tmp_path: Path) -> None:
    """claim then complete; completed/<today>.jsonl has the item."""
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    item_id = enqueue(_make(), repo_root=tmp_path)
    claim(item_id, claimer="director", repo_root=tmp_path)
    result = complete(item_id, outcome={"status": "ok"}, repo_root=tmp_path)

    assert result is True

    # claimed should be empty
    claimed = _read_jsonl(_queue_dir(tmp_path) / "claimed.jsonl")
    assert claimed == []

    # completed/<today>.jsonl should have the item
    completed_path = _queue_dir(tmp_path) / "completed" / f"{today}.jsonl"
    completed = _read_jsonl(completed_path)
    assert len(completed) == 1
    assert completed[0]["id"] == item_id
    assert completed[0]["outcome"] == {"status": "ok"}
    assert completed[0]["completed_at"] is not None


# ---------------------------------------------------------------------------
# 8. fail moves claimed to failed/<date>.jsonl with reason
# ---------------------------------------------------------------------------


def test_fail_moves_claimed_to_failed_dated_file(tmp_path: Path) -> None:
    """claim then fail; failed/<today>.jsonl has item with reason."""
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    item_id = enqueue(_make(), repo_root=tmp_path)
    claim(item_id, claimer="director", repo_root=tmp_path)
    result = fail(item_id, reason="unknown item type", repo_root=tmp_path)

    assert result is True

    # claimed should be empty
    claimed = _read_jsonl(_queue_dir(tmp_path) / "claimed.jsonl")
    assert claimed == []

    # failed/<today>.jsonl should have the item
    failed_path = _queue_dir(tmp_path) / "failed" / f"{today}.jsonl"
    failed = _read_jsonl(failed_path)
    assert len(failed) == 1
    assert failed[0]["id"] == item_id
    assert failed[0]["outcome"] == {"failed": True, "reason": "unknown item type"}
    assert failed[0]["completed_at"] is not None


# ---------------------------------------------------------------------------
# 9. summary returns correct counts
# ---------------------------------------------------------------------------


def test_summary_returns_correct_counts(tmp_path: Path) -> None:
    """Enqueue varied items, claim one, complete it; summary matches."""
    # Enqueue 3 normal, 1 urgent, 1 low
    ids = []
    ids.append(enqueue(_make({"payload": {"k": "n1"}}), repo_root=tmp_path))
    ids.append(enqueue(_make({"payload": {"k": "n2"}}), repo_root=tmp_path))
    ids.append(enqueue(_make({"payload": {"k": "n3"}}), repo_root=tmp_path))
    ids.append(
        enqueue(_make({"priority": "urgent", "payload": {"k": "u1"}}), repo_root=tmp_path)
    )
    ids.append(
        enqueue(_make({"priority": "low", "payload": {"k": "l1"}}), repo_root=tmp_path)
    )

    # Claim the urgent one
    urgent_id = enqueue(
        _make({"priority": "urgent", "payload": {"k": "u1"}}), repo_root=tmp_path
    )
    # Claim via peek (urgent is first)
    top = peek(repo_root=tmp_path)
    assert top is not None
    claim(top["id"], claimer="director", repo_root=tmp_path)
    complete(top["id"], outcome={"done": True}, repo_root=tmp_path)

    s = summary(repo_root=tmp_path)
    # After completing one: pending = 4, claimed = 0, completed_today = 1
    assert s["pending"] == 4
    assert s["claimed"] == 0
    assert s["completed_today"] == 1
    assert s["failed_today"] == 0
    # by_type covers pending+claimed only
    assert s["by_type"].get("phase_advance", 0) == 4
    # by_priority: 3 normal + 1 low remaining in pending
    assert s["by_priority"].get("normal", 0) == 3
    assert s["by_priority"].get("low", 0) == 1


# ---------------------------------------------------------------------------
# 10. reclaim_stale moves old claimed items back to pending
# ---------------------------------------------------------------------------


def test_reclaim_stale_moves_old_claimed_back_to_pending(tmp_path: Path) -> None:
    """Claim an item, backdate its claimed_at, reclaim_stale returns 1."""
    item_id = enqueue(_make(), repo_root=tmp_path)
    claim(item_id, claimer="director", repo_root=tmp_path)

    # Manually backdate claimed_at to 2 hours ago
    claimed_path = _queue_dir(tmp_path) / "claimed.jsonl"
    claimed = _read_jsonl(claimed_path)
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    claimed[0]["claimed_at"] = old_ts
    claimed_path.write_text(
        "\n".join(json.dumps(it) for it in claimed) + "\n", encoding="utf-8"
    )

    reclaimed = reclaim_stale(timeout_min=60, repo_root=tmp_path)
    assert reclaimed == 1

    # Item should be back in pending with reclaim_count=1
    result = peek(repo_root=tmp_path)
    assert result is not None
    assert result["id"] == item_id
    assert result["payload"]["reclaim_count"] == 1

    # claimed should be empty
    claimed_after = _read_jsonl(claimed_path)
    assert claimed_after == []


# ---------------------------------------------------------------------------
# 11. compute_id is deterministic
# ---------------------------------------------------------------------------


def test_compute_id_is_deterministic() -> None:
    """Same inputs → same id; different inputs → different id."""
    id_a = compute_id("phase_advance", "program_test", {"from_phase": "P1", "to_phase": "P2"})
    id_b = compute_id("phase_advance", "program_test", {"from_phase": "P1", "to_phase": "P2"})
    assert id_a == id_b
    assert id_a.startswith("wq-")
    assert len(id_a) == len("wq-") + 8  # wq- + 8 hex chars

    id_c = compute_id("phase_advance", "program_test", {"from_phase": "P1", "to_phase": "P3"})
    assert id_a != id_c

    id_d = compute_id("cell_failed", "program_test", {"from_phase": "P1", "to_phase": "P2"})
    assert id_a != id_d

    id_e = compute_id("phase_advance", "program_other", {"from_phase": "P1", "to_phase": "P2"})
    assert id_a != id_e


# ---------------------------------------------------------------------------
# 12. telemetry records every state change
# ---------------------------------------------------------------------------


def test_telemetry_records_every_state_change(tmp_path: Path) -> None:
    """enqueue + claim + complete → telemetry has 3 entries with correct actions."""
    item_id = enqueue(_make(), repo_root=tmp_path)
    claim(item_id, claimer="director", repo_root=tmp_path)
    complete(item_id, outcome={"ok": True}, repo_root=tmp_path)

    tel_path = _queue_dir(tmp_path) / "queue_telemetry.jsonl"
    entries = _read_jsonl(tel_path)
    assert len(entries) == 3

    actions = [e["action"] for e in entries]
    assert actions == ["enqueue", "claim", "complete"]

    for e in entries:
        assert e["item_id"] == item_id
        assert e["item_type"] == "phase_advance"
        assert "ts" in e

    # claim entry should record the claimer
    claim_entry = next(e for e in entries if e["action"] == "claim")
    assert claim_entry["claimer"] == "director"


# ---------------------------------------------------------------------------
# Bonus: edge-case tests
# ---------------------------------------------------------------------------


def test_enqueue_missing_required_field_raises(tmp_path: Path) -> None:
    """enqueue without 'type' raises ValueError."""
    with pytest.raises(ValueError, match="missing required fields"):
        enqueue({"priority": "normal", "created_by": "test"}, repo_root=tmp_path)


def test_enqueue_invalid_priority_raises(tmp_path: Path) -> None:
    """enqueue with invalid priority raises ValueError."""
    with pytest.raises(ValueError, match="invalid priority"):
        enqueue(
            {"type": "phase_advance", "priority": "critical", "created_by": "test"},
            repo_root=tmp_path,
        )


def test_claim_returns_none_for_nonexistent_id(tmp_path: Path) -> None:
    """claim on a non-existent id returns None."""
    result = claim("wq-00000000", claimer="director", repo_root=tmp_path)
    assert result is None


def test_complete_returns_false_for_nonexistent_id(tmp_path: Path) -> None:
    """complete on a non-existent id returns False."""
    result = complete("wq-00000000", outcome={}, repo_root=tmp_path)
    assert result is False


def test_fail_returns_false_for_nonexistent_id(tmp_path: Path) -> None:
    """fail on a non-existent id returns False."""
    result = fail("wq-00000000", reason="nope", repo_root=tmp_path)
    assert result is False


def test_peek_empty_queue_returns_none(tmp_path: Path) -> None:
    """peek on empty queue returns None."""
    assert peek(repo_root=tmp_path) is None


def test_reclaim_stale_fresh_item_not_reclaimed(tmp_path: Path) -> None:
    """An item claimed just now is not reclaimed by reclaim_stale(60)."""
    item_id = enqueue(_make(), repo_root=tmp_path)
    claim(item_id, claimer="director", repo_root=tmp_path)
    reclaimed = reclaim_stale(timeout_min=60, repo_root=tmp_path)
    assert reclaimed == 0


def test_enqueue_idempotent_when_already_claimed(tmp_path: Path) -> None:
    """Re-enqueueing an item that's in claimed returns its ID without duplication."""
    item_id = enqueue(_make(), repo_root=tmp_path)
    claim(item_id, claimer="director", repo_root=tmp_path)
    # Try to enqueue same item again — should be a no-op
    id2 = enqueue(_make(), repo_root=tmp_path)
    assert id2 == item_id
    # pending stays empty (item is in claimed, not re-added to pending)
    assert peek(repo_root=tmp_path) is None


def test_telemetry_records_fail_action(tmp_path: Path) -> None:
    """fail action is recorded in telemetry."""
    item_id = enqueue(_make(), repo_root=tmp_path)
    claim(item_id, claimer="director", repo_root=tmp_path)
    fail(item_id, reason="test failure", repo_root=tmp_path)

    tel = _read_jsonl(_queue_dir(tmp_path) / "queue_telemetry.jsonl")
    actions = [e["action"] for e in tel]
    assert "fail" in actions


def test_telemetry_records_reclaim_action(tmp_path: Path) -> None:
    """reclaim action is recorded in telemetry."""
    item_id = enqueue(_make(), repo_root=tmp_path)
    claim(item_id, claimer="director", repo_root=tmp_path)

    # Backdate to trigger reclaim
    claimed_path = _queue_dir(tmp_path) / "claimed.jsonl"
    claimed = _read_jsonl(claimed_path)
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    claimed[0]["claimed_at"] = old_ts
    claimed_path.write_text(
        "\n".join(json.dumps(it) for it in claimed) + "\n", encoding="utf-8"
    )

    reclaim_stale(timeout_min=60, repo_root=tmp_path)

    tel = _read_jsonl(_queue_dir(tmp_path) / "queue_telemetry.jsonl")
    actions = [e["action"] for e in tel]
    assert "reclaim" in actions
