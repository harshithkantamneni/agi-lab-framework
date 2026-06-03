"""tools/work_queue.py — file-backed priority work queue for the autonomous AGI lab.

Stage Q1 of the event-driven work queue (spec: docs/superpowers/specs/2026-05-04-work-queue.md).

Storage layout (relative to repo root):
    data/work_queue/pending.jsonl
    data/work_queue/claimed.jsonl
    data/work_queue/completed/<YYYY-MM-DD>.jsonl
    data/work_queue/failed/<YYYY-MM-DD>.jsonl
    data/work_queue/queue_telemetry.jsonl

Atomic moves are implemented via tmp-file + os.rename.  Single-Director
constraint: only one item may be claimed at a time.  Idempotent enqueue:
items with the same deterministic ID are silently deduped.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PRIORITY_RANK: dict[str, int] = {"urgent": 0, "normal": 1, "low": 2}
_REQUIRED_FIELDS: frozenset[str] = frozenset({"type", "priority", "created_by"})
_VALID_PRIORITIES: frozenset[str] = frozenset({"urgent", "normal", "low"})

# Module-level repo root override — set this (e.g. via monkeypatch in tests) to
# redirect all queue I/O to a different directory without passing repo_root to
# every call site.  Explicit repo_root arguments still take precedence.
_REPO: Path | None = None


def _queue_dir(repo_root: Path | None) -> Path:
    """Return the work-queue directory, creating it if absent."""
    root = repo_root if repo_root is not None else (_REPO if _REPO is not None else Path(__file__).resolve().parent.parent)
    d = root / "data" / "work_queue"
    d.mkdir(parents=True, exist_ok=True)
    (d / "completed").mkdir(exist_ok=True)
    (d / "failed").mkdir(exist_ok=True)
    return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file; return [] if absent or empty."""
    if not path.exists():
        return []
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(path: Path, items: list[dict]) -> None:
    """Atomically overwrite path with items (tmp + rename)."""
    if not items:
        # Write an empty file atomically
        _atomic_write(path, b"")
        return
    lines = "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n"
    _atomic_write(path, lines.encode("utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    """Write data to a tmp file then rename over path (atomic on POSIX)."""
    dir_ = path.parent
    dir_.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dir_), prefix=".wq_tmp_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.rename(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _append_jsonl(path: Path, item: dict) -> None:
    """Append one JSON line to path (creates if absent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_telemetry(
    qdir: Path,
    action: str,
    item_id: str,
    item_type: str,
    claimer: str | None = None,
) -> None:
    """Append one entry to queue_telemetry.jsonl."""
    entry: dict[str, Any] = {
        "ts": _now_iso(),
        "action": action,
        "item_id": item_id,
        "item_type": item_type,
    }
    if claimer is not None:
        entry["claimer"] = claimer
    _append_jsonl(qdir / "queue_telemetry.jsonl", entry)


def _ids_in_file(path: Path) -> set[str]:
    """Return the set of IDs present in a JSONL file."""
    return {item["id"] for item in _read_jsonl(path) if "id" in item}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_id(item_type: str, program: str | None, payload: dict) -> str:
    """Return a deterministic item ID derived from (type, program, payload).

    The ID is ``wq-<first-8-hex-chars-of-sha256>``.  Callers (queue_scanner,
    orchestrator) can call this before enqueue to pre-compute dedup keys.

    Stability short-circuit: if ``payload`` contains a ``_dedup_key`` field,
    only ``(type, program, _dedup_key)`` is hashed.  This prevents text edits
    to ``payload_template`` descriptions in ``artifact_schema.yaml`` from
    changing the item ID and orphaning existing pending items.
    """
    dedup_key = payload.get("_dedup_key") if isinstance(payload, dict) else None
    if dedup_key:
        canonical = json.dumps(
            {"type": item_type, "program": program, "_dedup_key": dedup_key},
            sort_keys=True,
            ensure_ascii=False,
        )
    else:
        # Sort keys for determinism; include program even if None
        canonical = json.dumps(
            {"type": item_type, "program": program, "payload": payload},
            sort_keys=True,
            ensure_ascii=False,
        )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return f"wq-{digest}"


def enqueue(item: dict, repo_root: Path | None = None) -> str:
    """Append item to pending.jsonl; return the item ID.

    Idempotent: if an item with the same deterministic ID already exists in
    pending OR claimed, return its existing ID without appending.

    Required fields: ``type``, ``priority``, ``created_by``.
    System fields (id, created_at, claimed_at, claimed_by, completed_at,
    outcome) are set automatically if absent.

    Raises:
        ValueError: if a required field is missing or priority is invalid.
    """
    # --- validation ---
    missing = _REQUIRED_FIELDS - item.keys()
    if missing:
        raise ValueError(f"enqueue: missing required fields: {missing}")
    if item["priority"] not in _VALID_PRIORITIES:
        raise ValueError(
            f"enqueue: invalid priority {item['priority']!r}; "
            f"must be one of {sorted(_VALID_PRIORITIES)}"
        )

    qdir = _queue_dir(repo_root)
    pending_path = qdir / "pending.jsonl"
    claimed_path = qdir / "claimed.jsonl"
    completed_dir = qdir / "completed"

    # --- build the full item ---
    full: dict[str, Any] = {
        "type": item["type"],
        "priority": item["priority"],
        "program": item.get("program"),
        "payload": item.get("payload", {}),
        "created_at": item.get("created_at", _now_iso()),
        "created_by": item["created_by"],
        "claimed_at": None,
        "claimed_by": None,
        "completed_at": None,
        "outcome": None,
    }
    # Compute ID from the canonical fields (not from mutable system fields)
    item_id = item.get("id") or compute_id(full["type"], full["program"], full["payload"])
    full["id"] = item_id

    # --- idempotency check (pending + claimed + completed-history) ---
    # P-D417 fix (2026-05-20): also scan completed/*.jsonl so that the same
    # dedup_key cannot be re-projected after the item already ran. Before this,
    # `artifact_queue_projector` could re-emit items every iteration once they
    # completed but the downstream artifact had a different shape than the
    # schema's declared path. Recurred 3× at D-417/D-419/D-420.
    existing_ids = _ids_in_file(pending_path) | _ids_in_file(claimed_path)
    if completed_dir.exists():
        for completed_file in completed_dir.glob("*.jsonl"):
            existing_ids |= _ids_in_file(completed_file)
    if item_id in existing_ids:
        return item_id

    # --- append ---
    _append_jsonl(pending_path, full)
    _write_telemetry(qdir, "enqueue", item_id, full["type"])
    return item_id


def peek(repo_root: Path | None = None) -> dict | None:
    """Return the highest-priority pending item without claiming it.

    Priority order: urgent > normal > low.  Within the same priority,
    FIFO by ``created_at`` (ascending).  Returns None if pending is empty.
    """
    qdir = _queue_dir(repo_root)
    items = _read_jsonl(qdir / "pending.jsonl")
    if not items:
        return None
    sorted_items = sorted(
        items,
        key=lambda x: (
            _PRIORITY_RANK.get(x.get("priority", "normal"), 1),
            x.get("created_at", ""),
        ),
    )
    return sorted_items[0]


def claim(item_id: str, claimer: str, repo_root: Path | None = None) -> dict | None:
    """Atomically move the item from pending → claimed.

    Sets ``claimed_at`` and ``claimed_by``.  Returns the claimed item dict,
    or None if the item is not in pending.

    Raises:
        ValueError: if ``claimed.jsonl`` already contains an item (one-at-a-time
                    constraint).
    """
    qdir = _queue_dir(repo_root)
    pending_path = qdir / "pending.jsonl"
    claimed_path = qdir / "claimed.jsonl"

    # --- one-at-a-time constraint ---
    current_claimed = _read_jsonl(claimed_path)
    if current_claimed:
        raise ValueError(
            f"claim: claimed.jsonl already has an item "
            f"({current_claimed[0]['id']}); complete or fail it first."
        )

    # --- find item in pending ---
    pending = _read_jsonl(pending_path)
    target: dict | None = None
    remaining: list[dict] = []
    for it in pending:
        if it["id"] == item_id:
            target = it
        else:
            remaining.append(it)

    if target is None:
        return None

    # --- update fields ---
    target = dict(target)
    target["claimed_at"] = _now_iso()
    target["claimed_by"] = claimer

    # --- atomic: write remaining to pending, write claimed ---
    _write_jsonl(pending_path, remaining)
    _write_jsonl(claimed_path, [target])
    _write_telemetry(qdir, "claim", item_id, target["type"], claimer=claimer)
    return target


def complete(item_id: str, outcome: dict, repo_root: Path | None = None) -> bool:
    """Atomically move the item from claimed → completed/<YYYY-MM-DD>.jsonl.

    Sets ``completed_at`` and ``outcome``.  Returns True if found and moved,
    False if the item was not in claimed.
    """
    qdir = _queue_dir(repo_root)
    claimed_path = qdir / "claimed.jsonl"

    claimed = _read_jsonl(claimed_path)
    target: dict | None = None
    remaining: list[dict] = []
    for it in claimed:
        if it["id"] == item_id:
            target = it
        else:
            remaining.append(it)

    if target is None:
        return False

    target = dict(target)
    target["completed_at"] = _now_iso()
    target["outcome"] = outcome

    completed_path = qdir / "completed" / f"{_today()}.jsonl"
    _write_jsonl(claimed_path, remaining)
    _append_jsonl(completed_path, target)
    _write_telemetry(qdir, "complete", item_id, target["type"])
    return True


def fail(item_id: str, reason: str, repo_root: Path | None = None) -> bool:
    """Atomically move the item from claimed → failed/<YYYY-MM-DD>.jsonl.

    Sets ``completed_at`` (= fail time) and ``outcome={"failed": True, "reason": reason}``.
    Returns True if found and moved, False if the item was not in claimed.
    """
    qdir = _queue_dir(repo_root)
    claimed_path = qdir / "claimed.jsonl"

    claimed = _read_jsonl(claimed_path)
    target: dict | None = None
    remaining: list[dict] = []
    for it in claimed:
        if it["id"] == item_id:
            target = it
        else:
            remaining.append(it)

    if target is None:
        return False

    target = dict(target)
    target["completed_at"] = _now_iso()
    target["outcome"] = {"failed": True, "reason": reason}

    failed_path = qdir / "failed" / f"{_today()}.jsonl"
    _write_jsonl(claimed_path, remaining)
    _append_jsonl(failed_path, target)
    _write_telemetry(qdir, "fail", item_id, target["type"])
    return True


def summary(repo_root: Path | None = None) -> dict:
    """Return queue counts across all states.

    Returns a dict with keys:
        pending, claimed, completed_today, failed_today,
        by_type (type → count across pending+claimed),
        by_priority (priority → count across pending+claimed).
    """
    qdir = _queue_dir(repo_root)
    today = _today()

    pending_items = _read_jsonl(qdir / "pending.jsonl")
    claimed_items = _read_jsonl(qdir / "claimed.jsonl")
    completed_items = _read_jsonl(qdir / "completed" / f"{today}.jsonl")
    failed_items = _read_jsonl(qdir / "failed" / f"{today}.jsonl")

    by_type: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    for it in pending_items + claimed_items:
        t = it.get("type", "unknown")
        p = it.get("priority", "normal")
        by_type[t] = by_type.get(t, 0) + 1
        by_priority[p] = by_priority.get(p, 0) + 1

    return {
        "pending": len(pending_items),
        "claimed": len(claimed_items),
        "completed_today": len(completed_items),
        "failed_today": len(failed_items),
        "by_type": by_type,
        "by_priority": by_priority,
    }


def reclaim_stale(timeout_min: int = 60, repo_root: Path | None = None) -> int:
    """Move claimed items older than ``timeout_min`` minutes back to pending.

    Increments ``payload["reclaim_count"]`` on each reclaimed item so that
    repeated reclaims are visible.  Returns the number of items reclaimed.

    This handles the case where a Director session crashes and leaves an item
    permanently claimed.
    """
    qdir = _queue_dir(repo_root)
    claimed_path = qdir / "claimed.jsonl"
    pending_path = qdir / "pending.jsonl"

    claimed = _read_jsonl(claimed_path)
    if not claimed:
        return 0

    now = datetime.now(timezone.utc)
    stale: list[dict] = []
    still_claimed: list[dict] = []

    for it in claimed:
        claimed_at_str = it.get("claimed_at")
        if claimed_at_str is None:
            still_claimed.append(it)
            continue
        try:
            claimed_dt = datetime.fromisoformat(claimed_at_str.replace("Z", "+00:00"))
        except ValueError:
            still_claimed.append(it)
            continue
        age_min = (now - claimed_dt).total_seconds() / 60.0
        if age_min >= timeout_min:
            stale.append(it)
        else:
            still_claimed.append(it)

    if not stale:
        return 0

    pending = _read_jsonl(pending_path)
    pending_ids = {it["id"] for it in pending}

    reclaimed = 0
    for it in stale:
        it = dict(it)
        it["claimed_at"] = None
        it["claimed_by"] = None
        # Increment reclaim_count in payload
        payload = dict(it.get("payload") or {})
        payload["reclaim_count"] = payload.get("reclaim_count", 0) + 1
        it["payload"] = payload
        if it["id"] not in pending_ids:
            pending.append(it)
            pending_ids.add(it["id"])
        _write_telemetry(qdir, "reclaim", it["id"], it["type"])
        reclaimed += 1

    _write_jsonl(claimed_path, still_claimed)
    _write_jsonl(pending_path, pending)
    return reclaimed
