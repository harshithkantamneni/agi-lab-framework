"""tools/queue_scanner.py — event detector for the autonomous AGI lab work queue.

Stage Q2 of the event-driven work queue (spec: docs/superpowers/specs/2026-05-04-work-queue.md).

Runs every runner iteration and produces queue items from external state.
All detectors are idempotent: re-scanning the same event produces the same ID,
and work_queue.enqueue deduplicates by ID.

Detects 8 event types:
    1. phase_advance          — programs/<active>/phase<N>_close_memo.md present
    2. operator_nudge         — data/user_notes.md mtime changed since last seen
    3. cell_complete          — run_index.json cell with state=completed, rc=0, no errors
    4. cell_failed            — run_index.json cell with rc!=0 or fatal>0 or error>0
    5. diagnostic_review      — new data/diagnostics/*.md files since last scan
    6. consolidator_run       — synthetic: ≥6h since last consolidator run
    7. heartbeat              — synthetic: ≥4h since last Director session log
    8. artifact_queue_projector — missing artifacts from active program's artifact_schema.yaml

State persistence: data/work_queue/last_seen.json tracks high-water marks.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure the repo root (parent of tools/) is always on sys.path so that
# "from tools.work_queue import ..." works when this file is run directly
# (e.g. python tools/queue_scanner.py) rather than via pytest or a runner
# that already has the repo root on sys.path.
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_ON_PATH = str(_TOOLS_DIR.parent)
if _REPO_ROOT_ON_PATH not in sys.path:
    sys.path.insert(0, _REPO_ROOT_ON_PATH)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REPO_ROOT_DEFAULT: Path | None = None  # resolved lazily


def _repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return repo_root
    # Default: two levels up from this file (tools/queue_scanner.py → repo root)
    return Path(__file__).resolve().parent.parent


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _last_seen_path(root: Path) -> Path:
    p = root / "data" / "work_queue" / "last_seen.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_last_seen(root: Path) -> dict:
    p = _last_seen_path(root)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_last_seen(root: Path, data: dict) -> None:
    p = _last_seen_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _active_program(root: Path) -> str | None:
    """Extract active_program name from data/memories/current.md.

    Strips fenced code blocks (```...```) before matching so that example
    values inside code blocks cannot shadow the real prose setting.
    """
    current_md = root / "data" / "memories" / "current.md"
    if not current_md.exists():
        return None
    try:
        text = current_md.read_text(encoding="utf-8")
    except OSError:
        return None

    # Remove fenced code block content (``` ... ```) so example values
    # inside code blocks can never match instead of the real prose line.
    prose = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)

    # Match: active_program: <name> anchored to start-of-line in prose.
    # (?m) enables multiline mode so ^ matches each line start.
    m = re.search(r"(?m)^active_program[:\s]+([a-zA-Z0-9_\-]+)", prose)
    if m:
        return m.group(1).strip("`").strip()
    # Also match the header style: ## Current Program: `program_2_example`
    m2 = re.search(r"Current Program:\s*`([^`]+)`", prose)
    if m2:
        return m2.group(1).strip()
    return None


def _warn(msg: str) -> None:
    print(f"[queue_scanner WARNING] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Detector 1: phase_advance
# ---------------------------------------------------------------------------

_MAX_PHASE = 14


def _detect_phase_advances(repo_root: Path | None = None) -> list[dict]:
    """Scan programs/<active>/phase<N>_close_memo.md and emit phase_advance items.

    Returns a list of item dicts ready for enqueue (not yet enqueued here).
    """
    root = _repo_root(repo_root)
    last_seen = _read_last_seen(root)
    queued: dict[str, bool] = last_seen.get("phase_advances_queued", {})

    program = _active_program(root)
    if not program:
        return []

    prog_dir = root / "programs" / program
    if not prog_dir.is_dir():
        return []

    # Find all phase<N>_close_memo.md files
    phase_nums: list[int] = []
    try:
        for f in prog_dir.glob("phase*_close_memo.md"):
            m = re.search(r"phase(\d+)_close_memo\.md", f.name)
            if m:
                phase_nums.append(int(m.group(1)))
    except OSError as exc:
        _warn(f"phase_advance: error globbing {prog_dir}: {exc}")
        return []

    if not phase_nums:
        return []

    max_n = max(phase_nums)
    next_n = max_n + 1

    # P14 is the final phase — don't advance past it
    if next_n > _MAX_PHASE:
        return []

    # Dedup key: (program, from_phase=PN)
    dedup_key = f"{program}_P{max_n}"
    if queued.get(dedup_key):
        return []

    from_phase = f"P{max_n}"
    to_phase = f"P{next_n}"

    item: dict[str, Any] = {
        "type": "phase_advance",
        "priority": "normal",
        "program": program,
        "payload": {
            "from_phase": from_phase,
            "to_phase": to_phase,
            "context": f"{from_phase} closed; advance to {to_phase}",
        },
        "created_by": "phase_close_scanner",
    }
    return [item]


# ---------------------------------------------------------------------------
# Detector 2: operator_nudge
# ---------------------------------------------------------------------------

_USER_NOTES_CONTEXT_BYTES = 1024


def _detect_operator_nudge(repo_root: Path | None = None) -> list[dict]:
    """Detect mtime changes to data/user_notes.md since last scan.

    Returns a list with 0 or 1 item dicts.
    """
    root = _repo_root(repo_root)
    last_seen = _read_last_seen(root)

    notes_path = root / "data" / "user_notes.md"
    if not notes_path.exists():
        return []

    try:
        mtime = notes_path.stat().st_mtime
    except OSError as exc:
        _warn(f"operator_nudge: stat error on {notes_path}: {exc}")
        return []

    last_mtime = last_seen.get("user_notes_mtime")

    # Same mtime → nothing new
    if last_mtime is not None and abs(mtime - last_mtime) < 0.001:
        return []

    # Read head of file for context
    try:
        raw = notes_path.read_bytes()
        context_text = raw[:_USER_NOTES_CONTEXT_BYTES].decode("utf-8", errors="replace")
    except OSError:
        context_text = ""

    item: dict[str, Any] = {
        "type": "operator_nudge",
        "priority": "urgent",
        "program": None,
        "payload": {
            "mtime": mtime,
            "context": context_text,
        },
        "created_by": "queue_scanner",
    }
    return [item]


# ---------------------------------------------------------------------------
# Detector 3: cell_complete / cell_failed
# ---------------------------------------------------------------------------

_RUN_INDEX_PATH = "data/checkpoints/phase3_factorial/run_index.json"


def _detect_cell_events(repo_root: Path | None = None) -> list[dict]:
    """Read run_index.json and emit cell_complete or cell_failed items.

    Returns a list of item dicts (may be empty).
    """
    root = _repo_root(repo_root)
    last_seen = _read_last_seen(root)
    cells_queued: dict[str, bool] = last_seen.get("cells_queued", {})

    run_index_path = root / _RUN_INDEX_PATH
    if not run_index_path.exists():
        return []

    try:
        run_index = json.loads(run_index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _warn(f"cell_events: error reading {run_index_path}: {exc}")
        return []

    # Active program for tagging items
    program = _active_program(root)

    items: list[dict] = []
    for cell_name, cell_data in run_index.items():
        if cell_name.startswith("_"):
            # Skip meta keys like _meta
            continue
        if not isinstance(cell_data, dict):
            continue

        state = cell_data.get("state", "")
        rc = cell_data.get("rc", 0)
        fatal = cell_data.get("fatal", 0)
        error = cell_data.get("error", 0)

        is_complete = state == "completed" and rc == 0 and fatal == 0 and error == 0
        is_failed = not is_complete and (
            rc != 0 or (fatal is not None and fatal > 0) or (error is not None and error > 0)
        )

        if is_complete:
            dedup_key = f"{cell_name}_complete"
            if cells_queued.get(dedup_key):
                continue
            items.append({
                "type": "cell_complete",
                "priority": "normal",
                "program": program,
                "payload": {
                    "cell": cell_name,
                    "state": state,
                    "rc": rc,
                },
                "created_by": "queue_scanner",
            })
        elif is_failed:
            dedup_key = f"{cell_name}_failed"
            if cells_queued.get(dedup_key):
                continue
            items.append({
                "type": "cell_failed",
                "priority": "urgent",
                "program": program,
                "payload": {
                    "cell": cell_name,
                    "state": state,
                    "rc": rc,
                    "fatal": fatal,
                    "error": error,
                },
                "created_by": "queue_scanner",
            })

    return items


# ---------------------------------------------------------------------------
# Detector 4: diagnostic_review
# ---------------------------------------------------------------------------

def _detect_diagnostic_files(repo_root: Path | None = None) -> list[dict]:
    """Scan data/diagnostics/*.md for files not yet seen.

    Returns a list of item dicts (one per new file).
    """
    root = _repo_root(repo_root)
    last_seen = _read_last_seen(root)
    seen_files: list[str] = last_seen.get("diagnostic_files_seen", [])
    seen_set: set[str] = set(seen_files)

    diag_dir = root / "data" / "diagnostics"
    if not diag_dir.is_dir():
        return []

    program = _active_program(root)

    items: list[dict] = []
    try:
        for f in sorted(diag_dir.glob("*.md")):
            rel = str(f.relative_to(root))
            if rel in seen_set:
                continue
            items.append({
                "type": "diagnostic_review",
                "priority": "normal",
                "program": program,
                "payload": {
                    "file": rel,
                },
                "created_by": "queue_scanner",
            })
    except OSError as exc:
        _warn(f"diagnostic_files: error scanning {diag_dir}: {exc}")
        return []

    return items


# ---------------------------------------------------------------------------
# Detector 5: consolidator_run (synthetic)
# ---------------------------------------------------------------------------

_CONSOLIDATOR_LAST_RUN_PATH = "data/infra/consolidator.last_run"
_CONSOLIDATOR_INTERVAL_HOURS = 6


def _detect_consolidator_run(repo_root: Path) -> list[dict]:
    """Emit a consolidator_run synthetic item when ≥6h since last consolidator run.

    Reads data/infra/consolidator.last_run (format: "<epoch_seconds> <log_dn_count>").
    File missing counts as last_run=0 (enqueue immediately).
    Bucket-deduped per 6-hour window via payload bucket field.
    Returns [] on any error.
    """
    last_run_path = repo_root / _CONSOLIDATOR_LAST_RUN_PATH
    now = time.time()
    last_run: float = 0
    if last_run_path.exists():
        try:
            content = last_run_path.read_text(encoding="utf-8").strip().split()
            last_run = int(content[0]) if content else 0
        except (ValueError, IndexError, OSError):
            last_run = 0

    hours_since = (now - last_run) / 3600
    if hours_since < _CONSOLIDATOR_INTERVAL_HOURS:
        return []

    # Bucket by 6-hour window so re-scanning within the window doesn't dupe
    bucket = int(now // (_CONSOLIDATOR_INTERVAL_HOURS * 3600))
    return [{
        "type": "consolidator_run",
        "priority": "low",
        "program": None,
        "payload": {
            "hours_since_last_run": round(hours_since, 1),
            "bucket": bucket,
            # P-D417 fix (2026-05-20): explicit _dedup_key so compute_id short-circuits
            # on the bucket value. Combined with the work_queue.py:enqueue idempotency
            # check now scanning completed/ history too, this guarantees at most ONE
            # consolidator_run per 6h window — past, present, or future. Prior version
            # relied on (type, program, payload)-hash, which produced different IDs
            # when `hours_since_last_run` rounded slightly differently per scan,
            # causing up to ~4 consolidator_runs/day to accumulate in pending/.
            "_dedup_key": f"consolidator.bucket_{bucket}",
        },
        "created_by": "queue_scanner.consolidator",
    }]


# ---------------------------------------------------------------------------
# Detector 6: heartbeat (synthetic)
# ---------------------------------------------------------------------------

_SESSION_LOGS_PATH = "data/infra/session_logs"
_HEARTBEAT_INTERVAL_HOURS = 4


def _detect_heartbeat(repo_root: Path) -> list[dict]:
    """Emit a heartbeat synthetic item when ≥4h since the newest Director session log.

    Checks mtime of the newest data/infra/session_logs/session_*.log file.
    Returns [] when no logs exist (cold start handled elsewhere).
    Bucket-deduped per 4-hour window via payload bucket field.
    Returns [] on any error.
    """
    log_dir = repo_root / _SESSION_LOGS_PATH
    if not log_dir.exists():
        return []  # cold start handled elsewhere

    now = time.time()
    try:
        sessions = list(log_dir.glob("session_*.log"))
    except OSError:
        return []

    if not sessions:
        return []

    try:
        newest_mt = max(s.stat().st_mtime for s in sessions)
    except OSError:
        return []

    hours_since = (now - newest_mt) / 3600
    if hours_since < _HEARTBEAT_INTERVAL_HOURS:
        return []

    bucket = int(now // (_HEARTBEAT_INTERVAL_HOURS * 3600))
    return [{
        "type": "heartbeat",
        "priority": "low",
        "program": None,
        "payload": {
            "hours_since_last_session": round(hours_since, 1),
            "bucket": bucket,
        },
        "created_by": "queue_scanner.heartbeat",
    }]


# ---------------------------------------------------------------------------
# Detector 7: orphan_deliverable
# ---------------------------------------------------------------------------

# Claim is considered orphaned only after this age — protects active sessions.
_ORPHAN_CLAIM_MIN_AGE_SECONDS = 30 * 60  # 30 minutes
# Deliverable file must be at least this size to count (avoids empty placeholders).
_ORPHAN_DELIVERABLE_MIN_SIZE = 100


def _detect_orphan_deliverable(repo_root: Path | None = None) -> list[dict]:
    """Detect orphan claims with on-disk deliverables but no completion event.

    Catches the case where post_director.py *also* failed to run (e.g., runner
    crashed mid-iteration). For each claim older than 30 min where the
    expected_deliverable_pattern (per work_queue_handlers.md) glob matches a
    file with mtime > claimed_at and no ``complete``/``fail`` event in
    queue_telemetry.jsonl, emit a session_recovery_review queue item.

    Idempotent: deterministic IDs derived from (orphan_item_id, "orphan_recovery")
    via compute_id() in scan().
    """
    root = _repo_root(repo_root)
    claimed_path = root / "data" / "work_queue" / "claimed.jsonl"
    if not claimed_path.exists():
        return []

    # Don't fire if claimed.jsonl has been touched recently (covers active sessions
    # that claimed an item less than the threshold ago).
    file_age = time.time() - claimed_path.stat().st_mtime
    if file_age < _ORPHAN_CLAIM_MIN_AGE_SECONDS:
        return []

    # Read claims, completion events, and handler schema
    claims: list[dict] = []
    for line in claimed_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            claims.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not claims:
        return []

    completed_ids = _read_completed_event_ids(root)
    handlers = _load_handlers_for_orphan(root)

    items: list[dict] = []
    for claim in claims:
        if claim.get("id") in completed_ids:
            continue
        schema = handlers.get(claim.get("type"), {})
        pattern = schema.get("expected_deliverable_pattern")
        if not pattern:
            continue
        # Render template against payload
        from tools.handler_schema import render_template  # noqa: PLC0415
        rendered = render_template(pattern, claim.get("payload", {}))
        if "{" in rendered and "}" in rendered:
            continue  # unbound template — refuse to glob

        # Glob and filter to fresh + sized matches
        try:
            claim_ts = _parse_iso_for_orphan(claim.get("claimed_at", ""))
        except ValueError:
            continue

        try:
            matches = list(root.glob(rendered))
        except Exception as exc:  # noqa: BLE001
            _warn(f"orphan_deliverable: glob error for pattern '{rendered}': {exc}")
            continue

        fresh = [
            m for m in matches
            if m.is_file()
            and m.stat().st_mtime > claim_ts
            and m.stat().st_size > _ORPHAN_DELIVERABLE_MIN_SIZE
        ]
        if not fresh:
            continue
        fresh.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        match = fresh[0]
        items.append({
            "type": "session_recovery_review",
            "priority": "normal",
            "program": claim.get("program", ""),
            "created_by": "queue_scanner_orphan_detector",
            "payload": {
                "orphan_item_id": claim["id"],
                "deliverable": str(match.relative_to(root)),
                "claimer": claim.get("claimed_by", "unknown"),
                "claimed_at": claim.get("claimed_at"),
                "_dedup_key": "orphan_recovery",  # keeps compute_id stable across scans
            },
        })

    return items


def _read_completed_event_ids(root: Path) -> set[str]:
    """Read item IDs that have a complete or fail event in queue_telemetry.jsonl."""
    telem = root / "data" / "work_queue" / "queue_telemetry.jsonl"
    out: set[str] = set()
    if not telem.exists():
        return out
    for line in telem.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("action") in ("complete", "fail") and rec.get("item_id"):
            out.add(rec["item_id"])
    return out


def _load_handlers_for_orphan(root: Path) -> dict:
    """Load work_queue_handlers.md schema; return {} on any error."""
    handlers_path = root / "data" / "agents" / "_shared" / "work_queue_handlers.md"
    if not handlers_path.exists():
        return {}
    try:
        from tools.handler_schema import load_schema  # noqa: PLC0415
        return load_schema(handlers_path)
    except Exception as exc:  # noqa: BLE001
        _warn(f"orphan_deliverable: handler schema load failed: {exc}")
        return {}


def _parse_iso_for_orphan(s: str) -> float:
    """Parse ISO 8601 (with optional Z suffix) to unix epoch."""
    if not s:
        raise ValueError("empty timestamp")
    s2 = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s2).timestamp()


# ---------------------------------------------------------------------------
# Public scan() API
# ---------------------------------------------------------------------------

def scan(repo_root: Path | None = None) -> list[str]:
    """Run all detectors and enqueue any new items.

    Returns list of newly-enqueued item IDs (empty if no new events).
    Updates data/work_queue/last_seen.json with high-water marks.
    Idempotent: running twice in a row produces 0 new IDs the second time.
    """
    root = _repo_root(repo_root)

    # Lazy import: keep module-level importable without work_queue present.
    from tools.work_queue import compute_id, enqueue  # noqa: PLC0415

    # Run detectors; each returns [] on error (non-fatal)
    phase_items: list[dict] = []
    nudge_items: list[dict] = []
    cell_items: list[dict] = []
    diag_items: list[dict] = []
    consolidator_items: list[dict] = []
    heartbeat_items: list[dict] = []
    orphan_items: list[dict] = []
    artifact_items: list[dict] = []

    try:
        phase_items = _detect_phase_advances(repo_root)
    except Exception as exc:  # noqa: BLE001
        _warn(f"phase_advance detector failed: {exc}")

    try:
        nudge_items = _detect_operator_nudge(repo_root)
    except Exception as exc:  # noqa: BLE001
        _warn(f"operator_nudge detector failed: {exc}")

    try:
        cell_items = _detect_cell_events(repo_root)
    except Exception as exc:  # noqa: BLE001
        _warn(f"cell_events detector failed: {exc}")

    try:
        diag_items = _detect_diagnostic_files(repo_root)
    except Exception as exc:  # noqa: BLE001
        _warn(f"diagnostic_files detector failed: {exc}")

    try:
        consolidator_items = _detect_consolidator_run(root)
    except Exception as exc:  # noqa: BLE001
        _warn(f"consolidator_run detector failed: {exc}")

    try:
        heartbeat_items = _detect_heartbeat(root)
    except Exception as exc:  # noqa: BLE001
        _warn(f"heartbeat detector failed: {exc}")

    try:
        orphan_items = _detect_orphan_deliverable(repo_root)
    except Exception as exc:  # noqa: BLE001
        _warn(f"orphan_deliverable detector failed: {exc}")

    try:
        from tools.artifact_queue_projector import project_artifacts  # noqa: PLC0415
        active_program = _active_program(root)
        if active_program:
            artifact_items = project_artifacts(active_program, repo_root=root)
    except Exception as exc:  # noqa: BLE001
        _warn(f"artifact_queue_projector detector failed: {exc}")

    all_items = phase_items + nudge_items + cell_items + diag_items + consolidator_items + heartbeat_items + orphan_items + artifact_items

    # Read last_seen once for mutation
    last_seen = _read_last_seen(root)
    if "phase_advances_queued" not in last_seen:
        last_seen["phase_advances_queued"] = {}
    if "cells_queued" not in last_seen:
        last_seen["cells_queued"] = {}
    if "diagnostic_files_seen" not in last_seen:
        last_seen["diagnostic_files_seen"] = []

    newly_enqueued: list[str] = []

    # Snapshot the current queue IDs once before the loop so we can distinguish
    # truly-new enqueues from already-present items (enqueue() is idempotent
    # and returns the same ID either way).
    qdir = root / "data" / "work_queue"
    _pending_path = qdir / "pending.jsonl"
    _claimed_path = qdir / "claimed.jsonl"

    def _read_existing_ids() -> set[str]:
        """Return IDs already in pending or claimed files."""
        ids: set[str] = set()
        for path in (_pending_path, _claimed_path):
            if path.exists():
                try:
                    import json as _json  # noqa: PLC0415
                    for line in path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line:
                            obj = _json.loads(line)
                            if "id" in obj:
                                ids.add(obj["id"])
                except (OSError, ValueError):
                    pass
        return ids

    existing_ids_before = _read_existing_ids()

    for item in all_items:
        # Pre-compute the deterministic ID so we can check if it was already in queue
        item_id = compute_id(item["type"], item.get("program"), item.get("payload", {}))

        # Skip items already present in the queue (synthetic detectors always fire
        # when condition is met; dedup is here rather than in the detector).
        if item_id in existing_ids_before:
            continue

        try:
            returned_id = enqueue(item, repo_root=root)
        except Exception as exc:  # noqa: BLE001
            _warn(f"enqueue failed for {item['type']}: {exc}")
            continue

        # Track in last_seen and collect truly new IDs
        if item["type"] == "phase_advance":
            program = item.get("program") or ""
            from_phase = item["payload"].get("from_phase", "")
            dedup_key = f"{program}_{from_phase}"
            last_seen["phase_advances_queued"][dedup_key] = True
            newly_enqueued.append(returned_id)

        elif item["type"] == "operator_nudge":
            mtime = item["payload"].get("mtime")
            if mtime is not None:
                last_seen["user_notes_mtime"] = mtime
            newly_enqueued.append(returned_id)

        elif item["type"] in ("cell_complete", "cell_failed"):
            cell_name = item["payload"].get("cell", "")
            suffix = "complete" if item["type"] == "cell_complete" else "failed"
            dedup_key = f"{cell_name}_{suffix}"
            last_seen["cells_queued"][dedup_key] = True
            newly_enqueued.append(returned_id)

        elif item["type"] == "diagnostic_review":
            file_rel = item["payload"].get("file", "")
            if file_rel and file_rel not in last_seen["diagnostic_files_seen"]:
                last_seen["diagnostic_files_seen"].append(file_rel)
            newly_enqueued.append(returned_id)

        elif item["type"] in ("consolidator_run", "heartbeat", "session_recovery_review"):
            # Idempotency is handled purely via compute_id + bucket in payload
            # plus the existing_ids_before pre-check above.
            newly_enqueued.append(returned_id)

        elif item.get("created_by") == "artifact_queue_projector":
            # Artifact items: idempotency via compute_id(_artifact_id in payload).
            newly_enqueued.append(returned_id)

    # Always update last_scan_at
    last_seen["last_scan_at"] = _now_iso()
    _write_last_seen(root, last_seen)

    return newly_enqueued


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry: run scan, print summary."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="queue_scanner: detect events and enqueue work items"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override repo root path (default: auto-detected from script location)",
    )
    args = parser.parse_args()

    root = args.repo_root
    ids = scan(repo_root=root)

    if ids:
        print(f"queue_scanner: {len(ids)} new item(s) enqueued")
        for item_id in ids:
            print(f"  {item_id}")
    else:
        print("queue_scanner: 0 new items (queue up to date)")


if __name__ == "__main__":
    main()
