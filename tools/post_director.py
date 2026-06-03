"""Runner-side close-out finalizer.

Runs unconditionally after every Director exit. Reads data/session_exit.md and
applies log/current/queue mutations per the structured JSON block. Falls back
to silent-death recovery when the file is missing or partial.

This module is built incrementally across Stages R3-R4 of the runner-owned
close-out spec (docs/superpowers/specs/2026-05-05-runner-owns-closeout.md).
Task 6 lays the scaffolding; subsequent tasks fill in:
- Task 7: read_session_exit() — JSON block extraction
- Task 8: finalize_from_structured() success branch
- Task 9: failure + no_op branches; idempotency
- Task 10: main() dispatch + telemetry
- Tasks 11-12: silent-death recovery
"""
from __future__ import annotations
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path as _PathForBootstrap

# Ensure the repo root (parent of tools/) is on sys.path so direct invocation
# (`python3 tools/post_director.py`) resolves `import tools.work_queue` and
# `import tools.handler_schema`. Without this, the runner's invocation fails
# with ModuleNotFoundError and post_director silently produces no mutations.
# Mirrors the pattern in tools/queue_scanner.py and tools/training_digest.py.
_TOOLS_DIR = _PathForBootstrap(__file__).resolve().parent
_REPO_ROOT_ON_PATH = str(_TOOLS_DIR.parent)
if _REPO_ROOT_ON_PATH not in sys.path:
    sys.path.insert(0, _REPO_ROOT_ON_PATH)
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def read_session_exit() -> dict | None:
    """Read data/session_exit.md and return the parsed JSON block.

    Return values:
    - ``None`` if the file does not exist.
    - A dict from the JSON code block if one is present and parses cleanly.
    - ``{"_legacy": True, "_raw": <full text>}`` if the file exists but has no
      JSON code block (backward-compat path for sessions written before the
      structured-channel migration).
    - ``{"_legacy": True, "_raw": <full text>, "_invalid_json": True}`` if a
      JSON block is present but malformed.
    """
    path = REPO / "data/session_exit.md"
    if not path.exists():
        return None
    text = path.read_text()
    m = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if not m:
        return {"_legacy": True, "_raw": text}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {"_legacy": True, "_raw": text, "_invalid_json": True}


def _validate_session_exit_schema(se: dict) -> tuple[bool, str | None]:
    """Returns (is_valid, violation_reason). reason is None when valid."""
    status = se.get("status")
    if status not in ("success", "partial", "failure", "no_op"):
        return False, f"invalid_status_{status!r}"

    # next_action requirement: success/partial MUST have next_action OR program_complete
    if status in ("success", "partial"):
        has_next_action = se.get("next_action") is not None
        program_complete = se.get("program_complete") is True
        if not has_next_action and not program_complete:
            return False, "next_action_null_without_program_complete"

    return True, None


def _check_contract(se: dict, role: str = "director") -> tuple[bool, str | None]:
    """Validate session against agent contract.

    Reads data/agents/_shared/agent_contracts.json. If contracts file missing
    or role not defined: returns (True, None) for backward compat.

    Each contract criterion has a `check` type:
    - file_exists: Path(criterion['path']) must exist relative to REPO
    - session_exit_schema_v1.1: delegates to _validate_session_exit_schema(se)
    - next_action_populated_or_program_complete_true: enforces the v1.1 rule
      directly (independent of the schema check, so contract is a strict
      superset)

    Returns (is_valid, violation_reason). reason includes the criterion id
    that failed for operator debugging.
    """
    contracts_path = REPO / "data" / "agents" / "_shared" / "agent_contracts.json"
    if not contracts_path.exists():
        return True, None
    try:
        contracts = json.loads(contracts_path.read_text()).get("contracts", {})
    except json.JSONDecodeError:
        return True, None  # backward compat: malformed contracts file = no enforcement

    contract = contracts.get(role)
    if not contract:
        return True, None  # no contract for this role

    for criterion in contract.get("success_criteria", []) or []:
        cid = criterion.get("id", "unknown_criterion")
        check_type = criterion.get("check")
        if check_type == "file_exists":
            path = REPO / criterion.get("path", "")
            if not path.exists():
                return False, f"contract criterion {cid!r}: file does not exist: {criterion.get('path')}"
        elif check_type == "session_exit_schema_v1.1":
            ok, reason = _validate_session_exit_schema(se)
            if not ok:
                return False, f"contract criterion {cid!r}: {reason}"
        elif check_type == "next_action_populated_or_program_complete_true":
            if se.get("status") in ("success", "partial"):
                has_next_action = se.get("next_action") is not None
                program_complete = se.get("program_complete") is True
                if not has_next_action and not program_complete:
                    return False, f"contract criterion {cid!r}: next_action null without program_complete"
        # Other check types are unknown to v1 — skip silently (forward compat)

    return True, None


# Redispatch ceiling tracking (L3.3 / D′)

_REDISPATCH_COUNT_PATH = "data/infra/director_redispatch_count.json"
_OPERATOR_REVIEW_PATH = "data/operator_review_pending.md"


def _read_redispatch_count() -> dict:
    """Read the redispatch counter. Returns {current_session_id, count}. Empty on missing/corrupt."""
    path = REPO / _REDISPATCH_COUNT_PATH
    if not path.exists():
        return {"current_session_id": None, "count": 0}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"current_session_id": None, "count": 0}


def _write_redispatch_count(session_id: str, count: int) -> None:
    """Atomically write the redispatch counter."""
    path = REPO / _REDISPATCH_COUNT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"current_session_id": session_id, "count": count}))
    tmp.replace(path)


def _max_redispatches() -> int:
    """Read the max from contracts file. Default 3 if missing."""
    contracts_path = REPO / "data/agents/_shared/agent_contracts.json"
    if not contracts_path.exists():
        return 3
    try:
        contracts = json.loads(contracts_path.read_text()).get("contracts", {})
        return contracts.get("director", {}).get("max_redispatches_per_session", 3)
    except json.JSONDecodeError:
        return 3


def _write_operator_review(session_id: str, count: int, last_violation_reason: str) -> None:
    """Write a markdown file flagging operator review for a looping Director session."""
    path = REPO / _OPERATOR_REVIEW_PATH
    timestamp = datetime.now(timezone.utc).isoformat()
    body = (
        f"# Operator Review Pending — Director Redispatch Ceiling Hit\n\n"
        f"**Created:** {timestamp}\n"
        f"**Session ID:** {session_id}\n"
        f"**Redispatch count:** {count} (exceeds max_redispatches_per_session)\n"
        f"**Last violation reason:** {last_violation_reason}\n\n"
        f"## Why this is here\n\n"
        f"Director session `{session_id}` produced {count} consecutive session_exit "
        f"violations (schema or contract). Per `data/agents/_shared/agent_contracts.json`, "
        f"the redispatch ceiling is configured at the max — further automatic redispatch "
        f"would loop. The runner has STOPPED redispatching this session and is awaiting "
        f"operator review.\n\n"
        f"## Suggested actions\n\n"
        f"1. Read `data/infra/post_director_telemetry.jsonl` (last ~10 records) to see "
        f"what Director kept getting wrong\n"
        f"2. Read `data/session_exit.md` (if present) to see Director's last attempted exit\n"
        f"3. Read `data/memories/log.md` and `data/memories/current.md` to understand state\n"
        f"4. If Director was correct (e.g., program genuinely complete), manually:\n"
        f"   - Set the right session_exit.md content (or just leave the lab idle)\n"
        f"   - Delete this file: `rm {_OPERATOR_REVIEW_PATH}`\n"
        f"   - Delete the counter file: `rm {_REDISPATCH_COUNT_PATH}`\n"
        f"5. If Director was wrong (genuine confusion), update its procedural or "
        f"context_brief, then clear the counter and let it retry.\n\n"
        f"## Recent telemetry\n\n"
        f"```\n"
    )
    # Append last 5 telemetry records for context
    telem_path = REPO / "data/infra/post_director_telemetry.jsonl"
    if telem_path.exists():
        lines = telem_path.read_text().strip().split("\n")
        for line in lines[-5:]:
            body += line + "\n"
    body += "```\n"
    path.write_text(body)


def _attempt_macos_notification(title: str, message: str) -> None:
    """Best-effort macOS notification (silent failure on non-macOS or osascript missing)."""
    try:
        import subprocess
        subprocess.run(
            ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
            timeout=5,
            capture_output=True,
        )
    except Exception:
        pass


def finalize_from_structured(se: dict) -> dict | None:
    """Apply mutations from a parsed session_exit dict.

    Returns None on success. Returns a violation marker dict
    {violation: True, reason: str, violation_type: "schema"|"contract"} when
    the session_exit fails validation — no mutations applied in that case.

    Status branches:
    - ``success`` / ``partial``: append log, apply current patches, complete
      the claim, enqueue ``next_action`` if non-null. ``partial`` suppresses
      the enqueue (Director flagged some work incomplete).
    - ``failure``: append log, fail the claim, enqueue diagnostic_review.
    - ``no_op``: append log only if non-empty. No queue mutations.

    Idempotent: re-running over the same session_exit dict is a no-op.
    """
    # v1.1 schema validation -- MUST pass before any mutations
    ok, reason = _validate_session_exit_schema(se)
    if not ok:
        return {"violation": True, "reason": reason, "violation_type": "schema"}

    # Agent contract check (D') -- fires after schema check
    ok, reason = _check_contract(se, role="director")
    if not ok:
        return {"violation": True, "reason": reason, "violation_type": "contract"}

    import tools.work_queue as wq

    status = se.get("status")
    sid = se.get("session_id", "")

    if status in ("success", "partial"):
        if se.get("log_entry_text"):
            _append_log_idempotent(se["log_entry_text"], sid)
        for patch in (se.get("current_md_patches") or []):
            _apply_current_patch(patch)
        if se.get("claimed_item_id") and not _already_completed(se["claimed_item_id"]):
            wq.complete(se["claimed_item_id"], outcome={"status": status, "session_id": sid})
        if status == "success" and se.get("next_action"):
            wq.enqueue(se["next_action"])
    elif status == "failure":
        if se.get("log_entry_text"):
            _append_log_idempotent(se["log_entry_text"], sid)
        if se.get("claimed_item_id") and not _already_failed_or_completed(se["claimed_item_id"]):
            wq.fail(se["claimed_item_id"], reason=se.get("reason", "director-reported-failure"))
        if not _diagnostic_already_enqueued(sid):
            wq.enqueue({
                "type": "diagnostic_review",
                "priority": "urgent",
                "program": se.get("program", ""),
                "created_by": f"post_director_{sid}",
                "payload": {
                    "failed_item_id": se.get("claimed_item_id"),
                    "reason": se.get("reason", "unknown"),
                    "notes": se.get("notes", ""),
                    "session_id": sid,
                },
            })
    elif status == "no_op":
        if se.get("log_entry_text"):
            _append_log_idempotent(se["log_entry_text"], sid)


def _append_log_idempotent(text: str, sid: str) -> None:
    """Append `text` to data/memories/log.md after the preamble, unless `sid`
    is already present in the file (already-appended case)."""
    log = REPO / "data/memories/log.md"
    body = log.read_text()
    if sid and sid in body:
        return
    # Insert after the preamble (first `---\n\n`) so newest entry sits at top
    parts = body.split("\n---\n\n", 1)
    if len(parts) == 2:
        new_body = parts[0] + "\n---\n\n" + text.rstrip() + "\n\n---\n\n" + parts[1]
    else:
        new_body = body.rstrip() + "\n\n" + text.rstrip() + "\n"
    log.write_text(new_body)


def _apply_current_patch(patch: dict) -> bool:
    """Apply a single str_replace patch to data/memories/current.md.

    Returns True if applied, False if skipped (either old-not-found-and-new-already-present,
    or both missing — the latter logged for operator review).
    """
    cur = REPO / "data/memories/current.md"
    body = cur.read_text()
    old = patch.get("old", "")
    new = patch.get("new", "")
    if old and old in body:
        cur.write_text(body.replace(old, new, 1))
        return True
    if new and new in body:
        return False  # already-applied
    return False  # not found and not applied — caller may log


def _already_completed(item_id: str) -> bool:
    """Check if item_id is already in a completed/<date>.jsonl file."""
    completed_dir = REPO / "data/work_queue/completed"
    if not completed_dir.exists():
        return False
    for f in completed_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("id") == item_id:
                    return True
            except json.JSONDecodeError:
                continue
    return False


def _already_failed_or_completed(item_id: str) -> bool:
    """Check if item_id is in completed or failed JSONL files."""
    if _already_completed(item_id):
        return True
    failed_dir = REPO / "data/work_queue/failed"
    if not failed_dir.exists():
        return False
    for f in failed_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("id") == item_id:
                    return True
            except json.JSONDecodeError:
                continue
    return False


def _diagnostic_already_enqueued(sid: str) -> bool:
    """Check if a diagnostic_review item for this session is already pending or processed."""
    pending = REPO / "data/work_queue/pending.jsonl"
    if pending.exists():
        for line in pending.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if (rec.get("type") == "diagnostic_review"
                        and rec.get("payload", {}).get("session_id") == sid):
                    return True
            except json.JSONDecodeError:
                continue
    # Also check completed/failed in case the diagnostic was already processed
    for sub in ("completed", "failed"):
        d = REPO / f"data/work_queue/{sub}"
        if not d.exists():
            continue
        for f in d.glob("*.jsonl"):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if (rec.get("type") == "diagnostic_review"
                            and rec.get("payload", {}).get("session_id") == sid):
                        return True
                except json.JSONDecodeError:
                    continue
    return False


def main() -> int:
    """Entry point. Reads session_exit, dispatches to appropriate finalizer,
    writes telemetry record. Always returns 0 (errors are logged in telemetry,
    not raised — runner should not crash because post_director crashed)."""
    se = read_session_exit()
    branch = "unknown"
    extra: dict = {}
    sid = None
    try:
        if se is None:
            sid = None
            extra, branch = _silent_death_branch()
        elif se.get("_legacy"):
            sid = None
            branch = "legacy_finalize"
            if se.get("_invalid_json"):
                extra["invalid_json"] = True
            legacy_finalize(se)
        else:
            sid = se.get("session_id")
            status = se.get("status", "")
            result = finalize_from_structured(se)
            # Detect violation marker returned by finalize_from_structured
            if isinstance(result, dict) and result.get("violation"):
                violation_type = result.get("violation_type", "unknown")
                reason = result.get("reason", "unknown")
                extra["violation_reason"] = reason
                extra["violation_type"] = violation_type

                # Update redispatch counter
                counter = _read_redispatch_count()
                if counter.get("current_session_id") == sid:
                    new_count = counter["count"] + 1
                else:
                    new_count = 1
                _write_redispatch_count(sid, new_count)
                extra["redispatch_count"] = new_count

                max_redispatches = _max_redispatches()
                if new_count >= max_redispatches:
                    # ESCALATE — stop redispatching, notify operator
                    branch = "max_redispatches_escalated"
                    _write_operator_review(sid, new_count, reason)
                    _attempt_macos_notification(
                        "AGI Lab: Director Looping",
                        f"Director {sid} hit {new_count} redispatches — operator review needed",
                    )
                    # Remove any existing redispatch flag so runner does NOT redispatch
                    (REPO / "data" / "session_exit_redispatch_pending.flag").unlink(missing_ok=True)
                else:
                    branch = f"{violation_type}_violation"
                    # Touch redispatch flag so runner (L1.3) can respawn Director
                    flag = REPO / "data" / "session_exit_redispatch_pending.flag"
                    flag.write_text(
                        f"violation_type={violation_type}\n"
                        f"reason={reason}\n"
                        f"session_id={sid}\n"
                    )
            else:
                branch = f"structured_{status}"
    except Exception as e:
        # Never crash the runner; record the error and continue.
        extra["error"] = str(e)
        extra["error_type"] = type(e).__name__
    finally:
        _log_telemetry({
            "ts": datetime.now(timezone.utc).isoformat(),
            "branch_taken": branch,
            "session_id": sid,
            **extra,
        })

    _fire_cost_rollup()  # non-fatal cost rollup refresh — fires on every Director exit
    return 0


def _silent_death_branch() -> tuple[dict, str]:
    """Decide silent_death sub-branch based on whether there's an active claim."""
    claimed_path = REPO / "data/work_queue/claimed.jsonl"
    if not claimed_path.exists() or not claimed_path.read_text().strip():
        return ({}, "silent_death_no_claim")
    # Real recovery happens in Task 11 (recover_from_silent_death)
    recover_from_silent_death()
    return ({}, "silent_death")


def legacy_finalize(se: dict) -> None:
    """No-op for legacy session_exit.md (no JSON block).

    The runner's existing code (line 790, run_agi_lab.sh) already parses the
    `reason:` line for backward-compat. This function is a placeholder that
    lets main() route legacy files cleanly; telemetry records that the legacy
    path was taken so we can monitor migration progress.
    """
    return


def recover_from_silent_death() -> None:
    """Walk data/work_queue/claimed.jsonl. For each claim, look up the
    handler's expected_deliverable_pattern, glob it against the repo, and:
    - If a matching deliverable exists with mtime > claim.claimed_at and
      non-trivial size: call _silent_death_recover().
    - Otherwise (or if handler opts out via null pattern): call
      _silent_death_fail().
    """
    import tools.work_queue as wq
    import tools.handler_schema as hs

    claimed = _read_claimed_items()
    if not claimed:
        return

    handlers_path = REPO / "data/agents/_shared/work_queue_handlers.md"
    handlers = hs.load_schema(handlers_path) if handlers_path.exists() else {}

    for claim in claimed:
        schema = handlers.get(claim.get("type"), {})
        pattern = schema.get("expected_deliverable_pattern")
        if not pattern:
            _silent_death_fail(claim, reason="silent_death_handler_opt_out")
            continue

        rendered = hs.render_template(pattern, claim.get("payload", {}))
        # Refuse to glob unrendered patterns (would match too broadly)
        if "{" in rendered and "}" in rendered:
            _silent_death_fail(claim, reason="silent_death_unrendered_pattern")
            continue

        try:
            claim_ts = _parse_iso(claim.get("claimed_at", ""))
        except ValueError:
            _silent_death_fail(claim, reason="silent_death_bad_claim_timestamp")
            continue

        matches = list(REPO.glob(rendered))
        fresh = [
            m for m in matches
            if m.is_file() and m.stat().st_mtime > claim_ts and m.stat().st_size > 100
        ]

        if fresh:
            # Pick the most recent match
            fresh.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            _silent_death_recover(claim, fresh[0], schema)
        else:
            _silent_death_fail(claim, reason="silent_death_no_deliverable")


def _read_claimed_items() -> list[dict]:
    """Parse all records from data/work_queue/claimed.jsonl. Empty if missing."""
    path = REPO / "data/work_queue/claimed.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _parse_iso(s: str) -> float:
    """Parse an ISO 8601 timestamp (with optional trailing Z) to unix epoch seconds."""
    if not s:
        raise ValueError("empty timestamp")
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).timestamp()


def _silent_death_recover(claim: dict, deliverable: Path, schema: dict) -> None:
    """Auto-recover a claim where a deliverable was found on disk."""
    import tools.work_queue as wq
    import tools.handler_schema as hs

    claim_id = claim["id"]
    sid_synthetic = f"D-RECOVERY-{claim_id}"
    rel_path = deliverable.relative_to(REPO)
    log_text = (
        f"{sid_synthetic} ({datetime.now(timezone.utc).isoformat()}, "
        f"**runner-auto-recovered**): **Director silent-died after deliverable; "
        f"auto-recovery performed.** Claim `{claim_id}` (type={claim['type']}); "
        f"deliverable `{rel_path}` ({deliverable.stat().st_size} bytes) on disk; "
        f"`work_queue.complete()` + follow-on enqueue + `session_recovery_review` "
        f"enqueued. STATUS=auto_recovered.\n"
    )
    _append_log_idempotent(log_text, sid_synthetic)

    if not _already_completed(claim_id):
        wq.complete(claim_id, outcome={
            "status": "success_recovered", "synthetic": True,
            "deliverable": str(rel_path),
        })

    # Enqueue follow-on per next_action_template (if defined)
    next_template = schema.get("next_action_template")
    if next_template:
        next_item = hs.render_template(next_template, claim.get("payload", {}))
        next_item.setdefault("program", claim.get("program", ""))
        next_item.setdefault("created_by", "post_director_silent_death_recovery")
        wq.enqueue(next_item)

    # Always enqueue a session_recovery_review for operator inspection
    wq.enqueue({
        "type": "session_recovery_review",
        "priority": "normal",
        "program": claim.get("program", ""),
        "created_by": "post_director_silent_death_recovery",
        "payload": {
            "orphan_item_id": claim_id,
            "deliverable": str(rel_path),
            "claimer": claim.get("claimed_by", "unknown"),
            "synthetic_session_id": sid_synthetic,
        },
    })


def _silent_death_fail(claim: dict, reason: str) -> None:
    """Fail a claim where no recoverable deliverable was found."""
    import tools.work_queue as wq

    claim_id = claim["id"]
    sid_synthetic = f"D-FAILED-{claim_id}"
    log_text = (
        f"{sid_synthetic} ({datetime.now(timezone.utc).isoformat()}, "
        f"**runner-detected**): **Director silent-died without deliverable.** "
        f"Claim `{claim_id}` (type={claim['type']}) failed (reason={reason}); "
        f"`diagnostic_review` enqueued. STATUS=silent_death_fail.\n"
    )
    _append_log_idempotent(log_text, sid_synthetic)

    if not _already_failed_or_completed(claim_id):
        wq.fail(claim_id, reason=reason)
    wq.enqueue({
        "type": "diagnostic_review",
        "priority": "urgent",
        "program": claim.get("program", ""),
        "created_by": "post_director_silent_death_fail",
        "payload": {
            "failed_item_id": claim_id,
            "reason": reason,
            "claimer": claim.get("claimed_by", "unknown"),
            "synthetic_session_id": sid_synthetic,
        },
    })


def _fire_cost_rollup() -> None:
    """Refresh the weekly cost rollup as a non-fatal subprocess.

    Runs after every post_director invocation so cost_rollup.md /
    cost_rollup.json stay fresh on every Director exit. Errors are
    swallowed — the rollup is best-effort and must not crash the runner.
    """
    import subprocess
    rollup_script = REPO / "tools" / "cost_rollup.py"
    if not rollup_script.exists():
        return
    try:
        subprocess.run(
            ["python3", str(rollup_script)],
            cwd=str(REPO),
            capture_output=True,
            check=False,
            timeout=30,
        )
    except Exception:
        # Silently swallow — cost rollup is best-effort.
        pass


def _log_telemetry(rec: dict) -> None:
    """Append a single telemetry record (one JSON object per line) to
    data/infra/post_director_telemetry.jsonl. Creates the file/dir if absent."""
    telem_path = REPO / "data/infra/post_director_telemetry.jsonl"
    telem_path.parent.mkdir(parents=True, exist_ok=True)
    with telem_path.open("a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    sys.exit(main())
