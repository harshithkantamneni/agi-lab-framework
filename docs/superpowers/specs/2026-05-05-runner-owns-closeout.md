# Runner-Owned Close-Out (RO-CO)

**Date:** 2026-05-05
**Status:** Drafted
**Triggers:** D-313 silent-death post-deliverable (2026-05-05); D-309 partial close-out (different shape, same root cause). Spec exists because manual recovery is structurally fragile.

## Problem

Director session can die mid-flow — context exhaust, premature stop-token, model timeout, or returning early — between subagent return and house-keeping. The lab strands silently because:

1. **`skip-when-stable` mistakes silence for stability.** No file changes after Director exit ⇒ "all is well" ⇒ runner skips next iteration.
2. **`reclaim_stale()` only fires on the dispatch path.** When the runner is in skip mode, stale claims age past 60 min unchecked.
3. **Deliverable orphans are invisible.** A 43.5 KB report on disk with no completion event in queue telemetry is, structurally, indistinguishable from "no work happened."
4. **Close-out is convention, not enforcement.** Director's procedural describes the close-out checklist; nothing verifies it ran.

D-313 exhibited the worst case: `mechanism_extractor` returned a complete `phase10_mechanism_report.md` (43.5 KB, §1-§8 with calibration block), but Director skipped log entry, current.md update, `work_queue.complete()`, follow-on enqueue, and episodic. Lab stranded for ~2.7h until operator intervention.

## Goal

**Move close-out from Director (LLM, unreliable) to runner (shell, deterministic).** Director becomes a deliverable-producer + structured-intent-writer. The runner — which cannot crash mid-flow — owns log/current/queue invariants based on Director's structured exit file, with a fallback path that recovers from a missing or partial exit file by walking the work-queue claim and inspecting the program directory for the expected deliverable.

## Why this is the structurally right answer

The current architecture treats Director as the sole authority on session outcome. Any failure mode (context exhaust, early return, crash, hang) breaks every downstream invariant. A runner-owned model treats LLM execution as inherently lossy and pushes invariant maintenance to a deterministic shell + Python layer.

This is the same controller/worker separation that the work-queue spec adopts: queue items are data, not heuristic. Close-out is now also data — an explicit JSON block in `session_exit.md` — with a deterministic finalizer (`tools/post_director.py`) that the runner runs unconditionally after every Director exit.

## Architecture

### 1. `data/session_exit.md` becomes a structured contract

Current state: freeform markdown with a `reason:` line. Existing parsing (line 790 in runner) reads only `reason:`.

New state: existing markdown body preserved (operator readability), **plus** a JSON code block at the top with structured fields. Director must write both — the JSON block is the authoritative structured channel; the markdown body is the human-readable narrative.

**Schema (JSON block):**

```json
{
  "schema_version": "1.0",
  "session_id": "D-313",
  "claimed_item_id": "wq-baa39841",
  "status": "success" | "partial" | "failure" | "no_op",
  "reason": "GRACEFUL_CHECKPOINT" | "RATE_LIMIT" | "EVALUATOR_FAIL" | "SESSION_RECOVERY_NEEDED" | ...,
  "log_entry_text": "D-313 (2026-05-05, ~04:48 UTC): **P10 mechanism report delivered** ... (full markdown body to append to log.md, INCLUDING the trailing `---` separator if applicable)",
  "current_md_patches": [
    {"old": "P9 ANALYZE CLOSED + P10 MECHANISM OPEN", "new": "P10 MECHANISM REPORT DELIVERED + P10 CLOSE GATE PENDING"}
  ],
  "deliverables": [
    "programs/program_2_dense_vs_moe_sub100m/phase10_mechanism_report.md"
  ],
  "next_action": {
    "type": "phase_advance",
    "priority": "normal",
    "program": "program_2_dense_vs_moe_sub100m",
    "created_by": "director_session_d313",
    "payload": {
      "from_phase": "P10",
      "to_phase": "P10_close",
      "next_action": "dispatch_pi_for_p10_close_gate",
      "context": "..."
    }
  },
  "notes": "Optional free text for operator review."
}
```

`status` values:
- `success`: Director completed all intended work, deliverable produced, ready for follow-on.
- `partial`: Some work succeeded, some did not. `notes` explains what's incomplete. Runner still does log/current/complete; does not enqueue follow-on (or enqueues a recovery item).
- `failure`: Director attempted but could not deliver. `claimed_item_id` is failed by runner. Runner enqueues a `diagnostic_review`.
- `no_op`: Director judged nothing-to-do (e.g., heartbeat with stable state). No claim to complete; no follow-on.

`current_md_patches` is a list of `str_replace`-style operations, applied in order.

`next_action` may be `null` for `no_op` and `failure`. For `success`/`partial`, the runner enqueues it and uses its `created_by` for telemetry.

### 2. `tools/post_director.py` — the runner-side finalizer

Runs unconditionally after every Director exit (any exit code, any duration). Logic:

```
def main():
    se = read_session_exit()                      # data/session_exit.md
    claim = read_active_claim_for_director()      # claimed.jsonl, picked Director session

    if se and se.has_structured_block():
        finalize_from_structured(se, claim)
    elif se and not se.has_structured_block():
        # Legacy / minimal session_exit.md (just the reason: line).
        # Treat as no_op for the new path; legacy reason: handling continues unchanged.
        legacy_finalize(se, claim)
    else:
        # No session_exit.md at all → silent death.
        recover_from_silent_death(claim)
```

**`finalize_from_structured(se, claim)`:**

1. If `se.status` ∈ {`success`, `partial`}:
   - Append `se.log_entry_text` to `data/memories/log.md` (idempotent: skip if `session_id` already appears in log).
   - Apply each `current_md_patches[i]` as `str_replace` (skip if `old` not found AND `new` already present — already-applied case).
   - If `claim` and `claim.id == se.claimed_item_id`: `work_queue.complete(claim.id, outcome=se.status)`.
   - If `se.next_action`: `work_queue.enqueue(se.next_action)` (idempotent via deterministic ID).
2. If `se.status == "failure"`:
   - Append log entry.
   - If `claim`: `work_queue.fail(claim.id, reason=se.reason or "director-reported-failure")`.
   - Enqueue `diagnostic_review` queue item with `payload.failed_item_id` and `payload.reason`.
3. If `se.status == "no_op"`:
   - Append log entry only if non-empty (heartbeat ticks may be empty).
   - No queue mutations.

All steps log to `data/infra/post_director_telemetry.jsonl` (one record per Director exit) with `{ts, session_id, claim_id, branch_taken, mutations: [...]}`.

**`recover_from_silent_death(claim)`:**

Triggered when `session_exit.md` is absent. Logic:

1. If no `claim`: nothing to do (Director ran but had no claim — possible during legacy paths). Log `branch_taken=silent_death_no_claim`.
2. If `claim` exists:
   - Look up `claim.type` in `work_queue_handlers` schema (next §) to get `expected_deliverable_pattern` (a glob with `{program}` / `{to_phase}` / etc. interpolated from `claim.payload`).
   - Glob the pattern. For each match, check `mtime > claim.claimed_at`.
   - If at least one fresh match: **deliverable orphan**. Action: write a synthetic D-N log entry (`D-{session_id}-RECOVERY`) noting silent-death with deliverable; `work_queue.complete(claim.id, outcome={"status": "success_recovered", "synthetic": true, "deliverable": match_path})`; enqueue `next_action_template` per handler schema (rendered with `claim.payload`); enqueue a `session_recovery_review` for operator inspection.
   - If no fresh match: **silent death without deliverable**. Action: synthetic log entry noting failure; `work_queue.fail(claim.id, reason="silent_death_no_deliverable")`; enqueue `diagnostic_review`.

### 3. `work_queue_handlers.md` schema upgrade

Each `## <type>` section gains two new fields rendered as a YAML-style block at the top:

```yaml
---
type: phase_advance
expected_deliverable_pattern: "programs/{program}/phase{to_phase_num}_*.md"
next_action_template:
  type: phase_advance
  priority: normal
  payload:
    from_phase: "{to_phase}"
    to_phase: "{to_phase}_close"
    next_action: "dispatch_close_gate_for_{to_phase}"
---
```

The existing prose sections (Meaning / Action / Complete / Fail) remain unchanged for Director's eyes-on use.

A small parser (`tools/handler_schema.py`) loads the YAML blocks for `post_director.py` and `queue_scanner.py` to consume.

Per-type values are concrete strings; templates use `{var}` placeholders that map to `claim.payload[var]` plus a few derived values (`to_phase_num` extracts numeric suffix from `to_phase`).

For item types where silent-death recovery is unsafe (e.g., `cell_failed`: don't auto-complete because the failure may still be unaddressed), set `expected_deliverable_pattern: null` — recovery falls through to "no fresh match" branch and item is failed for operator review.

### 4. `queue_scanner.py` orphan_deliverable detector

New detector that runs every scan iteration. Catches the case where post_director.py *also* failed to run (e.g., runner crashed). Logic:

For each item in `claimed.jsonl` older than 30 minutes:
- Load handler schema for `item.type`.
- If `expected_deliverable_pattern` is null: skip (handler opted out).
- Glob the rendered pattern. Filter to matches with `mtime > item.claimed_at`.
- If matches exist AND no `complete`/`fail` event for `item.id` in `queue_telemetry.jsonl`: emit `session_recovery_review` queue item with payload `{orphan_item_id, deliverable_path, claimed_at}`. Idempotent via deterministic ID derived from `(orphan_item_id, "orphan_recovery")`.

When Director processes a `session_recovery_review` item, the handler is: read the orphan item, examine the deliverable, decide if it's truly complete (run `work_queue.complete()`) or needs more work (re-enqueue the original item type).

### 5. `run_agi_lab.sh` integration

Three changes:

**a. Always reclaim stale before deciding.** Move `work_queue.reclaim_stale(timeout_min=60)` from the dispatch path to the top of the main loop. Runs every iteration regardless of mode (skip vs dispatch).

**b. Always run `post_director.py` after Director exit.** Currently the runner has bespoke handling for `EXIT_REASON` parsing, evaluator-FAIL detection, signature-forgery check. Keep all of those. Add `post_director.py` as the first step after `EXIT_CODE=$?` and before any of the existing checks. Existing `EXIT_REASON=$(grep "reason:" ...)` continues to work — it parses the markdown body of session_exit.md, which the structured block sits above.

**c. Queue scanner gains the orphan_deliverable detector.** Existing 6 detectors plus this 7th. Same idempotent-via-ID pattern.

### 6. Director procedural changes

Final step of every Director session becomes:

> **Step Last: Write `data/session_exit.md`** with both:
> - JSON code block at top (per schema in `data/agents/_shared/session_exit_schema.md`)
> - Existing markdown body (reason line + free narrative)
>
> No exceptions. Even routine no-op heartbeat sessions write `{"status": "no_op", ...}`. Even failure sessions write `{"status": "failure", "reason": "...", ...}`.
>
> The runner finalizes everything (log, current, queue.complete, next-item enqueue) based on this file. **You do not** call `work_queue.complete()` yourself, **you do not** edit log.md or current.md yourself — populate the JSON fields and the runner does the rest. This is a hard procedural change from the prior pattern.

The reference schema doc is a new file: `data/agents/_shared/session_exit_schema.md` — JSON Schema + worked examples for each `status` value.

### 7. Tests

- `test_post_director.py`: 8 tests covering all branches
  - structured success → log appended, current patched, complete called, enqueue called
  - structured partial → same minus enqueue
  - structured failure → log + fail + diagnostic_review enqueued
  - structured no_op → no mutations
  - legacy session_exit.md (reason: only) → legacy_finalize path
  - silent death with deliverable → synthetic log, complete-recovered, follow-on + recovery_review enqueued
  - silent death without deliverable → synthetic log, fail, diagnostic_review enqueued
  - idempotency: running twice over same session_exit.md is a no-op the second time
- `test_handler_schema.py`: 4 tests
  - schema fields parse from markdown YAML blocks
  - template rendering with `claim.payload` substitution
  - `expected_deliverable_pattern: null` opt-out
  - all 9 existing item types have schema entries (or explicit nulls)
- `test_orphan_detector.py`: 5 tests
  - orphan claim + deliverable on disk → emits recovery item
  - orphan claim + no deliverable → no emit
  - completed item + deliverable → no emit (telemetry shows completion)
  - young claim (<30 min) → no emit even if deliverable exists
  - idempotent: second scan does not duplicate the recovery item
- `test_runner_integration.py`: 3 tests
  - `reclaim_stale` called every iteration (mock the python3 invocation, verify call)
  - `post_director.py` called after every Director exit
  - existing `EXIT_REASON` parsing still works alongside structured block

## Out of scope (v1)

- **Multi-Director concurrency.** v1 keeps one-Director-at-a-time. The structured exit file is per-session.
- **Director session-resume.** When Director silent-dies, the next session is a fresh session that processes a new queue item (possibly the recovered follow-on or a recovery_review). We do not attempt to resume the dead session.
- **Distributed runners.** Single-node only.
- **Removing legacy session_exit.md handling.** The freeform markdown body and `reason:` line continue to be parsed by existing runner code. We layer the JSON block on top, not in place of.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Director writes invalid JSON in session_exit.md | `post_director.py` validates against schema; falls back to `legacy_finalize` if JSON parse fails. Logs `branch_taken=invalid_json` for operator review. |
| `current_md_patches` `str_replace` doesn't match (text has changed since Director read it) | Skip the patch with a warning logged to `post_director_telemetry.jsonl`; do not crash. Director's procedural says to apply patches with operations that are robust to small drift (use unique anchor strings). |
| `next_action` schema drift over time | Validate against `tools/work_queue.py` schema; failed enqueue logged + included in next operator review. |
| Silent-death recovery falsely recovers a half-finished deliverable | `expected_deliverable_pattern` is set conservatively (e.g., requires the file to exist + be non-trivial size). For item types where false-recovery is high-cost (paper draft, evaluator report), set pattern to null and let the case go to `diagnostic_review`. |
| Idempotency hole in log append (Director re-runs before runner finalizes) | `log_entry_text` includes `session_id`; `post_director.py` checks if `session_id` already present in log before appending. |
| Migration breaks existing handlers | New schema fields are additive. Existing prose sections in `work_queue_handlers.md` unchanged. Default behavior for missing schema fields = treat as opt-out (null pattern). |
| `reclaim_stale` now runs every iteration → race with active Director | Existing 60-min threshold still protects active sessions. Runner's own claim is excluded by claimer-name filter (won't reclaim own active dispatch). |

## Success criteria

- Zero silent-death-induced strands after deployment. Future Director silent-deaths get auto-recovered within one runner iteration.
- D-313 (the trigger case) had been recovered manually; the new path will auto-recover any future occurrences.
- `post_director.py` telemetry shows ≥99% sessions taking the `structured` branch (legacy/silent-death are exception paths).
- Operator can answer "why is the lab sleeping?" by reading queue state alone (after the work-queue spec) AND by reading `post_director_telemetry.jsonl` for the most recent Director exit (for this spec).

## Migration

Stage-gated, each commits independently:

- **Stage R1**: `tools/handler_schema.py` (parse YAML blocks from work_queue_handlers.md) + tests.
- **Stage R2**: `data/agents/_shared/work_queue_handlers.md` schema-upgrade (YAML blocks for all 9 item types) + `data/agents/_shared/session_exit_schema.md` reference doc.
- **Stage R3**: `tools/post_director.py` structured-finalize + legacy paths + tests.
- **Stage R4**: `tools/post_director.py` silent-death recovery + tests.
- **Stage R5**: `tools/queue_scanner.py` orphan_deliverable detector + tests.
- **Stage R6**: `run_agi_lab.sh` integration (reclaim_stale every iter, post_director call) + Director procedural rewrite + tests.
- **Stage R7**: End-to-end test: simulate D-313-like silent-death scenario in a fresh sandbox, verify auto-recovery; restart runner against live lab.

Total effort: ~7-9h via subagent-driven-development. Stage R6 has highest risk (touches the runner shell script that's been edited many times); spec-compliance review subagent must verify nothing existing is removed.

## D-313 retrospective fold-in

The D-313 manual recovery (operator-performed 2026-05-05 07:14 CDT) is the ground-truth example for this spec's silent-death-recovery branch. Recovery actions:

1. `work_queue.complete('wq-baa39841', outcome={status: success_recovered, deliverable: phase10_mechanism_report.md})`
2. `work_queue.enqueue(phase_advance P10→P10_close, created_by=manual_recovery)`
3. Append `D-313-RECOVERY` entry to log.md
4. Update current.md heading

These are exactly the actions `post_director.py:recover_from_silent_death()` should automate. Stage R7's end-to-end test reproduces this scenario in a sandbox and verifies the new tool's output matches the manual recovery's effect.
