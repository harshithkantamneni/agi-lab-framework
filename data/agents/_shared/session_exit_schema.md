# Session Exit Schema (RO-CO v1.1)

Director's last action of every session: write `data/session_exit.md` containing
**both** a JSON code block (structured channel) and a markdown body (operator
narrative + legacy `reason:` line). The runner's `tools/post_director.py`
finalizes log/current/queue/enqueue based on the JSON block; the markdown body
is for human consumption.

## JSON block schema

```json
{
  "schema_version": "1.1",
  "session_id": "D-313",
  "claimed_item_id": "wq-baa39841",
  "status": "success",
  "reason": "GRACEFUL_CHECKPOINT",
  "log_entry_text": "D-313 (2026-05-05, ~04:48 UTC): **Phase 3 P10 mechanism report delivered.** Brief Director session ... STATUS=success / KEY_FINDING=... / FILES_MODIFIED=... / SUMMARY=...\n\n---\n",
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
      "context": "P10 mechanism report delivered; P10 close gate pending PI dispatch."
    }
  },
  "program_complete": false,
  "notes": "Optional free text for operator review."
}
```

## Required field semantics (v1.1)

`next_action` is now **REQUIRED** when `status ∈ {success, partial}` UNLESS the operator/Director is explicitly declaring program completion.

The valid combinations are:

| status | next_action | program_complete | meaning |
|---|---|---|---|
| `success` | populated | false (default) | normal forward progress; runner enqueues next_action |
| `success` | null | true | Director declares this program is DONE; no follow-on |
| `partial` | populated | false | partial work done; runner enqueues follow-on |
| `partial` | null | true | Director declares partial work but program is DONE; runner accepts (rare but valid) |
| `failure` | null | false | runner fails the claim; enqueues diagnostic_review |
| `no_op` | null | false | heartbeat or similar; no mutation |

### Why this changed (B′ from lab self-direction v2)

Two observed failure modes (D-313, D-326..D-328) both produced session_exit.md that declared work remaining IN NARRATIVE but left `next_action: null`. Lab went idle for hours/days. v1.1 closes this gap architecturally: the runner refuses to accept the session_exit when work clearly remains. See spec at `docs/superpowers/specs/2026-05-14-lab-self-direction-v2.md`.

### Schema violation: what happens

When `tools/post_director.py` detects a violation, it:

1. Writes `branch_taken: "schema_violation"` to `post_director_telemetry.jsonl`
2. Touches `data/session_exit_redispatch_pending.flag` so the runner sees it
3. The runner's next iteration spawns Director with `DIRECTOR_REDISPATCH_REASON=schema_violation_previous_session_exit` env var set
4. Director's `context_brief.md` includes a CORRECTION PROMPT block explaining what was wrong and what to do this session

Director must either:
- Populate `next_action` with the actual next critical-path item, OR
- Set `program_complete: true` with reasoning in the `notes` field, OR
- Fix whatever was missing and re-write a valid session_exit.md

## Field semantics

- `schema_version`: always `"1.1"` for v1.1.
- `session_id`: `D-N` where N is the Director session number.
- `claimed_item_id`: the `wq-XXXX` ID Director claimed at session start (or `null` if no claim — e.g., heartbeat with empty queue).
- `status`: one of:
  - `"success"`: Director completed all intended work, deliverable on disk, ready for follow-on.
  - `"partial"`: some work succeeded, some did not. Runner does log/current/complete; does not enqueue follow-on.
  - `"failure"`: Director attempted but could not deliver. Runner fails the claim and enqueues `diagnostic_review`.
  - `"no_op"`: Director judged nothing-to-do (heartbeat with stable state). No mutations.
- `reason`: legacy field, must match an existing reason value (`GRACEFUL_CHECKPOINT`, `RATE_LIMIT`, `EVALUATOR_FAIL`, etc.). Runner's existing parsing reads this from the markdown body — keep them in sync.
- `log_entry_text`: full markdown body to append to `data/memories/log.md`. Include the trailing `---` separator and the `STATUS=`/`KEY_FINDING=`/etc. one-liner per existing log convention. Runner appends at top (after the preamble), preserving reverse-chronological order.
- `current_md_patches`: list of `{old, new}` str_replace ops applied in order. Each `old` must be a unique substring of current.md at the time of patch (use long anchor strings to be robust to drift).
- `deliverables`: list of file paths Director produced this session. Used by silent-death recovery as a cross-check.
- `next_action`: a queue item (same schema as `tools/work_queue.py` enqueue accepts) to enqueue, OR `null`. `created_by` should be `director_session_<lower(session_id)>`. **v1.1: REQUIRED (non-null) when `status ∈ {success, partial}` AND `program_complete != true`. See "Required field semantics (v1.1)" above.**
- `program_complete`: optional boolean (default `false`). Set to `true` when Director is explicitly declaring the current program is done. This is the only valid escape valve that permits `next_action: null` with `status=success/partial`.
- `notes`: free text for operator. Optional.

## Markdown body conventions

Below the JSON block, write the existing freeform body Director already produces:

```markdown
reason: GRACEFUL_CHECKPOINT
session_id: D-313

(Free-text narrative for operator: what was done, what's next, any flags.)
```

The `reason:` line is parsed by existing runner code (line 790, run_agi_lab.sh).
The JSON block must precede this body so the JSON parser can extract it
without colliding with `reason:`-style lines.

## Worked examples

### Example A: success with phase advance

(See top of this doc.)

### Example B: no_op heartbeat

```json
{
  "schema_version": "1.0",
  "session_id": "D-320",
  "claimed_item_id": "wq-9a8b7c6d",
  "status": "no_op",
  "reason": "GRACEFUL_CHECKPOINT",
  "log_entry_text": "",
  "current_md_patches": [],
  "deliverables": [],
  "next_action": null,
  "notes": "heartbeat tick: state stable, training silent, no anomalies"
}
```

(Empty `log_entry_text` → runner skips log append; runner still calls
`work_queue.complete()` on the heartbeat item.)

### Example C: failure with diagnostic enqueue

```json
{
  "schema_version": "1.0",
  "session_id": "D-405",
  "claimed_item_id": "wq-deadbeef",
  "status": "failure",
  "reason": "VERIFIER_REJECTION_LOOP",
  "log_entry_text": "D-405 (...): VERIFIER_REJECTION_LOOP after 2 iterations on paper_writer / scientific_reviewer pair.\n\n---\n",
  "current_md_patches": [],
  "deliverables": [],
  "next_action": null,
  "notes": "verifier rejected paper twice; needs methodological revision before retry"
}
```

(Runner: appends log, calls `work_queue.fail(wq-deadbeef, reason="VERIFIER_REJECTION_LOOP")`,
enqueues `diagnostic_review` with payload pointing at this failure.)

### Example D: schema violation (operator-facing reference)

Director writes (INVALID — status=success but next_action=null without program_complete):

```json
{
  "schema_version": "1.1",
  "session_id": "D-T999",
  "claimed_item_id": "wq-test",
  "status": "success",
  "reason": "GRACEFUL_CHECKPOINT",
  "log_entry_text": "D-T999: did some work. Next: tooling_engineer should rewrite the adapter.\n\n---\n",
  "current_md_patches": [],
  "deliverables": [],
  "next_action": null,
  "notes": "next steps documented in log_entry_text narrative"
}
```

Runner's post_director.py response:

1. Reads JSON block → status=success → checks next_action requirement
2. `next_action=null` AND `program_complete!=true` → VIOLATION
3. Writes telemetry record: `{"branch_taken": "schema_violation", "violation_reason": "next_action null without program_complete"}`
4. Touches `data/session_exit_redispatch_pending.flag`
5. Returns 0 (non-fatal)

Runner's main loop reads the flag, sets `DIRECTOR_REDISPATCH_REASON=schema_violation_previous_session_exit`, spawns Director. Director's session_brief includes a CORRECTION PROMPT:

> Your previous session exit was rejected because status=success required either next_action populated or program_complete=true. Read current.md and the prior log_entry_text. Either: (a) declare program_complete=true if Phase 11 / Program 2 is actually done, OR (b) populate next_action with the concrete next critical-path item (e.g., the apparatus rewrite mentioned in your narrative).

Director re-writes session_exit.md with valid next_action populated, exits cleanly.

## Idempotency

- `log_entry_text` must contain `session_id` somewhere (in the heading line, for
  example). `post_director.py` checks for substring presence before appending —
  re-running over the same session_exit.md is a no-op.
- `current_md_patches` must use unique anchor strings. `post_director.py` skips
  a patch if `old` is not found AND `new` is already present (already-applied case).
- `next_action` is enqueued via `work_queue.enqueue()` which is idempotent on
  deterministic IDs; running twice does not duplicate.
