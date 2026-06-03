# Work Queue Handler Reference

Per-item-type handler reference for Director. Read this when you claim an item
to determine what to do. Each section: Meaning / Action / Complete / Fail.

---

## `phase_advance`

```yaml
expected_deliverable_pattern: "programs/{program}/phase{to_phase_num}_*.md"
next_action_template:
  type: phase_advance
  priority: normal
  payload:
    from_phase: "{to_phase}"
    to_phase: "{to_phase}_close"
    next_action: "dispatch_close_gate_for_{to_phase}"
    context: "Auto-recovered from silent-death; deliverable on disk; {to_phase} close gate pending PI dispatch."
```

**Meaning:** A program phase has closed cleanly (phase-close memo exists) and
the next phase is ready to open.

**Action:** Read the program plan (`programs/<program>/plan.md`) to identify
the next phase deliverable and gate holder. Open the phase per the standard
phase-open sequence: update `current.md` state line, dispatch the appropriate
lead agent(s), log D-N entry.

**Complete:** Phase opened — `current.md` reflects the new phase, lead agent(s)
dispatched, D-N log entry written.

**Fail:** Program plan missing or next phase is ambiguous (e.g., no plan.md);
unanimous-required gate not met; or the program is in BLOCKED/KILLED state.

---

## `cell_complete`

```yaml
expected_deliverable_pattern: "data/runs/{cell_id}/done"
next_action_template: null  # cell completion handler decides on its own
```

**Meaning:** A training cell finished cleanly (run_index.json updated with
`status: complete`). The queue scanner detected the completion event.

**Action:** Read the cell's run_index.json entry. Record the result in the
program's training log. If this cell was the final cell of the current training
phase, trigger a `phase_advance` for P8→P9 (or the appropriate next phase).

**Complete:** Training result recorded in program log; `phase_advance` enqueued
if warranted; D-N log entry written.

**Fail:** run_index.json is missing or unreadable; cell result is ambiguous
(neither `complete` nor `failed`).

---

## `cell_failed` (urgent)

```yaml
expected_deliverable_pattern: null  # failures should not be auto-recovered
next_action_template: null
```

**Meaning:** A training cell hit a fatal error. Requires immediate Director
attention — continued training is blocked.

**Action:** Read the cell's error output from run_index.json and any
`data/diagnostics/` files. Decide: (a) recoverable error → dispatch
`infrastructure_architect` or `implementation_engineer_c` to fix and re-enqueue
the cell; (b) fundamental design issue → dispatch `chief_scientist` for
abort/pivot analysis; (c) ambiguous → log the failure and dispatch `profiler`
for diagnosis.

**Complete:** Decision made and acted upon — either recovery dispatched (with
follow-up item enqueued) or abort memo written and unanimous gate initiated.

**Fail:** Error output is absent or unreadable; item is a duplicate of an
already-investigated failure (check D-N log).

---

## `operator_nudge` (urgent)

```yaml
expected_deliverable_pattern: null  # nudges are read-and-respond, no deliverable artifact
next_action_template: null
```

**Meaning:** `data/user_notes.md` was modified since the last seen mtime. The
operator left instructions or asked a question.

**Action:** Read `data/user_notes.md` in full. Act on every new item: tactical
directives → execute directly; questions → answer in `data/session_exit.md`
under an `OPERATOR_RESPONSE` section; program-level changes → may require PI
co-sign (unanimous gate).

**Complete:** Every new item in user_notes.md is addressed. D-N log entry
records what was done per item.

**Fail:** user_notes.md is missing or unreadable; content is ambiguous and
requires operator clarification before action (log in session_exit.md and
fail with explanation).

---

## `verifier_review`

```yaml
expected_deliverable_pattern: "data/agents/{verifier_role}/episodic/*verifier_review*.md"
next_action_template: null  # verifier outcomes drive different follow-ons
```

**Meaning:** An opus-tier deliverable has been produced and is awaiting
verifier dispatch per `data/agents/_shared/verifier_pairs.json`.

**Action:** Look up the producer role in `verifier_pairs.json` to identify the
correct verifier and `max_iterations`. Dispatch the verifier with the
deliverable path from `payload["deliverable_path"]`. If verifier returns
VERIFY_FAIL and iterations remain, re-dispatch the producer with the critique,
then re-dispatch the verifier.

**Complete:** Verifier returns VERIFY_PASS (or VERIFY_PASS_WITH_FLAGS). Outcome
recorded in program log.

**Fail:** `max_iterations` exhausted without VERIFY_PASS → escalate via
`session_exit.md` with reason `VERIFIER_REJECTION_LOOP`. Also fail if
deliverable path is missing from disk.

---

## `consolidator_run` (low priority)

```yaml
expected_deliverable_pattern: "data/agents/findings_curator/episodic/*km_consolidation*.md"
next_action_template: null
```

**Meaning:** Async memory maintenance is due — the consolidator's schedule
threshold has been reached (typically triggered by log.md size or time elapsed).

**Action:** Dispatch `findings_curator` with `KM-Consolidation` instruction.
Consolidator handles its own scheduling; this item is purely a wake signal.

**Complete:** Consolidator returns success. Memory tier file sizes are within
caps (verify with `memory.view INDEX.md`).

**Fail:** Consolidator returns BLOCKED or CONTEXT_FULL mid-run. Log and retry
next session (re-enqueue with same item or let scanner re-detect).

---

## `paper_review`

```yaml
expected_deliverable_pattern: "programs/{program}/paper_review_*.md"
next_action_template: null
```

**Meaning:** A paper draft is ready for the P12-P14 review chain
(peer review → revision → final).

**Action:** Follow the P12-P14 sequence per the program plan. Dispatch
`scientific_reviewer`, `statistical_reviewer`, and `red_team` in parallel
(P12). Synthesize their verdicts. If PASS → dispatch `paper_writer` for
P13 final draft. If REVISE → dispatch `paper_writer` with critique, then
re-review.

**Complete:** Paper reaches ACCEPT or CONDITIONAL_ACCEPT from all three
reviewers. P14 close memo written.

**Fail:** Draft missing from disk; program is not in P12-P14 range; unanimous
gate required for paper approval has not been initiated.

---

## `carry_forward_resolve` (low priority)

```yaml
expected_deliverable_pattern: null  # resolution is in-place edit, no new file
next_action_template: null
```

**Meaning:** A P-* carry-forward token in `log.md` or `current.md` has aged
past the threshold (typically 7 days) without resolution.

**Action:** Read the carry-forward token from `payload["token_text"]`. Assess:
(a) still valid and actionable → schedule the work (enqueue explicit item or
dispatch now); (b) superseded → resolve the token with a note; (c) blocked on
prerequisite → extend with updated context and re-age the token.

**Complete:** Token is either actioned (item enqueued/work dispatched) or
resolved (struck through with explanation in log.md). D-N entry records the
decision.

**Fail:** Token text is absent from log.md (already resolved elsewhere); or
resolution requires PI co-sign that hasn't been initiated.

---

## `heartbeat` (low priority)

```yaml
expected_deliverable_pattern: null  # heartbeat has no deliverable
next_action_template: null
```

**Meaning:** Synthetic 4-hour liveness item. Ensures Director wakes even when
no other queue items are produced. This is a health-check, not a work item.

**Action:** Run a basic state check: read `context_brief.md`, verify no stale
claims in `claimed.jsonl`, confirm training (if active) is progressing,
check `data/diagnostics/` for new files. Log a one-line D-N health note.

**Complete:** State check done, no anomalies (or anomalies enqueued as
appropriate items). D-N health note written. Write session_exit.md with
GRACEFUL_CHECKPOINT.

**Fail:** Do not fail heartbeat items — always complete them. If anomalies are
found, enqueue the appropriate item type to handle them and still complete the
heartbeat.

---

## `session_recovery_review`

```yaml
expected_deliverable_pattern: null  # this IS the recovery-review item; no further deliverable
next_action_template: null
```

**Meaning:** A previous Director session silent-died after producing a
deliverable; `tools/post_director.py` auto-completed the orphan claim and
enqueued this item for operator-style inspection (verify the deliverable
was actually complete, not partial).

**Action:** Read `payload.orphan_item_id` and `payload.deliverable` from
the queue item. Open the deliverable file and verify it is a complete
artifact (not truncated mid-paragraph; not missing expected sections; not a
half-written checkpoint). Cross-reference against the original claim's
`type` and `payload` to confirm the deliverable matches what was requested.
Log a brief D-N entry recording the verification outcome.

If the deliverable is truly complete, the auto-recovery was correct —
nothing further to do. If the deliverable is partial or wrong, dispatch the
appropriate role (typically the same role that originally produced it) to
finish/redo the work, and re-enqueue the original item type as a follow-up.

**Complete:** Verification done; D-N entry written; either no further action
needed (deliverable was complete) or follow-up enqueued (deliverable was
partial).

**Fail:** Deliverable file is missing entirely (filesystem state diverged
since the auto-recovery — should not happen but log if it does); or the
recovery payload is malformed.

---

## `diagnostic_review`

```yaml
expected_deliverable_pattern: null
next_action_template: null
```

**Meaning:** A previous Director session failed (either declared
`status=failure` in `session_exit.md` or silent-died without a recoverable
deliverable). `tools/post_director.py` failed the original claim and
enqueued this urgent item so Director investigates and decides on next
steps.

**Action:** Read `payload.failed_item_id`, `payload.reason`, and any
`payload.notes` from the queue item. Read the failed item's last-known state
(check `data/work_queue/failed/<date>.jsonl` for the original record) and
any related files (Director's last log entry, the program's recent
deliverables, training stdout if applicable). Decide:

- **Recoverable** → dispatch the appropriate role to retry / fix / continue,
  and re-enqueue the original item type with updated payload.
- **Unrecoverable but informative** → log a D-N entry summarizing the
  failure mode, update `data/killed_ideas.md` if the failure invalidates a
  hypothesis, mark the item complete with `outcome.status=triaged`.
- **Needs operator** → write a `BLOCK_OPERATOR_REVIEW` line to
  `data/user_notes.md` with the diagnostic summary, mark the item complete
  with `outcome.status=escalated_to_operator`.

**Complete:** Decision made and acted upon; D-N entry written; failed item
is either re-attempted (new queue item enqueued), triaged (closed in lab
records), or escalated.

**Fail:** Original failure record cannot be located in
`data/work_queue/failed/`; or payload is incomplete (missing
`failed_item_id`).
