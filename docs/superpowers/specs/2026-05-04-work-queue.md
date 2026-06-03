# Event-Driven Work Queue

**Date:** 2026-05-04
**Status:** Approved (Option 3 from `2026-05-04-pi-self-correction.md` follow-up discussion)
**Replaces:** Eventually replaces skip-when-stable as the lab's wake/sleep gate. v1 is a layer on top.

## Goal

Make the lab's work explicit. Director runs when there's a work item; sleeps when the queue is empty. "Why is the lab sleeping?" is always answerable from the queue: empty = nothing to do; non-empty = next session picks up item X. Phase progression, operator nudges, training events, and verifier loops all become first-class queue items rather than implicit consequences of file mtimes.

## Why this is the structurally right answer

The current architecture conflates "no recent file changes" with "no work to do." That conflation works for `training in progress → monitor` and breaks for `training done, multi-phase close-out pending`, `paper draft awaiting review`, `verifier loop pending`, etc. Patching `_state_is_stable()` for each gap keeps the disconnect.

A work queue inverts the model: the lab's task list becomes data, not heuristic. Items are produced by events; Director is a queue processor. This matches how real research labs operate — you don't pause because nothing changed, you pause because the to-do list is empty.

## Architecture

### Storage

`data/work_queue/` directory:
- `pending.jsonl` — items awaiting processing (FIFO with priority override)
- `claimed.jsonl` — items currently being worked on (only one at a time per Director session)
- `completed/<YYYY-MM-DD>.jsonl` — daily completion log (audit trail, archived monthly)
- `failed/<YYYY-MM-DD>.jsonl` — items Director couldn't handle (escalation)

JSONL format. Each line is one item. Files are append-mostly; `pending → claimed` and `claimed → completed/failed` are atomic moves implemented by `tools/work_queue.py`.

### Item schema

```json
{
  "id": "wq-2026-05-04-0001",
  "type": "phase_advance",
  "priority": "normal",
  "program": "program_2_dense_vs_moe_sub100m",
  "payload": {
    "from_phase": "P9",
    "to_phase": "P10",
    "context": "P9 ANALYZE closed; advance to P10 mechanism extraction"
  },
  "created_at": "2026-05-04T22:30:00Z",
  "created_by": "phase_close_scanner",
  "claimed_at": null,
  "claimed_by": null,
  "completed_at": null,
  "outcome": null
}
```

### Item types (v1)

| type | priority | meaning | handler |
|---|---|---|---|
| `phase_advance` | normal | active phase done, advance to next | Director runs the next phase per program plan |
| `cell_complete` | normal | training cell finished cleanly | Director records, possibly triggers `phase_advance` |
| `cell_failed` | urgent | training cell hit fatal/error | Director investigates, recovers, decides continue/abort |
| `operator_nudge` | urgent | `user_notes.md` was modified | Director reads the new content, acts |
| `verifier_review` | normal | opus deliverable awaiting verifier | Director dispatches the verifier per `verifier_pairs.json` |
| `consolidator_run` | low | async memory maintenance due | Director dispatches consolidator |
| `paper_review` | normal | paper draft ready for review | Director runs through P12-P14 review chain |
| `carry_forward_resolve` | low | P-* token age threshold hit | Director reviews + resolves or punts |
| `heartbeat` | low | 4h idle floor — synthetic item to ensure liveness | Director runs basic state check |

### Priority handling

Priority drives ordering within `pending.jsonl`:
1. `urgent` items always processed before `normal`/`low`
2. Within same priority: FIFO by `created_at`
3. `low` items only processed when no `normal`/`urgent` pending

### Tools

**`tools/work_queue.py`** — programmatic queue operations:
- `enqueue(item) → id` — append to pending
- `peek(filter=None) → item | None` — top pending item without claiming
- `claim(id, claimer) → item` — atomic move pending → claimed
- `complete(id, outcome) → bool` — atomic move claimed → completed
- `fail(id, reason) → bool` — atomic move claimed → failed
- `summary() → dict` — counts per state per type (for telemetry)

**`tools/queue_scanner.py`** — periodic event detector that produces queue items from external state:
- Scans `data/checkpoints/*/run_index.json` for cell completions/failures not yet queued
- Compares `data/user_notes.md` mtime to last-seen → enqueue `operator_nudge` if newer
- Scans for phase-close memos → enqueue `phase_advance` for next phase
- Scans for new diagnostic files → enqueue `diagnostic_review`
- Idempotent (each item has a deterministic ID based on event hash, dedup at enqueue)

Run by the runner before each iteration.

### Runner integration

Two-stage migration:

**v1 (incremental, this spec):** queue layered ON TOP of existing skip-when-stable.
```
while true:
    queue_scanner.scan()              # populate queue from external events
    if queue.peek() is not None:
        spawn_director_with_queue_item()
    elif _state_is_stable() and within_heartbeat():
        skip
    else:
        spawn_director()  # legacy path — handles state changes the queue doesn't see yet
```

**v2 (full migration, future spec):** queue is the only signal. `_state_is_stable()` deleted. Heartbeat becomes a periodic synthetic queue item.

### Director procedural changes

Director's procedural reads `data/work_queue/pending.jsonl` at session start, claims the top item, dispatches per item type, completes/fails it on exit. Item type → handler is documented in `data/agents/_shared/work_queue_handlers.md` (a reference doc Director consults on demand).

For items Director can't handle (unknown type, prerequisites missing), it `fail()`s the item with a reason; failed items go to `data/work_queue/failed/<date>.jsonl` for operator review.

### Hooks (where items get added)

| Source | When | Item type |
|---|---|---|
| `queue_scanner.py` | Before each runner iteration | Multiple types from external state |
| Orchestrator (training scripts) | On cell exit | `cell_complete` or `cell_failed` |
| Operator | Edits `user_notes.md` | `operator_nudge` (via scanner) |
| Director | End of each session | `phase_advance`, `verifier_review`, deferred work |
| Async consolidator | After consolidation run | None (consolidator handles its own scheduling) |
| Heartbeat timer | Every 4h | `heartbeat` synthetic item |

### Telemetry

`data/work_queue/queue_telemetry.jsonl` — appended on every state change. Schema: `{ts, action, item_id, item_type, claimer}`. Used to compute:
- Throughput per item type
- Time-to-claim per priority
- Failed item rate per type
- Queue depth over time

Rollup script `tools/queue_rollup.py` produces `data/infra/queue_rollup.md` weekly.

## Out of scope (v1)

- **Multi-Director concurrency.** v1 keeps the existing one-Director-at-a-time constraint. Queue claim is atomic but only one claim is active.
- **Cross-program priority.** Queue items don't know about cross-program tradeoffs (e.g., "Program 2's paper is more urgent than Program 1's literature scan"). PI handles cross-program prioritization at the program level, not item level.
- **Distributed runners.** Single-node only.
- **Skip-when-stable removal.** v1 is a layer on top. Removal in v2 once queue is proven covering all cases.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Stale `claimed` items if Director crashes | `tools/work_queue.py` includes a `reclaim_stale(timeout_min=60)` operation; runner calls before each iteration |
| Queue grows unbounded | `completed/` archived monthly; `pending/` capped at 1000 items with overflow → `data/work_queue/overflow/<date>.jsonl` for operator review |
| Items duplicate (scanner runs scanner runs scanner runs) | Each item has a deterministic `id` derived from hash of (type, program, payload-key-fields); enqueue is idempotent on existing `id` |
| Director picks wrong item priority | Priority is set by item-creator (scanner / orchestrator / etc.) using documented rules; Director respects priority |
| Migration breaks during transition | v1 keeps both queue AND skip-when-stable; queue handles items it knows; skip-when-stable handles unknowns. Removal of skip-when-stable is a separate v2 spec. |
| Queue items miss real-time signals | `queue_scanner.py` runs every iteration (~50ms cost). Real-time signals → queue items within one iteration. |

## Success criteria

- "Why is the lab sleeping?" answerable from queue state in 1 line (empty = nothing pending; or "claimed: wq-XXX")
- Phase 3 close-out (P10-P14) progresses through 5 phases without operator intervention beyond initial seed
- 0 silent stalls during multi-phase close-out
- Queue summary shows balanced throughput across item types after a week of operation

## Migration

Stage-gated, each commits independently:
- **Stage Q1**: `tools/work_queue.py` + tests (the data layer, no integration)
- **Stage Q2**: `tools/queue_scanner.py` + tests (item production from external state)
- **Stage Q3**: Runner integration v1 (layer on top of skip-when-stable)
- **Stage Q4**: Director procedural updates (read queue, claim, complete)
- **Stage Q5**: Hooks in orchestrator + consolidator + verifier-loop dispatch (event sources)
- **Stage Q6**: Telemetry + rollup
- **Stage Q7** (separate spec, later): v2 migration — remove skip-when-stable, queue is sole signal

Total effort: ~6-8h for stages Q1-Q6. Stage Q7 is later, after queue proves itself in production.
