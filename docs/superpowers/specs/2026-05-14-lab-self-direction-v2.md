# Lab Self-Direction (LSD v2) — SOTA-Grounded Refresh

**Date:** 2026-05-14
**Status:** Drafted (supersedes informal B+C plan from 2026-05-13 brainstorming)
**Research basis:** `docs/superpowers/notes/2026-05-14-agentic-architecture-research.md`

## Problem

The lab goes idle when there's known critical-path work. Two observed failures:

1. **D-313 (2026-05-05)**: Director silent-died after producing `phase10_mechanism_report.md` (43.5 KB) without populating `next_action`. Lab stranded 2.7h until operator intervened.
2. **D-326..D-328 (2026-05-13)**: Director processed apparatus_build_followup items, exited declaring in narrative "Next: Phase B `wq-06743b3f` adapter rewrite is the remaining substantive P11 work" — but the work item was in `failed/` (operator-cleared at shutdown). Lab went idle for 30+ min before operator manual re-enqueue. Same root cause: Director's narrative knowledge never became a queue item.

The structural gap: **Director writes narrative; nothing parses narrative into queue items.** RO-CO v1 fixed silent-death recovery but didn't address the queue-derivation gap.

## Goal

Make the queue **structurally non-empty** when an active program has remaining work, without burning tokens on continuous reflection. Adopt SOTA patterns:

1. Schema-enforced `next_action` (ESAA — validated intention events)
2. Artifact-presence queue projection (AiScientist — workspace IS the queue)
3. Agent Contracts on Director (arXiv:2601.08815 — formal success criteria)
4. Hypothesis backlog with UCB prioritization (DeepScientist — self-replenishing)
5. BATS budget regimes (arXiv:2511.17006 — architectural cost discipline)
6. Threshold-triggered reflection (Generative Agents — replaces 4h heartbeat)

## Why this is the structurally right answer

Research finding: every reactive framework (LangGraph, AutoGen, CrewAI, Claude Code) is idle-by-design. Systems that don't go idle (Sakana, Google Co-Scientist, AlphaEvolve) use **output→input closure** structurally — outputs always become inputs to next cycle. The fix is not "make Director smarter about finding work" but "architect the queue so it cannot be empty while work remains."

The "Director knows what's next" property already exists in our system — it surfaces in narrative form (session_exit.md, current.md, dispatch packets). We need the **plumbing** to convert that knowledge into queue items, plus structural guarantees that the lab stays productive.

## Architecture (5 components, staged delivery)

### B': Schema-enforced `next_action` (ESAA pattern)

**Spec change:** `data/agents/_shared/session_exit_schema.md` v1.1 — `next_action` becomes REQUIRED for `status ∈ {success, partial}`. Schema validation includes:

```yaml
next_action:
  type: required-string (one of: phase_advance, apparatus_build, ..., null-if-program-complete)
  priority: required-enum (urgent | normal | low)
  program: required-string OR "" (for global items)
  payload:
    next_action: required-string (≥30 chars, describes the concrete task)
    context: required-string (≥100 chars, dispatch packet context)
    dispatch_packet_files: optional-list[string]
```

Director may set `next_action: null` ONLY if status==no_op OR an explicit `program_complete: true` field is also set.

**Runner change:** `tools/post_director.py` validates schema on read. If `status ∈ {success, partial}` AND `next_action == null` AND `program_complete != true`:
- Log `schema_violation` to telemetry
- Re-dispatch Director immediately with anomaly prompt: "Your last session exited without specifying next_action despite work remaining. Read current.md + program plan and either: (a) declare program_complete=true with reasoning, or (b) populate next_action with the next critical-path item."

This makes the gap **immediately detectable** rather than waiting 4h for heartbeat.

### C': Artifact-Presence Queue Projection (AiScientist pattern)

**New file:** `programs/<active>/artifact_schema.yaml` — declares the canonical set of artifacts per program phase. Example for P11:

```yaml
phase: P11
artifacts:
  - id: P11.apparatus
    path: programs/{program}/phase11_apparatus.md
    state: required-before-launch
  - id: P11.r3_cosign
    path: programs/{program}/phase11_r3_cosign.md
    state: required-before-launch
  - id: P11.smoke_pass
    path: data/eval/p11/smoke_a42.json
    state: required-before-launch
  - id: P11.harness_pin
    path: data/engineering/harness_commit_pin.md
    state: required-before-launch
  - id: P11.real_data_scoring
    path: programs/{program}/phase11_real_data_results.md
    state: required-for-phase-close
    prereq: [P11.apparatus, P11.r3_cosign, P11.smoke_pass, P11.harness_pin]
    handler: dispatch_measurement_theorist_and_statistical_reviewer
```

**New tool:** `tools/artifact_queue_projector.py` — runs every runner iteration. Reads artifact_schema.yaml + filesystem state, emits queue items for missing artifacts whose prereqs are satisfied. Deterministic — zero LLM calls.

**Idempotent:** ID derived from `(artifact_id, program, phase)` hash. Re-projection produces same IDs; existing pending/claimed items are not re-enqueued.

This makes the queue a **derived view of program state**, not a primary store Director must remember to maintain.

### D': Agent Contracts on Director

**New file:** `data/agents/_shared/agent_contracts.json`:

```json
{
  "director": {
    "max_tokens_per_session": 500000,
    "max_cycles_between_invocations": 14400,
    "success_criteria": [
      "session_exit.md present with valid schema",
      "next_action populated OR program_complete=true"
    ],
    "violation_handler": "immediate_redispatch_with_correction_prompt"
  },
  "pi": {
    "max_tokens_per_session": 250000,
    "success_criteria": [
      "verdict OR calibration logged",
      "no_op rationale documented if exiting without action"
    ]
  }
}
```

**Runner change:** `tools/post_director.py` checks contract on Director exit. Violation → re-dispatch with correction prompt + telemetry `contract_violation` record. The post_director_telemetry's `branch_taken` gains values: `contract_violated_redispatch`.

This catches the entire "silent fail" class structurally — no relying on Director to do the right thing.

### E': Hypothesis Backlog with UCB Prioritization

**New file:** `programs/<active>/hypothesis_backlog.jsonl` — each line:

```json
{
  "id": "hyp-001",
  "claim": "Increasing LR to 5e-3 at 100M scale will collapse MoE balance within 500 steps",
  "scores": {
    "utility": 0.7,
    "quality": 0.6,
    "exploration": 0.9,
    "ucb_combined": 1.83
  },
  "predecessor_results": ["phase3_C42_outlier"],
  "estimated_compute_hours": 12,
  "owner_role": "experimental_methodologist",
  "status": "pending"
}
```

**New tool:** `tools/hypothesis_ucb_scorer.py` — recomputes scores periodically from new evidence. UCB formula: `score = utility + sqrt(2*ln(N)/n_evidence)` where N=total experiments and n_evidence is evidence count per hypothesis. This balances exploitation (proven utility) with exploration (under-tested ideas).

**Queue draw:** When `artifact_queue_projector.py` produces no items (program in "between phases" state), runner draws from hypothesis_backlog: dequeue top-K by ucb_combined, propose as experiment design tasks for next phase.

This makes the queue **self-replenishing** at the program level. Lab never runs out of ideas within an active program.

### F': BATS Budget Regime Injection

**Spec change:** `tools/brief_assembler.py` writes a `budget_regime` block into `context_brief.md`:

```markdown
## Budget regime
- weekly_budget_remaining: $X.XX
- session_count_today: N
- regime: HIGH | MEDIUM | LOW | CRITICAL
- regime_guidance: [auto-populated based on regime]
```

Director procedural reads this block at session start. CRITICAL regime overrides exploration: write minimal `next_action`, exit fast. HIGH regime allows broader dispatch. Cost discipline becomes **architectural** rather than depending on Director judgment.

Source: `data/infra/cost_rollup.json` (already exists per CR v1) provides the budget state.

### G': Threshold-Triggered PI Reflection (replaces 4h heartbeat)

**Replace:** current `queue_scanner._detect_heartbeat` (every 4h) with `_detect_reflection_trigger`. Fires when:

- ≥3 completed phase_advance items since last reflection, OR
- ≥1 phase close memo written since last reflection, OR
- ≥24h since last reflection (floor)

When fired, emit a `pi_reflection` queue item that:
1. Routes through cheap-tier triage (sonnet 30s "anything new to think about?") FIRST
2. Escalates to opus PI only if triage returns yes

Cost: ~$0.05 per triage call × few/day = trivial. Opus escalation maybe 1-2 times/week ≈ $5-10/week. Total: <$50/month vs $200-1200/month for naive D+E.

## Out of scope (v2)

- Multi-program parallelism (sequential programs only)
- Cross-program hypothesis transfer (each program has its own backlog)
- Real-time dashboards (cost_rollup.md remains the operator UI)
- LLM-based artifact schema generation (operator defines schema at program start)

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Artifact schema bit-rot (operator forgets to update) | Schema validation in CI; missing schema = explicit operator-review item |
| Director redispatch loop (contract violation → redispatch → same violation) | Max 3 redispatches per session_exit; then schedule_operator_review |
| Hypothesis backlog hallucination | Hypotheses must cite predecessor_results (real artifacts on disk); UCB explicitly weights evidence count |
| UCB exploration takes lab into low-value territory | Per-program "max exploration weight" tunable; PI can override at reflection |
| Budget regime triggers false CRITICAL | Threshold tuned conservatively; CRITICAL = <10% weekly budget remaining (vs 30% in BATS — we're stricter) |
| Threshold reflection misses important events | 24h floor ensures min reflection cadence; importance-scoring open to operator override |

## Success criteria

- **Zero idle-with-work failures** for 2 weeks post-deploy (i.e., no 8-day stalls; no "lab stopped but had work" incidents)
- **Cost-rollup wastage events** (silent_death + contract_violated + holding_loop) trend to 0 per week
- **Queue depth never drops to 0 within an active program** (the structural invariant)
- **Total weekly lab cost stays within $50-150/week budget** despite the new patterns (artifact projection is free; UCB is cheap; reflection triage is cheap)

## Migration

Stage-gated, each commits independently:

- **Phase 1** (~2 days, FOUNDATION):
  - L1: session_exit_schema.md v1.1 + post_director.py validation + redispatch
  - L2: artifact_queue_projector.py + per-program artifact_schema.yaml for active programs
  - L3: agent_contracts.json + contract validation in post_director.py

- **Phase 2** (~1 day, SELF-REPLENISHING):
  - L4: hypothesis_backlog.jsonl schema + hypothesis_ucb_scorer.py
  - L5: queue_scanner integration (draw from backlog when artifact queue empty)

- **Phase 3** (~0.5 day, COST DISCIPLINE):
  - L6: brief_assembler budget_regime block + Director procedural read

- **Phase 4** (~1 day, OPTIONAL ALWAYS-THINKING):
  - L7: replace heartbeat with reflection_trigger + cheap-tier triage handler

**Total: 2-4.5 days via subagent-driven-development.**

Recommendation: **ship Phase 1 first**, watch 2 weeks, then decide Phase 2-4 based on observed behavior. Phase 1 alone is stronger than the original B+C combined.

## Supersedes

- The informal B+C plan brainstormed 2026-05-13. That was right intent, weaker implementation. SOTA-grounded patterns make B'+C' strictly better at same/lower effort.

## Citations (load-bearing for design decisions)

- ESAA validated intentions: [arXiv:2602.23193](https://arxiv.org/html/2602.23193)
- AiScientist artifact-presence: [arXiv:2604.13018](https://arxiv.org/abs/2604.13018)
- Agent Contracts: [arXiv:2601.08815](https://arxiv.org/html/2601.08815v1)
- DeepScientist UCB hypothesis backlog: [arXiv:2509.26603](https://arxiv.org/html/2509.26603v1)
- BATS budget regimes: [arXiv:2511.17006](https://arxiv.org/abs/2511.17006)
- Generative Agents threshold reflection: [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- Anthropic effective harnesses: [link](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Google AI Co-Scientist: [arXiv:2502.18864](https://arxiv.org/abs/2502.18864)
- Sakana AI Scientist v2: [arXiv:2504.08066](https://arxiv.org/abs/2504.08066)
