# Lab Architecture Refresh — Heterogeneous Models + Context Curation

**Date:** 2026-05-04
**Status:** Approved for implementation (user, this session)
**Predecessor specs:**
- 2026-04-17-scientific-research-lab-overhaul (program model)
- 2026-04-19-memory-tool-and-wiki-refactor (memory tiering)

## Goal

Reduce per-Director-session token cost by ~5× and per-agent-dispatch cost by ~2-3× while preserving (or improving) decision quality, by introducing:

1. Heterogeneous model tiering across the 30-agent roster (haiku-4-5 / sonnet-4-6 / opus-4-7) with self-escalation and cross-tier verification
2. A two-piece context handling system: deterministic Python brief assembler (every session, free) + async consolidator agent (between sessions, off the critical path)

## Background

Currently every agent runs `claude-opus-4-7` at max effort. Every Director session reads ~158 KB of curated state at startup regardless of session purpose. Empirical observation (Apr 27 - May 4 factorial, 12 cells): ~80%+ of Director sessions are routine monitor ticks where 90% of the loaded context is unread. Phase 3's minimum-viable-tick discipline (D-286) made this worse — Director writes constantly to log.md, which the original `_state_is_stable()` check used as a "real activity" signal, causing the skip-when-stable logic to deadlock and spawn back-to-back sessions.

Two design fixes already landed in this session:
- `_state_is_stable()` no longer counts log.md/current.md/session_exit.md as activity signals (Director write-targets shouldn't gate skipping)
- `_phase_just_ended()` wakes Director once when training finishes (orchestrator dead AND run_index newer than log.md)

This spec proposes the deeper architectural fixes on top of those.

## Research-validated patterns

Internet review (this session) confirmed the design aligns with:
- Anthropic multi-agent research system (Opus orchestrator + Sonnet workers, 90.2% improvement over single-Opus)
- Claude Agent SDK first-class `model: 'haiku'|'sonnet'|'opus'|'inherit'` field
- FrugalGPT / RouteLLM cost data (50-98% cost reductions at matched quality)
- Anthropic context engineering pattern: small index up front + just-in-time retrieval (NOT a synthesized brief)

And flagged two corrections to the original verbal design:
1. The originally-proposed `context_curator` (LLM, every session) is the documented #1 antipattern (lossy summarization, MAST taxonomy). Replace with deterministic Python assembler + async consolidator.
2. Self-escalation alone is fragile — haiku-tier models are documented to confidently produce shallow output without admitting uncertainty (OpenAI Sept 2025 hallucination paper, CMU overconfidence study). Requires external verifier on opus-tier output.

## Architecture

### Component 1: `tools/brief_assembler.py`

Deterministic Python script. Reads:
- `data/memories/log.md` (last 30 D-N entries, head)
- `data/memories/current.md` (state-line + active program block)
- `data/checkpoints/phase3_factorial/run_index.json` (cell state)
- `data/user_notes.md` (Active section)
- `data/session_exit.md` (last session's exit, if exists)
- Active program directory (`programs/<current>/`) — file index only
- Recent diagnostics (`data/diagnostics/*.md`, last 7 days)

Writes `data/memories/context_brief.md`:
```
---
generated_at: <ISO>
session_type: routine-monitor | phase-transition | user-action | post-failure
size_kb: <N>
---

# Active program / phase / 1-line status
# State delta since last Director session
# Last 5 decisions (D-N: key fact, file pointers)
# Active carry-forwards (P-* items)
# What this session likely needs to do (heuristic)
# Decision-critical files (paths only, expand on demand)
# Wiki tier files NOT loaded (assume current; expand if reasoning requires)
```

Heuristics for `session_type`:
- `phase-transition` → `_phase_just_ended()` is true
- `user-action` → `data/user_notes.md` modified within last 30 min
- `post-failure` → previous session_exit reason ≠ GRACEFUL_CHECKPOINT
- `routine-monitor` → otherwise

Target output size: 10-20 KB.

### Component 2: `consolidator` agent

New agent role (haiku-4-5). Triggered by:
- Time: ≥6 hours since last consolidator run, OR
- Threshold: ≥10 new D-N entries in log.md since last run, OR
- Phase boundary: orchestrator just exited (one-time)

Tasks:
- Re-read log.md, archive D-N entries older than 7 days to `data/archives/<date>/log_consolidated.md` with a breadcrumb
- Update wiki-tier summaries (governance/, programs/, shared.md) when their dependencies have changed substantively
- Refresh `data/memories/INDEX.md` if any wiki entry was rotated
- Flag stale assumptions in `data/diagnostics/stale_assumptions.md` for next Director session

Runs OFF the critical path — dispatched between sessions, not before.

### Component 3: Heterogeneous model tiering in `agents.json`

Each agent gets a `default_model` field:
- **Tier A (opus-4-7):** pi, director, code_reviewer, evaluator, paper_writer, lab_architect, experimental_methodologist, red_team
- **Tier B (sonnet-4-6):** tooling_engineer, infrastructure_architect, implementation_engineer_*, literature_hunter, memory_steward (when actually editing memory), figure_generator (when scientifically interpretive)
- **Tier C (haiku-4-5):** findings_curator, data_fetcher, build_engineer, figure_generator (when mechanical), brief_assembler caller, consolidator

Exact assignments determined by audit in Stage 2 below.

### Component 4: `tools/dispatch_helper.py`

Decision-table function `dispatch(role, task) → {model, effort}`:
- Reads role's `default_model` from agents.json
- Applies override rules:
  - Task description matches `review|propose|close|paper|verdict` → upgrade to opus
  - Task description matches `extract|archive|format|plot|fetch|index` → keep haiku (don't upgrade)
  - Task description matches `implement|fix|wire|integrate` → at least sonnet
- Returns final `(model, effort)` tuple

Director procedural: replace prose rubric with "call `dispatch_helper.dispatch(role, task)` and use returned model/effort, unless judgment overrides."

### Component 5: Self-escalation contract

Add to every agent's procedural:
> If on first reading the task you assess that it exceeds your model tier (requires nuance, judgment, or scientific reasoning beyond mechanical execution), return immediately with status=BLOCKED, key_finding="task exceeds tier", suggest_model="<higher tier>". Do not produce shallow output.

For Tier C agents specifically, an additional clause:
> When in doubt, escalate. False BLOCKED escalations are recoverable (Director re-dispatches at higher tier, ~5K wasted tokens). Confident shallow output is not (downstream consumers act on incorrect content).

### Component 6: Verifier loops on opus-tier output

For every opus-tier deliverable, pair with a verifier subagent on a different tier (sonnet for code review, haiku for structural checks like "does this paper section have all required subsections"). Verifier output:
- VERIFY_PASS → accept producer output
- VERIFY_FAIL → return to producer with specific issues, max 2 iterations before escalating

Producer roles needing verifier loops:
- pi (proposal review): verifier = lab_architect (sonnet)
- paper_writer: verifier = code_reviewer (different opus instance) for technical claims, evaluator (sonnet) for structure
- code_reviewer: verifier = evaluator (sonnet) for completeness check

### Component 7: Telemetry — escalation + verifier-rejection rates

`data/infra/dispatch_telemetry.jsonl` — append-only log, one line per dispatch:
```json
{"ts": "ISO", "role": "...", "model_dispatched": "...", "task_class": "...", "escalated": false, "verifier_pass": null}
```

Aggregator: weekly rollup `data/infra/dispatch_rollup.md` — per-role escalation rate, verifier-pass rate. Flag roles where escalation > 30% (consider tier upgrade) or where verifier-pass < 80% (consider model change or procedural fix).

## Out of scope (this spec)

- Vector retrieval / RAG over memory tier (would supersede brief_assembler eventually, but adds infrastructure cost)
- Learned context compression (research-grade, not production-ready in this lab)
- Multi-step planning agents (Director already does this implicitly)
- Cost-based routing (FrugalGPT-style learned router) — overkill at this scale

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| brief_assembler misses critical context | Director's procedural permits "if you can't reason, expand wiki." Conservative defaults (always include current state-line + last 5 decisions + active carry-forwards). |
| Tier C agents confidently produce shallow output | Self-escalation contract + Tier C "when in doubt, escalate" clause + verifier loops on judgment-tier output (catches "too small but didn't admit it"). |
| Mid-session changes to agents.json/procedural break running Director | Implement Stage 1 (purely additive Python) while Director runs; gate Stages 2-7 on Director session completion. |
| Telemetry overhead grows unbounded | Weekly rollup + archive monthly to `data/archives/YYYY-MM/dispatch_telemetry.jsonl.gz`. |

## Success criteria

- Per-Director-session startup-read drops from ~158 KB to ~20-30 KB on routine ticks (telemetry: `startup_kb` field already logged per D-288)
- Per-Director-session token cost drops ≥50% on routine monitor ticks
- No regression in close-out / phase-transition session quality (qualitative: PI + paper outputs maintained vs prior episodic records)
- Tier-C agent escalation rate stabilizes < 30% within 2 weeks of operation (else tier was wrong)
- Zero verifier-rejected deliverables shipped to user/operator (close-out memos, papers)

## Migration

Stage-gated. Each stage commits + reviewable before next stage begins.
- **Stage 1**: brief_assembler.py + tests (additive only, safe with running Director)
- **Stage 2**: agents.json tiering + escape hatch
- **Stage 3**: dispatch_helper.py + Director procedural change
- **Stage 4**: Self-escalation contracts in all agent procedurals
- **Stage 5**: Verifier loops wired up
- **Stage 6**: Async consolidator agent (procedural + trigger logic in runner)
- **Stage 7**: Telemetry + weekly rollup
