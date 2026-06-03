# Cost Rollup (CR v1)

**Date:** 2026-05-13
**Status:** Drafted
**Triggers:** Operator question "how much did all these changes save us?" — can't answer without ground-truth per-session token + cost data. Existing rollups (dispatch, calibration, queue) track activity but not $.

## Problem

We have rich telemetry — `dispatch_telemetry.jsonl`, `post_director_telemetry.jsonl`, `calibration_telemetry.jsonl`, `queue_telemetry.jsonl`, `memory_telemetry.log` — but none of it captures token spend. The actual token data lives in `~/.claude/projects/<sanitized>/*.jsonl` (Claude Code's per-session message logs), which has accurate `input_tokens / output_tokens / cache_read_input_tokens / cache_creation_input_tokens` per message + the model name.

To answer "are the architecture changes paying off?" we need:
1. Lab-wide token + $ spend per week
2. Breakdown by model tier (opus / sonnet / haiku)
3. Breakdown by role (which agents are expensive per dispatch)
4. Wastage event counts (silent-death recoveries, failed dispatches, retries)
5. Week-over-week delta for trend visibility

Without this, "did tiering save us money?" remains a guess.

## Goal

A heuristic Python tool — `tools/cost_rollup.py` — that joins the session token logs with our existing role telemetry, multiplies by hardcoded Anthropic pricing, and emits a weekly rollup at `data/infra/cost_rollup.md` + `cost_rollup.json`. Run by the runner on Director exit (cheap, idempotent) plus available for manual invocation.

## Why this is the structurally right answer

Same architectural pattern as `tools/post_director.py` and `tools/training_digest.py`: a Python tool that joins existing structured logs into a consumable artifact. No new data plumbing — the data is already there, we just need to aggregate it.

Hardcoded pricing constants at module top mean the tool stays accurate as long as someone updates the constants when Anthropic pricing changes. This is a known trade-off vs reading prices from an API (we don't want a network call in a cron-like rollup tool).

Heuristic session→role mapping (timestamp proximity) lets us tag subagent sessions without changing the dispatch path. Future v1.1 can capture session_id at dispatch time for exact matching.

## Architecture

### 1. `tools/cost_rollup.py` — single-file tool

**Inputs (read-only):**
- `~/.claude/projects/<repo>/*.jsonl` — per-message token usage
- `data/infra/dispatch_telemetry.jsonl` — role-per-dispatch mapping
- `data/infra/post_director_telemetry.jsonl` — silent_death / error events
- `data/work_queue/queue_telemetry.jsonl` — fail / reclaim events

**Outputs:**
- `data/infra/cost_rollup.md` (~3 KB human-readable)
- `data/infra/cost_rollup.json` (~5-10 KB structured, for downstream programmatic use)

**CLI:**
```bash
python3 tools/cost_rollup.py [--week YYYY-WW] [--since YYYY-MM-DD] [--until YYYY-MM-DD]
```

Defaults: current ISO week (YYYY-Www). `--since` / `--until` override for ad-hoc ranges.

### 2. Pricing constants (module top)

```python
PRICING = {
    "claude-opus-4-7":              {"input": 15.0, "cache_read": 1.50, "output": 75.0},
    "claude-opus-4-7[1m]":          {"input": 15.0, "cache_read": 1.50, "output": 75.0},
    "claude-sonnet-4-6":            {"input":  3.0, "cache_read": 0.30, "output": 15.0},
    "claude-haiku-4-5":             {"input":  1.0, "cache_read": 0.10, "output":  5.0},
    "claude-haiku-4-5-20251001":    {"input":  1.0, "cache_read": 0.10, "output":  5.0},
    "claude-opus-4-5-20251101":     {"input": 15.0, "cache_read": 1.50, "output": 75.0},
}
# All values $ per million tokens. cache_creation priced same as input (per Anthropic docs).
DEFAULT_PRICING = PRICING["claude-opus-4-7"]  # fallback for unknown models
```

### 3. Session → role mapping (heuristic)

For each jsonl session file:
1. Read first assistant message to get the session start timestamp + model.
2. Look up nearby (within ±60 seconds) `dispatch_telemetry.jsonl` records.
3. If a match exists: tag the session as a subagent dispatch with that `role`.
4. If no match AND the session's UUID doesn't start with `agent-`: tag as Director session.
5. If no match AND UUID starts with `agent-`: tag as `subagent_unknown_role` (rare — usually means dispatch_telemetry write failed or out of window).

Cross-check: `post_director_telemetry.jsonl` records have `session_id: D-N` form. We can map "Director session N" to jsonl session_id by ordering — but this is brittle since Director numbering can collide. For v1, treat all non-agent-prefixed sessions as Director sessions, lumped together.

### 4. Aggregation logic

```python
# Per-week aggregate
{
  "week": "2026-W20",
  "date_range": ["2026-05-11", "2026-05-17"],
  "total_cost_usd": 47.32,
  "total_input_tokens": 1234567,
  "total_output_tokens": 23456,
  "total_cache_read_tokens": 12345678,
  "session_count": 42,
  "by_model": {
    "claude-opus-4-7":    {"sessions": 12, "cost_usd": 38.50, "tokens": 8.9e6},
    "claude-sonnet-4-6":  {"sessions": 25, "cost_usd":  6.20, "tokens": 4.1e6},
    "claude-haiku-4-5":   {"sessions":  5, "cost_usd":  0.40, "tokens": 0.6e6},
  },
  "by_role": {
    "director":             {"dispatches": 8,  "cost_usd": 14.20},
    "pi":                   {"dispatches": 3,  "cost_usd":  7.50},
    "measurement_theorist": {"dispatches": 1,  "cost_usd": 12.00},
    "tooling_engineer":     {"dispatches": 5,  "cost_usd":  2.50},
    "findings_curator":     {"dispatches": 8,  "cost_usd":  1.20},
    ...
  },
  "wastage_events": {
    "silent_death_recoveries": 0,
    "post_director_errors":    0,
    "failed_claims":           1,
    "escalated_dispatches":    2,
    "verifier_failures":       0,
    "holding_loop_count":      0,
  },
  "outlier_sessions": [
    {"session_id": "fd20dc...", "role": "director", "cost_usd": 4.50, "duration_min": 32},
    ...
  ],
  "delta_vs_previous_week": {
    "cost_pct_change": -22.5,
    "session_pct_change": +5.0,
    "wastage_pct_change": -80.0
  },
}
```

### 5. Wastage event detection (rules)

| Event | Source | Detection |
|---|---|---|
| `silent_death_recoveries` | post_director_telemetry | records where `branch_taken == "silent_death"` AND `error` is null |
| `post_director_errors` | post_director_telemetry | records with `error` field populated (e.g., the sys.path bug we just hit) |
| `failed_claims` | queue_telemetry | records with `action == "fail"` |
| `escalated_dispatches` | dispatch_telemetry | records with `escalated == true` |
| `verifier_failures` | dispatch_telemetry | records with `verifier_pass == false` |
| `holding_loop_count` | post_director_telemetry | sequences of ≥5 sessions within 2h where session_id repeats or branch_taken keeps cycling on the same item |

Each event class also gets a `cost_usd` estimate when we can attribute it to specific sessions (e.g., the 3 failed post_director runs from the sys.path bug cost ~$5-10 — sum the cost of those specific session_ids).

### 6. Markdown narrative shape (~3 KB)

```markdown
# Cost Rollup — Week 2026-W20 (2026-05-11 to 2026-05-17)

**Total:** $47.32  · **Sessions:** 42  · **Δ vs prev week:** −22.5%

## By model tier
| model | sessions | tokens | $ |
|---|---|---|---|
| opus-4-7 | 12 | 8.9M | $38.50 |
| sonnet-4-6 | 25 | 4.1M | $6.20 |
| haiku-4-5 | 5 | 0.6M | $0.40 |

## By role (top dispatchers)
| role | dispatches | model | total $ | $/dispatch |
| director | 8 | opus | $14.20 | $1.78 |
| measurement_theorist | 1 | opus | $12.00 | $12.00 |
| pi | 3 | opus | $7.50 | $2.50 |
| tooling_engineer | 5 | sonnet | $2.50 | $0.50 |
| findings_curator | 8 | sonnet | $1.20 | $0.15 |

## Wastage events
- 🟢 silent_death_recoveries: 0 (RO-CO eliminating this class)
- 🟢 post_director_errors: 0
- 🟡 failed_claims: 1 ($0.20 sunk)
- 🟡 escalated_dispatches: 2 ($1.50 incremental)

## Outlier sessions (>2σ above median)
- D-324 — measurement_theorist dispatch — 32 min — $14.50

## Δ vs Week 2026-W19
- Cost: $61.10 → $47.32 (−22.5%)
- Wastage cost: $12.20 → $1.70 (−86%)
- Sonnet share: 38% → 60% (tiered dispatch working)

---
*Generated by tools/cost_rollup.py on 2026-05-17. Pricing constants at module top — update when Anthropic pricing changes.*
```

### 7. Runner integration

Option A (preferred): run rollup as part of `post_director.py`'s success path — at the bottom of `main()`, after telemetry is written, fire a non-fatal `subprocess.run(["python3", "tools/cost_rollup.py"], check=False, timeout=30)`. Cheap (~2-5s), idempotent (running twice over the same week produces the same output).

Option B: separate cron-style invocation in the runner main loop, every N iterations. More controllable but adds complexity.

Spec adopts Option A.

### 8. Tests

`tests/test_cost_rollup.py`:
- 4 tests for jsonl parsing (token extraction, model extraction, missing-field handling, corrupt-line skip)
- 3 tests for pricing math (opus / sonnet / haiku rates correctly applied)
- 4 tests for session→role mapping (timestamp proximity match, agent-prefix tag, no-match tag, Director session tag)
- 3 tests for aggregation (by_model, by_role, wastage events)
- 2 tests for week/range filtering
- 2 tests for delta-vs-previous-week computation
- 1 test for markdown render (no banned ANOVA-style fields, all key columns present)

Total: 19 tests.

## Out of scope (v1)

- **Live pricing fetch.** Hardcoded constants at module top.
- **Exact session→role mapping at the message level.** v1 uses timestamp proximity. v1.1 can add session_id capture to dispatch_telemetry at dispatch time.
- **Per-task-class cost.** v1 only does per-role. Per-task-class (apparatus_build vs P10 dispatch vs paper_review) would need additional `task_class` resolution that's only partially in dispatch_telemetry.
- **Real-time dashboard.** v1 produces a weekly rollup. Real-time is what the runner already gives you via tmux.
- **Cross-project aggregation.** Single-project (this AGI directory).

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Anthropic pricing changes invalidate report | Constants at module top; date the file when prices change. v2 could fetch pricing from a JSON config. |
| jsonl session log gets archived/rotated | Tool reads from default `~/.claude/projects/<sanitized>/` path. Archived sessions are out of scope (operator can re-run with `--since` to backfill). |
| Session → role mapping false-matches (timestamp window collision) | ±60s window is conservative. Collisions logged in JSON output's `mapping_warnings` field for operator review. |
| Tool itself adds non-trivial token cost (parsing 9000+ jsonl files weekly) | No LLM involvement — pure Python parse. Should run in <5s for current corpus. Benchmarked at <10s for 14K jsonls in our earlier audit script. |
| Rollup runs concurrently with itself (multiple Director exits in rapid succession) | Idempotent — each run produces same output for the same week. Atomic file replace (write to .tmp then rename) prevents partial reads. |
| Director procedural never reads cost_rollup.md → reports unused | v1 explicitly does NOT mandate Director read it. Cost rollup is operator-facing; Director's decision-making isn't cost-aware in v1 (might be a v2 concern: PI factors cost into program selection). |

## Success criteria

- `cost_rollup.md` exists and is fresh (mtime within 24h) after every Director session.
- Total $ for the current week matches a hand-calculated audit within ±5% (parse error tolerance).
- Wastage event counts in the JSON match independent counts from each source telemetry.
- Δ vs previous week is computable (i.e., 2 consecutive weeks of data produces a non-null delta).

## Migration

Stage-gated, each commits independently:

- **Stage C1**: `tools/cost_rollup.py` jsonl session parser + pricing math + tests (~1.5h)
- **Stage C2**: session→role mapping (timestamp proximity) + aggregation by model/role + tests (~1.5h)
- **Stage C3**: wastage event detection + outlier identification + delta-vs-previous-week + tests (~1h)
- **Stage C4**: markdown + JSON renderers + main() wiring + week filtering + tests (~1h)
- **Stage C5**: runner integration (post_director.py calls cost_rollup on success branch) + end-to-end validation against this conversation's actual data (~30 min)

Total effort: ~5-6h via subagent-driven-development.

## Out-of-scope follow-ups (future specs)

- **Per-task-class cost** (which task classes are expensive)
- **Per-cell cost during training** (which factorial cells cost the most)
- **Cost-aware program selection** (PI factors $ into go/no-go decisions)
- **Real-time dashboard** (HTML page that auto-refreshes)
- **Anomaly detection** (week-over-week spike alerts)
- **Cross-project rollup** (if the operator runs multiple labs)
