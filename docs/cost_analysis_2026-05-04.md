# Cost Analysis — Lab Architecture Refresh (Stages 1-7)

**Date:** 2026-05-04 (re-run 21:15 CDT)
**Method:** Direct token-usage measurements from Claude Code session jsonl files at `~/.claude/projects/<repo>/`. Each jsonl line includes a `usage` block with `input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens` per turn.

**This doc was rewritten after the first pass overstated savings.** The first pass extrapolated daily volume from a 24-hour window that included a 12+ hour rate-limit gap. With more careful sample selection, the picture is more nuanced.

## Director session sizes (apples-to-apples, profile-filtered)

Profile filter: `30 ≤ turns ≤ 200 AND 40K ≤ output ≤ 250K` (excludes operator's mega-session, sub-dispatches, and unusually large validation sessions).

| | count | span | rate (sess/hr) | avg input | avg output | avg turns |
|---|---|---|---|---|---|---|
| Pre-refactor (Apr 27-29) | 70 | 27.8 h | 2.52 | 4,156,351 | 88,201 | 48 |
| Post-refactor (May 4-5) | 2 | 0.3 h | 6.30 | 4,234,636 | 85,338 | 44 |

**Per-session input size: essentially unchanged (+2%).** Per-session output: ~unchanged (-3%).

The brief_assembler reduces what Director's procedural *tells* it to load (158 KB → 4 KB on the read list), but the per-session token count is dominated by:
- Director's procedural in the cache prefix (~25 KB, grew during refactor by ~134 lines)
- Tool results (file reads, bash output, Agent dispatches with their results)
- Sub-task context returning into Director's session

The brief assembler's wiki-tier savings are real but small relative to the rest of the session.

**Session frequency: post-refactor sample too small to conclude.** Post-refactor rate of 6.3/hour is from 2 sessions in 18 minutes (Director catching up after a 5-day rate-limit gap). Steady-state rate may settle anywhere from 1-3 sessions/hour depending on workload (training-active vs. between-phases).

## Where the cost savings ACTUALLY are

**Heterogeneous tiering on agent dispatches is producing real savings.**

In a 2.5-hour post-refactor window:

| Model | Dispatches | Cumulative input | Cumulative output |
|---|---|---|---|
| claude-haiku-4-5 | 25 | 2,717,059 | 6,718 |
| claude-opus-4-7 | 1 | 43,529 | 6 |

**96% of sub-dispatches now run on haiku instead of opus.** That's ~15× cost reduction per dispatch (haiku cached: $0.10/M input vs opus cached: $1.50/M).

If sub-dispatches happened at the same rate pre-refactor (all opus), the same 2.76M input tokens would have cost ~$4.14 cached. Post-refactor: ~$0.27 cached. **~$3.87 saved in 2.5 hours of agent dispatches.**

Extrapolated to 24 hours of typical activity: **~$30-40/day saved on sub-dispatches alone.**

## Skip-when-stable activity

Real measurement: **40 successful skip events** in the post-refactor period. Each skip prevents one Director session.

Effective savings per skip: 1 Director session = ~4M cached input tokens × $1.50/M = $6 + output ~$6 = ~$12 per session prevented.

40 skips × $12 = **~$480 saved by skip-when-stable** since the runner restart on Apr 29.

But this depends on whether those skips actually represented "would have been a session." The original (buggy) skip-when-stable was firing back-to-back sessions, so many of the 40 skips were preventing the loop bug, not legitimate-Director-work-skipped. Real value: probably 50% of that = ~$240 saved.

## Combined cost picture

**Honest breakdown:**

- **Director session cost: roughly unchanged per session.** Per-session size held steady; frequency is hard to measure with current sample.
- **Sub-dispatch cost: ~15× cheaper per dispatch** thanks to heterogeneous tiering. ~$30-40/day savings at typical activity rate.
- **Skip-when-stable: real but partially fictional savings** (preventing back-to-back loop bug, not legitimate work).
- **Brief assembler: small per-session reduction** masked by Director's procedural growth + tool result accumulation.
- **Verifier loops: not yet active** — no measurable cost or savings.
- **Consolidator: not yet fired** (next trigger in ~3h or on phase boundary).

**Estimated daily cost reduction: 2-3×, not the 6-9× I first claimed.**

The original brief's "$435/day → $50-75/day" was wrong because:
1. The "0.21 sessions/hour" frequency included a 5-day rate-limit gap
2. Per-session size didn't actually drop measurably
3. I projected sub-dispatch savings without telemetry data

A more defensible estimate:
- Pre-refactor: ~$150-200/day (typical training-monitoring period)
- Post-refactor: ~$60-100/day (with sub-dispatch tier savings + some session frequency reduction)
- **Realistic savings: ~50-60% daily cost reduction.**

## What's actually working (ranked by confidence)

1. **Heterogeneous tiering ✓** — 96% of sub-dispatches now haiku. Concrete telemetry. ~15× per-dispatch savings.
2. **Skip-when-stable bugs fixed ✓** — back-to-back loop bug eliminated. Real (but partially-fictional, since the bug was the savings driver).
3. **Brief assembler ✓** — produces 4.7 KB tailored briefs. Director loads them. But savings are small, not the 40× I implied.
4. **Self-escalation contracts ✓** — present in all 30 procedurals. Not measurable yet (needs failure mode to fire).
5. **Telemetry populating ✓** — 1 entry recorded post-procedural-update. Will grow.
6. **Verifier loops ✓ configured** — but un-fired. No measurable signal yet.
7. **Consolidator ✓ configured** — un-fired. No measurable signal yet.

## What I got wrong in the first pass

- **Per-session token reduction**: claimed 22%, actually +2%. The brief_assembler's "158 KB → 4 KB" applies to the procedural's *read list*, not the session's actual token count. Procedural size, tool results, and sub-task context dominate.
- **89% daily token reduction**: claimed based on 0.21 sessions/hour rate that included a 5-day rate-limit gap. Actual steady-state rate uncertain; reduction is probably 30-50%, not 89%.
- **6-9× cost reduction**: overstated. Realistic is 2-3× given that per-session size didn't drop and frequency reduction is uncertain.

## What's defensible to claim

- **Heterogeneous tiering reduces per-dispatch cost ~15×** for sub-dispatches that hit the haiku tier (96% of them in our sample).
- **Estimated daily savings: 50-60%** (could be higher in steady state once verifier loops + consolidator come online).
- **Skip-when-stable + phase-end detector both fire correctly** in production (40 skips, 1 phase-end wake on user nudge).
- **Brief assembler produces a usable artifact** — Director loaded a 3.8 KB brief in production and confirmed all 3 patterns working.

## Recommendation

Re-run this analysis on **2026-05-11** with a full week of:
- Steady-state Director session frequency (no rate-limit gap)
- Populated dispatch telemetry (50-100 entries → real per-role rates)
- At least 1 verifier loop firing (paper draft or PI proposal)
- 1-2 consolidator runs

That data will turn the cost projection from "estimate with caveats" to "measured with confidence interval."
