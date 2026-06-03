# Architecture Refresh — Tuning Pass

**Date:** 2026-05-04
**Status:** Scoped, awaiting user pick of execution order
**Predecessor:** `docs/superpowers/specs/2026-05-04-lab-architecture-refresh.md` (Stages 1-7 complete)

## Why this exists

Stages 1-7 of the architecture refresh produced ~2-3× cost reduction (vs. the 6-9× I projected — see `docs/cost_analysis_2026-05-04.md`). The shortfall is concentrated in Director session cost: per-session size didn't drop because (a) the procedural got *bigger* during the refresh, (b) tool results accumulate in-session, and (c) the brief_assembler reduced read-list size but not actual session token count.

This tuning pass targets the three highest-leverage adjustments, each independent.

---

## Improvement A: Slim the Director procedural

**Goal:** Reduce Director's cached procedural from ~25 KB to ~12-15 KB without losing functionality.

**What grew during the refresh:**
- `+74 lines` (eaaf057): context_brief reading instructions + full dispatch_helper section with code examples
- `+27 lines` (ad85160): verifier loops section (full workflow described in prose)
- `+20 lines` (ddb6f3d): log_outcome telemetry block
- `+13 lines` (4c6567f): self-escalation contract reference

**Refactor strategy:**
- Dispatch_helper section: 50 lines of prose+code → 5-line pointer to `tools/dispatch_helper.py` docstring (Director already imports + uses it)
- Verifier loops section: full workflow prose → 3-line pointer to `data/agents/_shared/verifier_pairs.json` (config is the source of truth)
- log_outcome block: 20 lines of prose+example → 4-line pointer
- Self-escalation reference: keep (already short)

Net target: ~134 added lines → ~50 lines. Procedural drops from ~25 KB to ~13 KB.

**Files:**
- Modify: `data/agents/director/procedural.md`
- Modify: `tools/dispatch_helper.py` (add a comprehensive module docstring so the pointer in the procedural is sufficient)

**Risks:**
- Director may not "remember" how to use dispatch_helper if the in-procedural example is gone. **Mitigation:** module docstring + a one-line example in the procedural (not the full 20-line block).
- Verifier loop discipline may slip if the workflow isn't spelled out in the procedural. **Mitigation:** the verifier_pairs.json schema includes a `note` field we can read; Director's procedural says "follow the workflow described in verifier_pairs.json".

**Effort:** ~1h. Mostly prose surgery + 1 docstring.

**Success metric:** Director procedural size drops from 24,595 bytes → ~13,000 bytes. Next Director session's `cache_read_input_tokens` per turn drops proportionally (~5% per-session input reduction directly attributable, more if it cascades).

---

## Improvement B: Tier Director itself for routine monitor ticks

**Goal:** Router-style Director — Sonnet for routine monitor ticks, Opus for substantive sessions.

**What this changes:**
The runner currently launches every Director session at `--model claude-opus-4-7 --effort max`. Most sessions during training periods are routine ticks (write 1 KB to log.md, exit GRACEFUL_CHECKPOINT). They don't need opus.

**Implementation:**
- Pre-launch: `tools/brief_assembler.py` already classifies session_type. Read its output before launch.
- Select model based on session_type:
  - `routine-monitor` → `claude-sonnet-4-6 --effort high`
  - `phase-transition` / `user-action` / `post-failure` → `claude-opus-4-7 --effort max`
- Pass selected model through to `claude --print --model <X> --effort <Y>`
- If session ended without GRACEFUL_CHECKPOINT (ran into something it couldn't handle at sonnet tier), runner relaunches at opus.

**Files:**
- Modify: `run_agi_lab.sh` — read context_brief.md after `write_session_brief`, parse `session_type:` from front-matter, select model accordingly
- Modify: `data/agents/director/procedural.md` — add a "if you can't handle this at your current tier, exit with reason `TIER_TOO_LOW`" clause (analogous to the self-escalation contract for subagents)
- Modify: runner main loop — `EXIT_REASON=TIER_TOO_LOW → relaunch immediately at opus`

**Risks:**
- **Sonnet missing nuance on what looks like a routine tick.** The brief assembler classifies based on heuristics; a "routine" tick might actually need a substantive decision. **Mitigation:** the TIER_TOO_LOW exit path lets Director self-escalate. Adds one wasted sonnet session per false-routine classification (~$1) — acceptable cost.
- **Sonnet's prompt cache may behave differently** from opus. **Mitigation:** test with one routine tick, measure cache_read behavior; rollback if cache hit rate drops significantly.
- **Self-escalation discipline.** If sonnet Director silently produces shallow output instead of returning TIER_TOO_LOW, we get the same "confident wrong output" failure mode the verifier loops were designed to catch. **Mitigation:** Director procedural's TIER_TOO_LOW clause explicitly: "when in doubt, escalate" (mirrors Tier C clause in self-escalation contract).

**Effort:** ~3h. Runner edit + procedural addition + 2-3 test sessions to validate.

**Success metric:** ~80% of sessions during training periods run at sonnet. Per-tick Director cost drops ~5× on routine sessions (sonnet cached $0.30/M vs opus cached $1.50/M). Daily Director cost: ~$150/day → ~$40-60/day.

**This is the single highest-leverage change in the tuning pass.**

---

## Improvement C: Smarter skip-when-stable + dynamic heartbeat

**Goal:** Increase skip rate during quiet training periods. Fewer sessions per day = lower aggregate cost.

**Current behavior:**
- Skip if all of `user_notes.md`, `run_index.json`, `phase3_factorial_log/`, diagnostics, USER_*.md unchanged in last 30 min AND last director session within 4-hour heartbeat.

**Two refinements:**

**C1. Health-aware skip extension.** If state has been stable for ≥2 hours AND `run_index.json` shows training in_progress with healthy step rate (e.g., last cell's `last_step` is incrementing), extend the skip window from 30 min to 2 hours. Tells the loop: "training is healthy and unchanging, don't bother Director until something happens."

**C2. Dynamic heartbeat.** Current heartbeat is fixed 4h. Make it adaptive:
- **2h** during phase transitions (close-out work, new program kickoff)
- **4h** default
- **8h** during stable training plateaus (≥6h since last decision, training healthy)

The runner can compute the heartbeat per iteration based on:
- `_phase_just_ended()` → 2h heartbeat (Director catching up)
- recent log.md activity → 4h heartbeat (default)
- run_index shows steady training + log quiet → 8h heartbeat (deep monitor mode)

**Files:**
- Modify: `run_agi_lab.sh` — extend `_state_is_stable()` to return a "stability tier" (transient / stable / deep-stable), use that to compute window + heartbeat
- Add: `_compute_dynamic_heartbeat()` helper

**Risks:**
- **Director misses something important during 8h skip window.** **Mitigation:** any external state change (run_index update, user_notes nudge, diagnostic flagged) immediately ends skip. The 8h window is only when *nothing happens*, which is precisely when Director has nothing to do.
- **Stuck in deep-stable mode if heuristic misses a real signal.** **Mitigation:** monthly review of skip log entries — if Director "missed" a real event (caught only at heartbeat), tune the stability detector.

**Effort:** ~2h. Helper function + main-loop integration + telemetry-line addition for visibility.

**Success metric:** Director session frequency during stable training drops from ~2.5/hour to ~0.5/hour. Skip log shows healthy mix of "quiet (8h heartbeat)" and "active (2-4h heartbeat)" entries.

---

## Combined leverage

Sequential layering (each builds on the previous):

| | Pre | After A | After A+B | After A+B+C |
|---|---|---|---|---|
| Director per-session input (cache_read) | 4.2M | ~3.6M | ~3.6M opus / ~3.6M sonnet | unchanged |
| Cost per Director session | $6 | $5 | $5 opus / $1 sonnet | unchanged |
| Sessions/hour (steady state) | 2.5 | 2.5 | 2.5 | ~0.5 |
| Daily Director cost | $360 | $300 | ~$72 | ~$15 |

(Numbers projected, will validate with telemetry.)

If all three land: **~24× reduction in daily Director cost** vs. pre-refresh. Combined with the existing sub-dispatch tiering (already saving ~$30-40/day), realistic total daily savings: **~$300-400/day → ~$30-50/day = 8-10× total cost reduction.**

Caveat: those numbers assume steady-state operation, not refactor-period contamination. Re-validate after a week.

---

## Recommended order

Highest leverage per hour of effort:

1. **Improvement B** (Director tiering) — single biggest win, ~3h effort
2. **Improvement A** (slim procedural) — easy, fast, compounds with B, ~1h effort
3. **Improvement C** (smarter skip + dynamic heartbeat) — substantial savings on quiet days, ~2h effort

Total: ~6h of work for an additional ~3-5× cost reduction on top of current state.

If you want to pick one to start: **B** (Director tiering for routine ticks) is the highest-leverage. Easiest checkpoint: implement, run for 24 hours, measure cache hit rate + session quality, expand or rollback.
