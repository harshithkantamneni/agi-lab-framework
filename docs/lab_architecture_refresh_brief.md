# Lab Architecture Refresh — A Pattern Brief for Other Autonomous Labs

**Date:** 2026-05-04
**Context:** Refactor of an autonomous LLM-driven research lab running on a single MacBook (18 GB RAM). ~30 specialized agent roles. Director coordinates, PI does science, paper_writer drafts, etc. Original config: every agent ran the most capable model at max effort, every Director session loaded ~158 KB of state. Result was expensive and slow.

This brief documents the seven patterns we landed and what each replaces, so other autonomous labs can adopt or adapt without re-deriving the rationale.

---

## TL;DR

Five token-efficiency patterns and two organic-learning patterns:

1. **Deterministic context briefs** replace "load everything every session"
2. **Heterogeneous tiering** (haiku / sonnet / opus per role) replaces "always opus"
3. **Decision-table dispatcher** replaces "Director decides each model by hand"
4. **Self-escalation contract** mitigates smaller-model overconfidence
5. **Cross-model verifier loops** catch confident-wrong-output on judgment-tier work
6. **Async consolidator** moves memory maintenance off the critical path
7. **Per-dispatch telemetry + rollup** makes default-tier adjustment data-driven

Combined: ~40× reduction in Director startup-read context (158 KB → 4 KB), ~35% cost cut on agent dispatches, and a closed organic-learning loop. All seven patterns validated against published Anthropic + research literature (citations at the bottom).

---

## The problems we hit

### Problem 1: Director sessions burn tokens on noise

A Director monitoring a long-running training run is mostly checking "is the loss still going down?" Loading the full mission, governance, agent procedurals, program docs, and decision history every session — 158 KB of context — to produce a 1 KB tick is profligate. We measured: 80%+ of Director sessions during plateau periods didn't read most of the loaded context.

### Problem 2: Skip-when-stable can deadlock on Director's own writes

Our first pass added "if nothing changed in 30 min, skip the Director session." Bug: we counted Director's own log writes as "activity," so once one session fired, the next 30-min check saw "fresh activity" and ran another session. Sessions back-to-back, infinite loop. **Fix: distinguish *external signals* (orchestrator log, user notes, run_index, training process state) from *Director write-targets* (decision log, current state file, exit reason).** Only external signals gate skipping.

### Problem 3: Phase end is silence, not signal

Variant of Problem 2: when training *finishes*, the orchestrator stops writing. To the skip-detector this looks identical to "between events" plateau. Director never wakes to do close-out. **Fix: add a phase-end detector** (orchestrator process gone AND run_index newer than last-decision file → wake once).

### Problem 4: Smaller models confidently produce shallow output

Documented in OpenAI's Sept 2025 hallucination paper and CMU's overconfidence study. If you tier `findings_curator` down to haiku without a contract, it will sometimes silently produce a wrong synthesis instead of admitting it's underqualified. Self-assessment alone is fragile.

---

## The architecture (7 patterns)

### Pattern 1 — Deterministic Context Brief

**Replaces:** "Read these N files always" in the Director's procedural.

**What:** A small Python script (`brief_assembler.py`, ~200 LOC, no LLM) runs before every Director session. It reads recent state files by deterministic rules and writes `context_brief.md` (~3 KB) tailored to the upcoming session's `session_type` (routine-monitor / phase-transition / user-action / post-failure).

**Why deterministic, not an LLM curator:** lossy summarization is the #1 documented multi-agent antipattern (MAST taxonomy, 1600 traces). Asking a smaller LLM to decide what a judgment-heavy Director session needs is the worst case. A Python script with explicit rules ("include last 5 D-N entries; include all P-* carry-forwards in head 10 KB; include active program file pointers if phase-transition") is auditable, free, fast (~50 ms), and never hallucinates.

**Director's procedural changes from "load everything" to "read brief; expand wiki on demand."** The brief includes pointers (paths only) to wiki-tier files NOT preloaded — Director can `Read` them when judgment requires.

**Result:** ~158 KB → ~3-4 KB Director startup context. ~40× reduction on routine ticks.

### Pattern 2 — Heterogeneous Model Tiering

**Replaces:** Every agent runs the most capable model.

**What:** Three tiers in `agents.json`:
- **Tier A (opus):** scientific judgment, paper writing, code review, evaluation, scientific verdicts (~19 roles in our roster)
- **Tier B (sonnet):** implementation, integration, literature search, mechanical-but-not-trivial extraction (~11 roles)
- **Tier C (haiku):** file ops, archiving, mechanical plotting, schema-validated extraction (in our roster: 0 in `agents.json` because every role had judgment content; the haiku tier hosts deterministic infra agents like `consolidator` and the `brief_assembler` caller)

**Why:** Anthropic's published multi-agent research system (Opus orchestrator + Sonnet workers) outperforms single-Opus by ~90% at lower cost. SWE-bench Verified gap between Haiku 4.5 and Opus 4.6 is <8 points. FrugalGPT/RouteLLM data: 50-98% cost reductions at matched quality on routed tasks. Tiering is conservative, not aggressive.

**Audit before applying:** Don't trust a one-line description. Read each agent's procedural; categorize by *what it produces* and *who consumes the output*. When in doubt, lean conservative — the dispatcher (Pattern 3) handles per-task downgrades.

### Pattern 3 — Decision-Table Dispatcher

**Replaces:** Director picks model by hand for each dispatch (cognitive overhead, inconsistent).

**What:** `dispatch_helper.py` exposes `dispatch(role, task) -> {model, effort, reasoning}`. It loads the role's default tier from `agents.json`, then applies keyword-based overrides:
- Opus keywords (review, propose, verdict, paper, close-out, judge) → upgrade to opus, max effort
- Sonnet keywords (implement, fix, wire, integrate, build, design) → at least sonnet, high effort
- Haiku keywords (extract, archive, format, plot, fetch, index) → downgrade to haiku, low effort (preserves opus tier for judgment roles, just at low effort)
- No match → role default, high effort

**Why a dispatcher, not prose:** an auditable function is testable, deterministic, and degrades gracefully. The Director's procedural says "call dispatch_helper.dispatch; override only if you have a judgment reason." That's minimal cognitive load.

### Pattern 4 — Self-Escalation Contract

**Replaces:** Trusting models to silently know their limits.

**What:** A shared contract document (`data/agents/_shared/self_escalation_contract.md`) referenced by every agent's procedural. The contract: on first read of an assigned task, assess fit against your tier. If task exceeds tier, return BLOCKED with `suggest_model: <higher tier>` instead of producing output.

**Tier C special clause:** "When in doubt, escalate. False BLOCKED is recoverable; confident shallow output is not."

**Why:** OpenAI 2025 hallucination paper + CMU overconfidence study both confirm models — especially smaller ones — are systematically miscalibrated. Self-assessment fails silently. **The contract makes self-assessment a *structured output* (BLOCKED / suggest_model) the dispatcher can act on,** instead of a silent capability claim. Doesn't fully solve overconfidence (Pattern 5 closes that hole), but it's the cheap layer.

### Pattern 5 — Cross-Model Verifier Loops

**Replaces:** Trusting opus-tier output unverified.

**What:** A config file (`verifier_pairs.json`) maps every opus-tier producer to a verifier role on a *different* model instance. Director enforces the loop: after producer returns, dispatch verifier; verifier returns VERIFY_PASS or VERIFY_FAIL with specific issues; max N iterations before escalating to user.

Our pairs (illustrative):
- `pi` (opus) → `lab_architect` (sonnet) — cross-tier verification of program proposals
- `paper_writer` (opus) → `code_reviewer` (different opus instance) for technical claims, `evaluator` (sonnet) for structure
- `code_reviewer` (opus) → `evaluator` (sonnet) — completeness check on the verdict
- `experimental_methodologist` (opus) → `scientific_reviewer` (different opus) — "would a reviewer accept this?"
- `mechanism_extractor` (opus) → `red_team` (different opus, adversarial)

**Why different model:** same-instance verification is documented to be miscalibrated (you can't reliably check your own work with the same prompt context). Cross-tier or cross-instance gives independent judgment. Sonnet checking opus catches "the producer overran." Opus-2 checking opus-1 catches hallucinations the original instance committed to.

### Pattern 6 — Async Consolidator

**Replaces:** Inline memory maintenance during Director sessions.

**What:** A new agent (`consolidator`, haiku-4-5) dispatched by the runner BETWEEN Director sessions, off the critical path. Triggers: ≥6 hours since last run, OR phase boundary just crossed, OR ≥10 new decision entries in the log. Tasks: archive aged log entries to `data/archives/<date>/`, refresh wiki INDEX, flag stale carry-forward tokens, log run summary.

**Why async, not inline:** matches the documented "sleep-time agent" / LightMem pattern. Inline maintenance steals tokens from the work-product session. Async consolidation gets 5-117× efficiency wins (per LightMem paper) by running off-critical-path. The runner dispatches it with `{ ... } &` — fire and forget; Director doesn't wait.

### Pattern 7 — Per-Dispatch Telemetry + Rollup

**Replaces:** Static role tiering forever.

**What:** `dispatch_helper.log_outcome(role, model, escalated, verifier_pass, task_class)` appends one JSON line per dispatch to `data/infra/dispatch_telemetry.jsonl`. A rollup script (`dispatch_rollup.py`, called by the runner each iteration) aggregates per-role rates and produces a markdown table at `data/infra/dispatch_rollup.md`. Flags raised when:
- Escalation rate > 30% per role → consider tier upgrade
- Verifier-fail rate > 20% per role → consider model change or procedural fix

**Why both rates, not just escalation:** escalation alone is one-sided — it only catches "model too small" when the model admits it. Verifier-fail catches "model too small but didn't admit it." Together, they're the actual signal. After enough dispatches accumulate (~100/role), the rollup gives data-driven evidence to adjust the static tiers in `agents.json`. **This is the organic-learning loop.**

---

## Implementation sketch

If you're adapting this to your own lab, the minimal viable build order:

1. **Pattern 1 first** (`brief_assembler.py`). Highest ROI, easiest test (just compare brief output to old startup-read size).
2. **Patterns 2 + 3 together** (audit + dispatcher). Tier the roster conservatively, dispatcher handles per-task variance.
3. **Pattern 4** (self-escalation contract) once tiering is live. Otherwise smaller models will confidently produce wrong output.
4. **Pattern 7** (telemetry) early — populating `dispatch_telemetry.jsonl` from dispatch 1 means the rollup is meaningful by the time you need it. Add `log_outcome()` calls to your dispatcher's caller pattern from day one.
5. **Pattern 5** (verifier loops) — for high-stakes deliverables only. Don't add a verifier to every dispatch; pick the top 5-7 producer roles where wrong-output costs the most.
6. **Pattern 6** (consolidator) — only after you have enough memory state to consolidate. Premature on a fresh lab.

Test discipline: every pattern landed with TDD. Tests at the time of writing this brief: 42 across 8 test files. Tests cover: brief structure + size cap, dispatcher branch logic, agents.json schema, procedural references, verifier-pair role existence, telemetry persistence, rollup math.

Suggested file layout (matches what we landed):
```
tools/
  brief_assembler.py        # Pattern 1
  dispatch_helper.py        # Pattern 3 + 7
  dispatch_rollup.py        # Pattern 7
data/agents/
  agents.json                       # Pattern 2 (tier per role)
  _shared/
    self_escalation_contract.md     # Pattern 4
    verifier_pairs.json             # Pattern 5
  consolidator/procedural.md        # Pattern 6
  <role>/procedural.md              # references _shared/* + context_brief
data/memories/
  context_brief.md          # Pattern 1 output (regenerated each session)
data/infra/
  dispatch_telemetry.jsonl  # Pattern 7 input
  dispatch_rollup.md        # Pattern 7 output
  consolidator.log          # Pattern 6 output
  stable_skip.log           # Skip-when-stable visibility
```

---

## What we'd do differently

- **Add Pattern 7 telemetry from day one.** We added it last; the rollup is empty until Director starts logging outcomes. Bootstrap the loop earlier so adjustments have data faster.
- **Skip-when-stable's `_state_is_stable()` should never include the agents' own write-targets.** Took us a bug to learn this. If you're building skip-detection, only use *external* signals.
- **Don't trust the dispatch_helper's task-keyword lookup as a complete substitute for judgment.** It catches the obvious cases. Director still needs to override on ambiguous tasks. A keyword like "review" doesn't always mean opus-tier work — `review the file list` is haiku. The dispatcher's `reasoning` field exists so Director can sanity-check.

## What we explicitly chose NOT to do

- **No vector retrieval / RAG over memory.** Adds infra complexity (embedding service, index maintenance) that's overkill at lab scale. Deterministic brief + on-demand reads handles 95% of context needs.
- **No learned model router** (FrugalGPT-style). The keyword decision table covers the common cases at zero ML cost. A learned router only pays back at much higher dispatch volume.
- **No planning agent above Director.** Director already does this implicitly. Adding a planner would create a coordination problem we don't have.
- **No per-subagent tailored brief.** Subagents get focused task descriptions; loading 3 KB of context per dispatch wastes tokens on most. We added a one-line pointer in every agent procedural so subagents that need context can read the brief on demand — opt-in, not opt-out.

## Cost / quality data points

From our deployment:
- **Director startup-read context:** ~158 KB → ~3-4 KB (tailored brief)
- **Sessions per day:** previously ~12-50 during training (back-to-back loop bug); now ~6-12 (skip-when-stable working correctly)
- **Cost per Director session:** roughly halved on routine ticks (smaller startup + reads only on demand)
- **Sonnet vs Opus per token:** ~5× cheaper. With ~37% of agent dispatches now defaulting to Sonnet, expected dispatch cost reduction ~25-35%.
- **Verifier loop overhead:** ~1 extra dispatch per opus-tier deliverable. Pays for itself if it catches even 1 wrong-output incident per dozen verifications (which research suggests it should).

## Citations

The patterns above were validated against:

- Anthropic — "How we built our multi-agent research system" (Opus + Sonnet workers, 90% improvement over single-Opus)
- Anthropic — "Effective context engineering for AI agents" (small index up-front + just-in-time retrieval)
- Anthropic — "Effective harnesses for long-running agents" (durable-spine vs. ephemeral-context split)
- Cognition — "Don't Build Multi-Agents" (carve-outs for well-defined subagent tasks)
- MAST taxonomy (arxiv 2503.13657) — 1600 multi-agent traces, 14 failure modes; "context handoff loss" and "verification gap" are the load-bearing ones for our patterns
- Chroma — "Context Rot" (focused-prompt advantage even at 300 tokens vs. 113k full prompts)
- FrugalGPT (TMLR 2024) + RouteLLM survey (arxiv 2502.00409) — cost-routing data
- Speculative cascades (Google Research) — confidence-based deferral
- OpenAI Sept 2025 — "Why Language Models Hallucinate" (systematic miscalibration)
- CMU Dietrich 2025 — AI overconfidence study
- LightMem (arxiv 2510.18866) — async consolidation efficiency wins
- Claude Agent SDK — first-class `model: 'haiku'|'sonnet'|'opus'|'inherit'` field

---

## Apply this to your lab

If you run an autonomous LLM-driven lab:

1. **Read your Director (or top-level coordinator) procedural.** Count the bytes it loads at startup. If > 30 KB, Pattern 1 will pay off.
2. **List your agent roles by output type.** "Produces verdicts" vs. "implements code" vs. "extracts data." Pattern 2 is conservative tiering on that classification.
3. **Pick your 5 highest-stakes producer roles** — Pattern 5 verifier loops are most valuable there.
4. **Instrument from dispatch 1.** Pattern 7's telemetry is meaningless without data; start logging early.
5. **Don't skip Pattern 4.** Without the self-escalation contract, smaller models will fail silently and you'll only catch it via the verifier loops (more expensive).

The pattern set is intentionally minimal. Each piece exists because we hit a specific failure mode (or because published research said we would). If your lab doesn't have a given failure mode, skip the corresponding pattern.

---

*This brief is committed to the lab repo at `docs/lab_architecture_refresh_brief.md` and licensed for reuse — adapt as you need.*
