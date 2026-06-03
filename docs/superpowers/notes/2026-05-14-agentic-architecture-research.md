# Agentic Architecture Research — 2026-05-14

Synthesis of two parallel research threads (agent framework architectures + autonomous research agents) commissioned to refine the B+C plan before implementation.

## Headline findings

### 1. Every reactive framework is idle-by-design

LangGraph, AutoGen v0.4, CrewAI Flows, Claude Code, OpenHands — all terminate when the queue empties. None auto-generate work. The architectural answer to "what do you do when there's no work?" is consistently: **don't go idle — architect so the queue structurally cannot be empty.**

### 2. Long-running systems use output→input closure

Systems that never go idle (Sakana AI Scientist, Google AI Co-Scientist, AlphaEvolve, FunSearch) do so through closed loops where outputs become inputs to the next cycle:

- Sakana: paper review → seed for next ideation
- Google: tournament winners → evolution candidates
- AlphaEvolve: Program Database always has candidates to sample

This is **structural**, not behavioral. The agent doesn't "decide" to keep working — the loop has no idle state.

### 3. The "Director-knows-but-doesn't-queue" gap is not solved generally

Long-running multi-week research programs with narrative plans → automatic queue items remains an open problem. Existing solutions:
- Make it structurally impossible (closed loops)
- Terminate rather than sustain (per-paper systems)
- Dedicated Supervisor with weighted sampling (Google AI Co-Scientist)

No published system parses narrative directly into queue items reliably.

### 4. Anthropic's own pattern: structured machine-readable backlog

From [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents): the agent reads a feature list / progress file at session start. **Never infers next work from narrative — reads a prioritized machine-readable list.** This is the direct analog of what our session_exit.md should be.

## SOTA patterns directly applicable

### Pattern 1: Validated Intention Schema (ESAA, arXiv:2602.23193)

Agents emit `{type: "intention", diff: {...}}` events; orchestrator validates and applies. State is a deterministic projection of an event log. Critically: **agents do not write state directly** — they emit validated intentions.

**Application:** Our `session_exit.md` is a weak version. Make `next_action` a required, schema-validated field. Runner rejects missing/invalid → immediate re-dispatch.

### Pattern 2: Artifact-Presence-as-Queue (AiScientist, arXiv:2604.13018)

The workspace IS the queue. Define canonical artifacts per stage. Inspect filesystem — missing artifact = work item. **Zero LLM call** for queue derivation.

**Application:** Replace our planned C (parse plan markdown) with deterministic filesystem inspection against a per-phase artifact schema.

### Pattern 3: Agent Contracts (arXiv:2601.08815)

Formal tuple `C=(I, O, S, R, T, Φ, Ψ)` — resource bounds (tokens, calls, iterations, time, cost) + success criteria. Reports **90% token reduction and 525× lower variance** vs unconstrained agents. One published case: unmonitored agent ran 11 days in a loop without contracts.

**Application:** Director gets a contract — success criterion "produce ≥N queue items per session OR declare program-complete." Zero queue items + program incomplete = detectable contract violation → runner re-dispatches.

### Pattern 4: Hypothesis Backlog with UCB Prioritization (DeepScientist)

Deep backlog of candidate hypotheses scored by `utility × quality × exploration` (Upper Confidence Bound). Experiment outcomes update scores. **Self-replenishing** — the lab can never run out of ideas within an active program.

**Application:** Maintain `data/programs/<active>/hypothesis_backlog.jsonl`. When work queue empties, draw from backlog. Update scores from experiment results.

### Pattern 5: Threshold-Triggered Reflection (Generative Agents, arXiv:2304.03442)

Reflection fires when importance-sum of recent events exceeds threshold (~2-3 times/day in original paper). Prevents both idle drift AND token waste from continuous reflection.

**Application:** PI reflection triggered by event-importance threshold, not continuously. Cheap-tier triage first, opus only on escalation.

### Pattern 6: BATS Budget Regimes (arXiv:2511.17006)

In-context budget signaling: HIGH/MEDIUM/LOW/CRITICAL regime injected into system prompt. CRITICAL → procedural override to exit fast. Makes cost discipline **architectural**, not best-effort.

**Application:** Inject budget status into Director's session_brief. CRITICAL regime overrides exploration; writes `next_action` and exits.

## Failure modes published in the literature

| Mode | Source | Our exposure |
|---|---|---|
| Hallucinated work items | HORIZON (arXiv:2604.11978) | Medium — narrative parsing could invent tasks |
| Infinite relay loops | HORIZON, OpenClaw | Low — work queue ID-based dedup catches this |
| Direction drift / catastrophic forgetting | HORIZON | Medium — long-running programs need anchor docs |
| Token-blowing reflection | Multi-agent Reflexion | High — D+E naive would burn $200-1200/month |
| Novelty blindness | AI Scientist eval | Medium — our literature_hunter scan cadence |
| Unconstrained agent loops | Agent Contracts paper | **Catastrophic — 11-day unattended loop case is our exact failure mode** |

## Direct comparison: original B+C vs SOTA-refined

| Concern | Original B+C | SOTA-refined |
|---|---|---|
| `next_action` enforcement | Procedural step Director might forget | Schema-validated; runner rejects + re-dispatches |
| Queue derivation | Parse plan markdown via heuristics | Filesystem artifact-presence inspection |
| Silent-fail detection | Heartbeat anomaly clause (4h delay) | Contract violation (immediate) |
| Self-replenishment | None | Hypothesis backlog drawn when queue empty |
| Token discipline | Best-effort skip-when-stable | Budget regime architectural override |
| Reflection cadence | (skipped) | Threshold-triggered (cheap tier triage) |

## Implementation recommendation

**Phase 1 (foundation, ~2 days):** B' + C' + D'
- B': Required schema-validated `next_action` (ESAA pattern)
- C': Artifact-presence queue projection (AiScientist pattern)
- D': Agent Contracts on Director (arXiv:2601.08815)

**Phase 2 (self-replenishing, ~1 day):** E'
- Hypothesis backlog with UCB scoring

**Phase 3 (cost discipline, ~0.5 day):** F'
- BATS budget regime injection

**Phase 4 (always-thinking, OPTIONAL, ~1 day):** G'
- Threshold-triggered PI reflection (replaces 4h heartbeat)

**Total: 2-4.5 days.** Phase 1 alone is more bang-for-buck than original B+C combined.

## Honest caveats

- Plan-text brittleness: artifact-presence avoids the issue C had with prose parsing, but the artifact schema itself needs operator-level definition per phase.
- UCB scoring requires explicit utility/quality functions per program — not free.
- Threshold-triggered reflection needs importance-scoring on events — design work.
- All patterns are well-published in 2024-2026 literature; none are unproven research.

## Citations

Core papers:
- [Effective harnesses for long-running agents (Anthropic)](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Towards an AI co-scientist (Google)](https://arxiv.org/abs/2502.18864)
- [The AI Scientist v2 (Sakana)](https://arxiv.org/abs/2504.08066)
- [Toward Autonomous Long-Horizon Engineering (AiScientist)](https://arxiv.org/abs/2604.13018)
- [Event Sourcing for Autonomous Agents (ESAA)](https://arxiv.org/html/2602.23193)
- [Agent Contracts](https://arxiv.org/html/2601.08815v1)
- [Budget-Aware Tool-Use (BATS)](https://arxiv.org/abs/2511.17006)
- [Generative Agents](https://arxiv.org/abs/2304.03442)
- [DeepScientist](https://arxiv.org/html/2509.26603v1)
- [CoALA: Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)
- [The Long-Horizon Task Mirage](https://arxiv.org/html/2604.11978)
- [Multi-agent research system (Anthropic)](https://www.anthropic.com/engineering/multi-agent-research-system)
