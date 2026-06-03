# Scientific Research Lab Overhaul — Design Document

**Date:** 2026-04-17
**Author:** Director session (Claude, with user Harshith as PI-in-training)
**Status:** DRAFT — pending user review
**Supersedes:** `2026-04-13-agent-organization-design.md` (first org spec), `data/system_redesign.md` (2026-04-15 restructuring), `AGI_LAB_ORG_DIAGNOSIS.md` (HIVE external diagnosis)

---

## Executive Summary

The AGI lab is shifting from an **engineering-sprint rhythm** with empirical feedback loops to a **research-program rhythm** that models actual scientific practice. The redesign replaces cycles as the primary unit of work with Research Programs — each pursuing ONE scientific question through 15 phases to a paper draft.

**Core bet**: radical specialization + emergent composition. Decompose R&D into many minute, well-defined roles that each stay within LLM strengths; the collective produces better research than a generalist-heavy engineering loop.

**Scope of changes:**
- **Governance**: PI + Director dual-head with unanimous-compromise protocol and a mediator agent for deadlocks.
- **Unit of work**: Research Program (5-15 sessions each) replaces the cycle.
- **Phases**: 15 phases × ~130 sub-steps per program, with explicit gates.
- **Roster**: 30 seed roles at launch, growing organically to ~80-100 over ~10 programs via a role-promotion mechanism.
- **Context**: Cognitive Context Architecture (CCA) — four layers of memory based on CoALA taxonomy + agentic RAG retrieval. Fully local (no cloud services).
- **Infrastructure**: new `tools/lab_memory.py` (SQLite + sqlite-vec + sentence-transformers) as local semantic memory.
- **Rate-limit handling**: adopt HIVE's precise-wait system (pre-emptive compaction, stream-level detection, exact resetsAt parsing).

**Transition**: ~6-8 sessions of pre-work (infrastructure + retrospective) before Program 1 opens on the mission question.

**What stays**: the three principles, values.md (updated), pi_notes.md (refreshed), the existing C/Metal/Swift code, the mission ("beat Opus 4.7"), the constraint (18GB local), all prior research as inherited knowledge.

---

## Motivation

External diagnosis (HIVE lab) showed the current org is pattern-bound: 66% of dispatches go to `general-purpose`, 8 of 19 roles dormant, Director routinely doing specialist work, PI directives needed to override drift. The HIVE prescription (pi_notes + evaluator + templates + reframing roster) addressed the organizational rigidity but NOT the underlying issue: the lab operates as an engineering sprint with empirical feedback, not as a scientific research lab.

Evidence of the gap:
- **No scaling-law analysis** for our own architecture at our own scale.
- **No information-theoretic framing** of router collapse — solution was stacked interventions (LFB + Plan B + entropy hinge + τ-anneal) with no first-principles derivation of which is load-bearing.
- **No dense baseline** — can't attribute wins to MoE vs. recipe.
- **Single-seed runs** promoted as evidence.
- **Cycles force topic-switching** that fights deep focus.
- **"Paper" is not a first-class output** — work products are decisions, not publishable findings.
- **Retired theory roles** (physics, novel_compute, neuro) during lab_architect's Cycle 30 retro because they were dormant — the feedback loop optimized AWAY from theory generation.

The user's strategic framing: the mission "beat Opus 4.7-level capability on 18GB M3 Pro laptop" is **aspirational by design** — it forces the lab to develop capabilities it wouldn't otherwise build. The redesign restructures the lab to actually do research (not just run experiments) so it has a chance at the mission.

User's core bet: LLM cognition is most useful in narrow, well-defined specialist roles. A hive of ~80-100 minute specialists producing through a scientific process will out-research a small team of generalist agents doing ad-hoc engineering work.

---

## Section 1 — Lab Architecture

**Unit of work**: **Research Program** = one scientific question pursued through 15 phases to a paper draft. Spans 5-15 sessions typical; as many as needed.

**Sessions**: Director's execution windows (context-bounded chunks). Multiple sessions per phase, multiple phases per program. A session no longer has identity beyond "work window for the current program phase."

**Governance (L1 — 3 roles)**:
- **PI**: scientific direction. Owns: which questions, when to kill, what counts as interesting, paper-level approval.
- **Director**: execution. Owns: which agents to dispatch, phase orchestration, resource management, response to tactical user_notes.
- **Unanimous-compromise mediator**: runs only when PI ≠ Director on a unanimous-required decision. Produces a weighted memo and a proposed compromise. Does not itself decide.

**Parallelism**: Default 1 program at a time (deep focus). PI may approve a 2nd parallel program if scientifically independent AND resources allow. Hard cap 2 early.

**Meta cadence**:
- Evaluator runs at the end of **each phase** (not each cycle — phases are the new rigor unit).
- `lab_architect` runs every **3 programs** for org retros.
- `grant_reviewer` runs every **5 programs** for external skeptical review.
- PI strategic reflection: whenever triggered; at minimum every 5 programs.

---

## Section 2 — Roles & Responsibilities (30 Seed)

Each role has four memory slots per CoALA (procedural/working/semantic/episodic).

### L1 — Direction (3)
- **PI** — scientific direction, question selection, kill calls, paper-level approval.
- **Director** — execution, dispatch, phase orchestration, resource allocation.
- **unanimous_compromise_mediator** — dispatched only when PI ≠ Director.

### L2 — Scientific core (6)
- **chief_scientist** — research division lead; synthesizes cross-role findings.
- **math_theorist** — info theory, optimization theory, scaling laws, bounds.
- **experimental_methodologist** — experimental design, controls, ablations, confound mitigation.
- **hypothesis_generator** — brainstorms + formalizes + ranks falsifiable hypotheses.
- **mechanism_extractor** — first-principles explanations of observed results.
- **measurement_theorist** — metric validity, construct validity, external validity.

### L3 — Engineering grunt (5)
- **infrastructure_architect** — designs THIS program's apparatus (data pipeline, harness, tooling needs).
- **implementation_engineer_c** — C implementation, TDD, Unity tests.
- **SOTA_scout** — continuously scans literature for applicable SOTA techniques; always-on.
- **tooling_engineer** — builds program-specific measurement/profiling/analysis tools.
- **reproducibility_engineer** — seeds, versioning, data provenance, config locks.

### L4 — Optimization (3)
- **profiler** — identifies bottlenecks, roofline analysis.
- **kernel_specialist** — Metal/AMX/CPU hot-path optimization.
- **memory_optimizer** — fits more in 18GB (quantization, activation memory, gradient checkpointing).

### L5 — Quality / rigor (5)
- **scientific_reviewer** — experimental design soundness review.
- **statistical_reviewer** — analysis validity, effect sizes, confidence.
- **red_team** — adversarial reviewer; alternative explanations; stress-tests conclusions.
- **pre_reg_auditor** — pre-registration compliance.
- **code_reviewer** — correctness of C/Python code, TDD compliance.

### L6 — Knowledge integration (3)
- **literature_hunter** — finds relevant papers (arxiv, Semantic Scholar, citation chains).
- **paper_digester** — deep-reads → per-paper digest.
- **findings_curator** — maintains lab-level semantic memory, bibliography, cross-program synthesis.

### L7 — Communication (2)
- **paper_writer** — outline, prose, structure, revision.
- **figure_generator** — plots, diagrams, tables.

### L8 — Meta (3)
- **lab_architect** — org health, role promotion, retro every 3 programs.
- **grant_reviewer** — skeptical outsider review every 5 programs.
- **evaluator** — per-phase rigor audit.

### Memory structure per role
- **Procedural**: role prompt (static, in registry at `data/agents/<role>/procedural.md`).
- **Working**: current task dispatch (PI/Director hand them the task this invocation).
- **Semantic**: `data/agents/<role>/semantic.md` — accumulated domain knowledge, updated by role at end of each invocation.
- **Episodic**: `data/agents/<role>/episodic/YYYY-MM-DD_<program>_<phase>.md` — dated invocation records.

### Growth mechanism
When Director dispatches `general-purpose` with the same task_category >2× in one program, `lab_architect` auto-proposes promoting it to a registered role. When a role is dormant across 3 programs, `lab_architect` proposes retirement (moved to `retired.json` with reinstate_triggers).

Target growth trajectory: ~3-5 new roles per program in early programs, tapering to ~80-100 roles over ~10 programs.

---

## Section 3 — Phases, Deliverables, Gates

Each phase has (a) a primary deliverable file, (b) a gate holder, (c) optional back-send targets.

| # | Phase | Deliverable | Gate | Back-send targets |
|---|---|---|---|---|
| P1 | Question formation | `question.md` | PI + Director (unanimous) | — |
| P2 | Literature saturation | `prior_work.md` | chief_scientist | — |
| P3 | Theoretical framing | `theoretical_frame.md` | chief_scientist + math_theorist | P2 if lit gap |
| P4 | Hypothesis formation | `hypotheses.md` | chief_scientist + PI | P3 if frame insufficient |
| P5 | Experimental design | `experimental_design.md` | experimental_methodologist + scientific_reviewer | P4 if untestable |
| P6 | Pre-registration | `preregistration.md` (LOCKED) | PI + Director + pre_reg_auditor | P5 only; hard lock after |
| P7 | Infrastructure build | `apparatus_manifest.md` + apparatus | infrastructure_architect + code_reviewer | — |
| P8 | Execution | `execution_log.md` + raw results (N≥3 seeds default) | Director | P7 if apparatus fails |
| P9 | Analysis | `analysis.md` | statistical_reviewer | P8 if data insufficient |
| P10 | Mechanism extraction | `mechanism.md` | chief_scientist + math_theorist | P3 if theory gap (opens P3 amendment), P5 if missed confound |
| P11 | Measurement validation | `measurement_audit.md` | measurement_theorist | P5, P9 |
| P12 | Internal peer review | `peer_review.md` + responses | red_team + pre_reg_auditor | any prior with evidence |
| P13 | Paper draft | `paper_draft_v1.md` + figures | paper_writer | — |
| P14 | Revision | `paper_draft_v2.md` | PI + Director (unanimous) | P13 iteration |
| P15 | Program close | archive + findings extracted + semantic index updated | findings_curator + evaluator | — |

### Sub-step dispatch
The ~130 sub-steps (enumerated in discussion, summarized here) are the Director's dispatch menu. Director + PI pick which sub-steps apply to this program (not all 130 needed per program; small programs may use 40). Each sub-step is a discrete agent invocation with a specific deliverable.

### Phase transitions
1. Director proposes transition → gate holder reviews deliverable → if APPROVED, PI co-signs → phase closes.
2. Rejection path: gate holder lists what's missing → phase continues with additional sub-steps → re-review.
3. Escape hatch: if a phase stalls for 3+ sessions, evaluator flags it and `lab_architect` can propose restructuring.

### P6 hard lock
Pre-registration is the ONLY hard lock. Once signed, outcome-interpretation mapping cannot change. Amendments require: PI + Director + pre_reg_auditor sign-off + written justification + disclosure in the paper.

### Back-sends
When P10 finds a theory gap, it doesn't invalidate the program — it opens a **P3 amendment** (new theory work explaining observation). Pre-reg survives. This is "we learned something requiring updated theoretical frame," not "restart P3."

### Multi-seed as P8 requirement
Execution always includes N≥3 seeds unless explicitly exempted with pre-registered reason. Reproducibility-by-default.

---

## Section 4 — Context Architecture (CCA, local)

**Four integrated layers, all local. No cloud services.**

### Layer 1 — Role memory (CoALA + Intrinsic Memory)
- `data/agents/<role>/procedural.md` — role prompt (static)
- `data/agents/<role>/semantic.md` — accumulated domain knowledge, evolves with each invocation
- `data/agents/<role>/episodic/YYYY-MM-DD_<program>_<phase>.md` — dated invocation records
- Role updates its own semantic memory at the end of each invocation (part of the return template).

### Layer 2 — Program working set
- `programs/<program_name>/` with all phase deliverables
- Every agent dispatched to a program gets: `question.md` + current-phase summary + their task inputs
- Always a floor: no agent is ever missing "what program am I in"

### Layer 3 — Lab-level memory (file hierarchy + local semantic index)
- `data/shared_knowledge.md` — cross-program findings (auto-updated by findings_curator)
- `data/bibliography.md` — curated paper list
- `data/killed_ideas.md` — dead ends
- `data/knowledge_index.md` — topic-to-file map (curated; grep first, browse second)
- `programs/archive/` — closed programs (read-only, preserved)
- `legacy/cycles_1_31/` — inherited prior work (Program 0 retrospective)
- **`tools/lab_memory.py`** — local semantic index over everything above (see Section 7)

### Layer 4 — Retrieval as tools (agentic, file-based + semantic)
Every agent has these tools:
- `Grep`, `Glob`, `Read` (existing)
- `search_knowledge_index(topic)` — looks up topic → returns file paths
- `search_program(query, program=any)` — greps across program files
- `search_role_history(role, query)` — greps role's episodic files
- `find_paper(query)` — greps bibliography.md
- `search_killed(query)` — greps killed_ideas.md
- `semantic_search(query, filters, top_k)` — queries `tools/lab_memory.py` for semantic similarity hits

Agents **pull** context via tools. Director/PI hand: role + program link + task. Agent retrieves the rest.

### Why this stack
- **Agentic RAG over static RAG**: 78% vs 34% accuracy on complex queries (per 2026 survey, arXiv:2501.09136).
- **CoALA 4-memory taxonomy**: standard cognitive-science-derived framework (Princeton 2023).
- **MemGPT-style external context**: prevents context-window overflow; agents page in what they need.
- **Intrinsic role memory**: enables specialization to compound across programs — math_theorist builds a theoretical vocabulary over time; SOTA_scout builds a technique inventory.
- **Fully local**: respects "everything local" constraint; lab_memory.py runs on sqlite-vec + sentence-transformers, no external APIs.

---

## Section 5 — Unanimous Compromise Protocol (Governance)

### Default scopes (no unanimous needed)

**Director alone** decides:
- Agent dispatch (for the current sub-step)
- Session-level execution order
- Context curation for specific dispatches
- Tool/infra tasks when clearly needed
- Response to tactical user_notes items

**PI alone** decides:
- Which questions are scientifically interesting
- Hypothesis validity (falsifiability, importance)
- Mechanism plausibility (first-principles vs hand-wavy)
- Paper-level "is this a finding worth publishing"

### Unanimous required (program-level)

1. **Open a program** — is this the right question, now?
2. **Kill a program** — quitting because evidence says stop, not because frustrated?
3. **Phase gate passage** — does deliverable meet gate criteria?
4. **Pre-registration lock (P6)** — signed by PI + Director + pre_reg_auditor; amendment requires same trio.
5. **Program pivot** — changing main question mid-flight.
6. **Paper approval** — P14 final sign-off.
7. **Promote a new seed role** — adds to agents.json.

### Deadlock protocol

1. **Position statements**: each side writes `disagreement_PI.md` / `disagreement_Director.md` (<200 words, must include evidence + risk).
2. **Mediator dispatched** (`unanimous_compromise_mediator`). Reads both + relevant context. Produces `mediation_memo.md` with: PI position + why, Director position + why, agreement points, disagreement points, proposed compromise, fallback if rejected.
3. **Both review memo**. Three outcomes:
   - Both accept → adopted; log as `UNANIMOUS_COMPROMISE: <summary>` in decisions_recent.md.
   - Modification requested → mediator iterates once (max).
   - Still deadlocked → escalate to user with the memo.

### Evidence override (from values.md §1)
Evidence outranks authority. Empirical result cleanly settling a question cannot be blocked by intuition. Override requires counter-evidence, not argument.

### Timing
- Normal unanimous check: in-session (same work window).
- Mediation: one session max.
- Escalation: next user interaction.
- Hard deadline: no program-level decision sits open for >3 work sessions. Otherwise Director defaults to conservative action (don't pivot, don't kill).

### Anti-pattern prevention
- Director cannot claim "PI didn't respond in time" to proceed unilaterally on unanimous-required items.
- PI cannot veto execution-level decisions.
- Mediator cannot decide — only propose.
- Max 2 user escalations per program; more → `lab_architect` flags miscalibration.

---

## Section 6 — Transition Plan + File Structure

### Transition in 4 stages

**Stage 1 — Close current cycle mode (1-2 sessions)**
Let Cycle 32 finish: endpoint analysis of PID 38002, gate decision, benchmark eval if PASS. Close as the last cycle under old structure. Work becomes Program 0 data.

**Stage 2 — Lab infrastructure build (~3 sessions, pre-Program-1)**
- **Session A**: Build `tools/lab_memory.py` core (SQLite + sqlite-vec + sentence-transformers + Python CLI). Ingest/search/list/get/delete primitives. Unit tests.
- **Session B**: Write 30 seed-role prompts. Migrate old roles to `retired.json`. Rewrite Director prompt for program-based operation. Write new `pi_prompt.md`, `mediator_prompt.md`. Rewrite `procedures.md`.
- **Session C**: Build tool wrappers for agents (`search_knowledge_index`, `search_program`, `search_role_history`, `find_paper`, `search_killed`, `semantic_search`). Add rate-limit enhancements (see Section 7). Update `run_agi_lab.sh`.

**Stage 3 — Program 0 retrospective (~2-3 sessions)**
Create `programs/program_0_retrospective/` with structured deliverables. Index everything into `lab_memory.py`. Program 0 closes as "inherited program" — pre-indexed for semantic retrieval.

**Stage 4 — Program 1 launch**
PI + Director unanimous: open Program 1 on mission question. Program 1 Phase 1 starts with full semantic access to Program 0.

**Total transition**: ~6-8 sessions before Program 1's first real phase opens.

### File Structure

```
data/
├── pi_notes.md                        (refreshed for program-mode)
├── values.md                          (keep)
├── state.md                           (reframed: program-based)
├── killed_ideas.md                    (keep)
├── shared_knowledge.md                (keep)
├── bibliography.md                    (keep, merged with findings)
├── knowledge_index.md                 (NEW — topic → file map)
├── user_notes.md                      (keep; tactical only)
├── decisions_recent.md                (keep, rolling ~10)
├── decisions_archive.md               (keep)
├── procedures.md                      (rewritten: phases, mediation, role lifecycles)
├── index.md                           (updated)
├── agents/
│   ├── agents.json                    (30 seed roles)
│   ├── retired.json                   (old roles + reinstate_triggers)
│   ├── director_prompt.md             (rewritten for program-mode)
│   ├── pi_prompt.md                   (NEW)
│   ├── mediator_prompt.md             (NEW)
│   ├── templates/                     (engineer.md, researcher.md, + new)
│   └── <role_name>/
│       ├── procedural.md              (role's prompt)
│       ├── semantic.md                (accumulated knowledge)
│       └── episodic/
│           └── YYYY-MM-DD_<program>_<phase>.md
├── archives/                          (preserved; archive never delete)
└── legacy/
    └── cycles_1_31/
        ├── decisions_archive.md
        ├── summaries/
        ├── session_logs/
        ├── state_snapshots/
        └── org_retros/
programs/
├── program_0_retrospective/
│   ├── question.md
│   ├── prior_work.md
│   ├── findings.md
│   ├── killed.md
│   ├── open_hypotheses.md
│   ├── infrastructure.md
│   ├── data_assets.md
│   └── bibliography_inherited.md
├── program_1_opus47_on_18gb/
│   ├── question.md                    (P1)
│   ├── prior_work.md                  (P2)
│   ├── theoretical_frame.md           (P3)
│   ├── hypotheses.md                  (P4)
│   ├── experimental_design.md         (P5)
│   ├── preregistration.md             (P6 — LOCKED)
│   ├── apparatus_manifest.md          (P7)
│   ├── execution_log.md               (P8)
│   ├── results_raw/                   (P8)
│   ├── analysis.md                    (P9)
│   ├── mechanism.md                   (P10)
│   ├── measurement_audit.md           (P11)
│   ├── peer_review.md                 (P12)
│   ├── paper_draft_v1.md              (P13)
│   ├── figures/                       (P13)
│   ├── paper_draft_v2.md              (P14)
│   └── close_manifest.md              (P15)
├── archive/                           (closed programs; read-only)
└── portfolio.md                       (program portfolio manager)
tools/
├── lab_memory.py                      (NEW)
├── lab_memory.db                      (generated; SQLite with sqlite-vec)
├── stream_formatter.py                (updated — see Section 7)
└── ... (existing tools preserved)
```

### Program 0 retrospective content (concrete)
- `findings.md`: distilled from `shared_knowledge.md` + D-079 through D-099, organized by topic (training methods, architecture, optimization, data).
- `killed.md`: distilled from `killed_ideas.md` + failed experiments with mechanism.
- `open_hypotheses.md`: the questions cycles 1-31 raised but didn't close — candidates for future programs.
- `infrastructure.md`: tour of `src/` — what's built, what's tested, what works.
- `data_assets.md`: what's tokenized (WT-103 147M tokens; OWT 3GB; Python 2GB; GSM8K; MATH; 32K tokenizer).

---

## Section 7 — Infrastructure Enhancements

### 7.1 `tools/lab_memory.py` — local semantic memory (replaces goodmem)

**Why**: goodmem is cloud-hosted (requires `GOODMEM_BASE_URL` + API key), violates "everything local" constraint.

**Stack**:
- SQLite (stdlib) for metadata: programs, phases, roles, timestamps, source paths.
- `sqlite-vec` extension for vector storage + cosine similarity. Single SQLite extension, embedded, no separate server.
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings (~80MB, CPU ~50ms/chunk, MTEB-competitive).
- Python wrapper exposing: `ingest(path)`, `search(query, filters, top_k)`, `list(space)`, `get(id)`, `delete(id)`.
- Bash CLI: `python tools/lab_memory.py search "query" --program all --role math_theorist --top-k 5`.

**Memory footprint when running**:
- Embedding model in RAM: ~200MB (loaded on demand, cached).
- SQLite DB: starts small, ~500KB per 1000 chunks.
- Fits easily in 18GB, doesn't interfere with training.

**Size estimate**: ~600-700 LOC Python, buildable in 2-3 sessions.

**Lab-specific schema** (not generic):
- `program_id`, `phase`, `role`, `deliverable_type`, `closed_program_flag`, `timestamp`, `source_path`, `chunk_text`, `embedding`.
- First-class APIs matching our workflow: `recall_role(role, query)`, `search_program(query, program)`, `find_killed(query)`.

### 7.2 Rate-limit handling (adopt HIVE's system)

Current `run_agi_lab.sh` has basic rate-limit detection (session_exit.md RATE_LIMIT + log-scan fallback) with blunt 5-minute sleep + probe. HIVE's system is more precise. Adopt it.

**Additions**:

1. **Pre-emptive compaction**: Export `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50` in runner. Cycles compact at 50% of context instead of ~90% default → smaller requests → more cycles per rate-limit window.

2. **Stream-level detection**: Modify `tools/stream_formatter.py` to parse `rate_limit_event` messages in the stream-json output. Capture `resetsAt` (Unix timestamp) to `data/infra/rate_limit_resets_at`. Only act on `status="rejected"` (suppress noisy `allowed_warning`).

```python
# tools/stream_formatter.py additions
elif msg_type == "rate_limit_event":
    info = msg.get("rate_limit_info", {})
    resets_at = info.get("resetsAt", None)
    if resets_at:
        with open("data/infra/rate_limit_resets_at", "w") as f:
            f.write(str(resets_at))
```

3. **Precise wait function** in `run_agi_lab.sh`:

```bash
wait_for_rate_limit_reset() {
    if [ -f "data/infra/rate_limit_resets_at" ]; then
        RESETS_AT=$(cat data/infra/rate_limit_resets_at)
        NOW=$(date +%s)
        WAIT_SECS=$((RESETS_AT - NOW + 30))   # +30s buffer
        if [ $WAIT_SECS -gt 0 ]; then
            echo "Rate limit resets at $(date -r $RESETS_AT). Waiting ${WAIT_SECS}s."
            sleep "$WAIT_SECS"
        fi
    else
        while ! probe_ready; do
            echo "No reset timestamp; probing every 5m. ($(date))"
            sleep 300
        done
    fi
}
```

4. **Replace current blunt sleep** in RATE_LIMIT case with `wait_for_rate_limit_reset`.

5. **No fallback model** — no Opus→Haiku degradation. Wait, don't downgrade. Consistent cycle quality.

**What this buys us**:
- Waking up exactly when capacity returns (vs guessing with 5-min polls).
- More cycles per rate-limit window (50% compact → 2× frequency).
- Better rate-limit robustness mid-program (current cycle 31 hit rate limit at 2am reset; HIVE's system would have resumed at exactly 2:00:30am instead of guessing).

### 7.3 Role registry infrastructure

With 30+ growing to 100+ roles, prompt dispatch via `Agent` tool needs a registry lookup — Director can't hold all prompts.

**Mechanism**:
- `data/agents/agents.json` is the registry (machine-readable).
- Each role's `procedural.md` is the human-readable prompt (mirror of JSON).
- Director invokes via `Agent` tool with `subagent_type: "<role>"`.
- Runner's `--agents $AGENTS_FILE` flag already handles this — it's loaded per-session.
- Growth: `lab_architect` proposes JSON additions between programs; Director applies.

### 7.4 Evaluator teeth (preserved from current design)

Current runner already has `EVALUATOR_FAIL_UNADDRESSED` exit code handling. Keep it. Extend to check at phase boundaries (not only session exits) — evaluator runs per-phase in the new design, so its FAIL flag gates phase transitions, not just session exits.

### 7.5 Model & Effort

**Model**: All 30 seed roles run on **Opus 4.7 explicitly**. `agents.json` uses `"model": "claude-opus-4-7"` (not the string `"opus"` which depends on Claude Code default resolution and could silently pin to 4.6). Runner flag also made explicit: `--model claude-opus-4-7`.

**Effort**: `--effort max` at the Director session level. This propagates to subagent dispatches. No `auto` mode anywhere.

**Rationale**:
- `auto` uses a heuristic to decide reasoning depth per turn. For research tasks — which often look simple surface-level but need deep reasoning to do well (e.g., "write prior_work.md", "digest this paper", "extract mechanism") — the heuristic under-reasons. `max` forces always-max.
- Cost of `max` is rate limits, not dollars. HIVE's rate-limit system (7.2) compensates: precise wait timing + 50% autocompact override means more cycles per rate-limit window.
- CLAUDE.md directive "quality over everything, time not a constraint" aligns with max over auto.

**Uniform Opus for all roles (initial)**:
Start with uniform Opus 4.7 for all 30 seed roles. Do NOT pre-tier by role complexity.

**Organic tiering (deferred)**:
As the lab matures, some roles will consistently produce simple outputs where Opus reasoning is overkill (e.g., `literature_hunter` doing keyword searches; `figure_generator` making basic plots). When `lab_architect` notices this pattern in a retro, it proposes downgrading specific roles to **Sonnet 4.6** or **Haiku 4.5** via the standard role-update mechanism. PI + Director unanimous approves. Lab memory (role's `semantic.md`) is preserved across model changes. This tiering is organic (emerges from observed patterns) not prescriptive (we don't decide upfront).

**Anti-pattern**: Director must NOT downgrade a role's model mid-program to save rate-limit budget. Model changes apply between programs only. Mid-program model changes break reproducibility.

**Implementation**: Locked in during Stage 2B (role prompt migration). All 30 seed roles get `"model": "claude-opus-4-7"` in agents.json. Runner flag updated to `--model claude-opus-4-7 --effort max`.

---

## Section 8 — Program 1 Scope

**Question (inherited from mission, framed scientifically)**: Can we fit an Opus-4.7-equivalent model on an 18GB M3 Pro laptop? More specifically: what is the feasibility envelope given current techniques + 18GB unified memory + no cloud compute?

**Prior work inheritance**: Cycles 1-31 become Program 0. Their findings are indexed. Program 1's P2 (literature saturation) treats Program 0's outputs as "our own lab's prior work" alongside external literature.

**Why this as Program 1**: It's the lab's north star. Every subsequent program contributes a chunk toward answering it. The first program's job is to frame the problem scientifically, survey what's known (internal + external), derive theoretical predictions, and design the first investigation.

**Expected output of Program 1**: a paper that either (a) lays out the feasibility envelope with evidence, including what's needed to close gaps, OR (b) reports negative results with specific conditions under which the mission as stated is not achievable at current SOTA. Both are valuable — a negative result with clear conditions is a paper that saves years.

**This is not an expectation of success at the mission in Program 1**. It's an expectation of scientifically rigorous first-pass investigation. Success = a defensible paper either way.

---

## Section 9 — What Stays Unchanged

- **Three principles** (Evidence Before Action / Don't Waste Resources / Question the Direction) — keep.
- **Values.md** — minor updates for program-mode phrasing; core values preserved.
- **Mission** ("beat Opus 4.7 on all benchmarks") — unchanged. Aspirational by design.
- **18GB local constraint** — unchanged. Non-negotiable.
- **All existing C/Metal/Swift code** — preserved. Program 1's infrastructure phase may extend but not rebuild.
- **All raw data** — tokenizer_32k.bin, WT-103 tokenized, OpenWebText, Python code, GSM8K, MATH — preserved.
- **Archive-never-delete policy** — unchanged.
- **HIVE's 5 structural fixes already applied** (pi_notes.md, director prompt reframe, templates, evaluator agent, runner comment) — preserved and built upon.

---

## Section 10 — Open Questions / Decisions Deferred

1. **Exact PI prompt authorship** — write it during Stage 2 Session B. Should sound like a scientist, not a manager.
2. **How "closed" is a closed program?** Can a later program REOPEN a closed program if new evidence invalidates the paper's conclusions? Proposed: yes, but creates a new program (e.g., `program_5_reopen_of_1`) citing the original.
3. **External publication policy** — do we ever actually submit to arxiv? Deferred. Default: no, internal lab archive is the end.
4. **Data provenance & audit** — how strict is our provenance chain? Reproducibility_engineer role handles the basics; deeper data-audit roles can grow organically.
5. **Who writes the Program 0 retrospective?** — proposal: a one-time dispatch of `findings_curator` + `literature_hunter` + `paper_digester` working as a team, PI + Director reviewing.
6. **What if Program 1's P1 (question formation) decides the mission question is too broad for one program?** — allowed. Program 1 scopes to a specific sub-question; the mission stays as the lab's north star for the program portfolio.

---

## Section 11 — Implementation Order

1. **User reviews this spec** (blocking).
2. **Stage 1**: close Cycle 32 under old structure (~1-2 sessions).
3. **Stage 2A**: build `tools/lab_memory.py` + unit tests.
4. **Stage 2B**: write 30 seed role prompts, migrate roster, rewrite Director/PI/Mediator prompts, rewrite procedures.md.
5. **Stage 2C**: tool wrappers for agents, rate-limit enhancements to runner + stream_formatter.
6. **Stage 3**: Program 0 retrospective, indexed into lab_memory.
7. **Stage 4**: open Program 1 on the mission question. P1 begins.

---

## Appendix A — Sources

- [CoALA: Cognitive Architectures for Language Agents (Princeton, 2023)](https://openreview.net/pdf?id=U51WxL382H)
- [MemGPT: LLMs as Operating Systems (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)
- [Agentic RAG Survey (arXiv:2501.09136)](https://arxiv.org/abs/2501.09136)
- [Sakana AI Scientist v2 — workshop paper accepted at ICLR](https://github.com/SakanaAI/AI-Scientist-v2)
- [Intrinsic Memory Agents (OpenReview)](https://openreview.net/forum?id=UbSUxAK3BI)
- [A-RAG: Hierarchical Agentic RAG (arXiv:2602.03442)](https://arxiv.org/html/2602.03442v1)
- [AI-Researcher (NeurIPS 2025, HKU)](https://arxiv.org/html/2505.18705v1)
- `AGI_LAB_ORG_DIAGNOSIS.md` (HIVE's external diagnosis, 2026-04-16)
- `data/system_redesign.md` (first internal restructuring, 2026-04-15)
- `docs/superpowers/specs/2026-04-13-agent-organization-design.md` (first org spec)
- HIVE lab rate-limit handling (private; summarized in Section 7.2)

---

## Appendix B — The ~130 Sub-Steps (reference)

**P1**: (1a) trigger ID, (1b) question drafting, (1c) importance assessment, (1d) tractability check, (1e) prior-work check, (1f) refinement, (1g) sub-question decomposition, (1h) program scoping, (1i) PI+Director lock, (1j) publish.

**P2**: (2a) query formulation, (2b) paper hunt, (2c) relevance filter, (2d) importance rank, (2e) deep reading, (2f) per-paper digest, (2g) citation graph, (2h) expertise map, (2i) gap analysis, (2j) methodology inventory, (2k) metric inventory, (2l) synthesis.

**P3**: (3a) relevant-theory ID, (3b) model application, (3c) prediction derivation, (3d) boundary conditions, (3e) unknown-unknowns, (3f) scaling-law analysis, (3g) symbolic formalization, (3h) write-up.

**P4**: (4a) divergent brainstorm, (4b) formalization, (4c) importance rank, (4d) feasibility rank, (4e) falsifiability check, (4f) orthogonality check, (4g) risk analysis, (4h) documentation.

**P5**: (5a) response-var, (5b) indep-var, (5c) controls, (5d) baseline selection, (5e) measurement theory, (5f) statistical power, (5g) confound ID, (5h) confound mitigation, (5i) ablation plan, (5j) sequence, (5k) resource budget, (5l) failure-mode plan.

**P6**: (6a) outcome-interpretation map, (6b) kill criteria, (6c) success criteria, (6d) PI+Director+auditor sign-off, (6e) amendment protocol.

**P7**: (7a) apparatus spec, (7b) tooling gap ID, (7c) measurement tools, (7d) profiling tools, (7e) data pipeline, (7f) reproducibility setup, (7g) SOTA integration, (7h) apparatus validation smoke.

**P8**: (8a) launch protocol, (8b) pre-run validation, (8c) active monitoring, (8d) mid-run diagnostics, (8e) kill-signal enforcement, (8f) checkpoint management, (8g) failure handling, (8h) completion protocol.

**P9**: (9a) raw-data validation, (9b) cleaning, (9c) descriptive stats, (9d) inferential stats, (9e) effect size, (9f) confidence intervals, (9g) ablation analysis, (9h) cross-seed aggregation, (9i) robustness checks, (9j) exploratory viz, (9k) write-up.

**P10**: (10a) observed-behavior enumeration, (10b) candidate mechanism generation, (10c) first-principles derivation, (10d) mechanism testing, (10e) alternative-mechanism comparison, (10f) selection, (10g) write-up.

**P11**: (11a) metric-claim alignment, (11b) construct validity, (11c) external validity, (11d) measurement-noise estimation, (11e) correction recommendations.

**P12**: (12a) red-team assignment, (12b) statistical review, (12c) pre-reg compliance audit, (12d) reproducibility audit, (12e) mechanism-plausibility review, (12f) alternative-explanation generation, (12g) limitations enumeration, (12h) write-up, (12i) responses.

**P13**: (13a) outline, (13b) abstract, (13c) intro, (13d) related-work, (13e) method, (13f) results, (13g) discussion, (13h) limitations, (13i) conclusion, (13j) figures, (13k) tables, (13l) appendix.

**P14**: (14a) PI review, (14b) Director review, (14c) unanimous-compromise iterations, (14d) final polish.

**P15**: (15a) raw data archive, (15b) finding extraction, (15c) shared knowledge update, (15d) bibliography update, (15e) killed-ideas update, (15f) knowledge-graph link, (15g) next-program candidates, (15h) program retro.

**Total: ~130 sub-steps.** Program uses a subset per phase based on relevance.

---

*End of design document.*
