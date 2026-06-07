# AGI Lab Agent Organization Design

**Date:** 2026-04-13
**Goal:** Fully autonomous AI agent organization that self-directs until it builds an AGI that beats Opus 4.6 on standardized benchmarks.
**Hardware:** MacBook Pro M3 Pro — 12 CPU cores, 18 GPU cores, 18GB unified RAM, ~233GB disk
**Constraints:** No cloud compute, no frameworks, everything from scratch, everything local. iCloud Drive available for storage.
**Operating model:** User (Harshith) sets the vision and only reviews the final result. Agents operate fully autonomously, including pivoting on dead ends.
**Model:** Claude Opus 4.6 (1M context) for all agents, max effort. No fallback to smaller models.
**Time:** Not a constraint. Quality over everything. The lab runs as long as it takes.
**Data strategy:** Chunked streaming — download data, process/train, delete, next chunk. iCloud for backup/overflow. No dataset size limit.

---

## Success Criteria

Beat Opus 4.6 on all of the following standardized benchmarks:

- MMLU / MMLU-Pro — broad knowledge and reasoning
- HumanEval / MBPP — code generation
- ARC-AGI — abstract reasoning
- GSM8K / MATH — mathematical reasoning
- HellaSwag — commonsense reasoning
- TruthfulQA — factual accuracy
- WinoGrande — linguistic reasoning
- BigBench-Hard — diverse hard tasks

---

## Organization Structure

### Director Agent (1 agent)

**Role:** Top-level autonomous orchestrator. The only agent that communicates results to the user.

**Responsibilities:**
- Maintains the master roadmap and milestone tracker
- Allocates agent sessions across divisions based on priority
- Receives cycle synthesis reports from each Chief
- Makes go/no-go decisions on research directions
- Triggers pivots when a stream hits a dead end
- Monitors benchmark progress against Opus 4.6 targets
- Manages token budget and session efficiency

**Decision authority:**
- Can kill a research stream if it's not producing results after N cycles
- Can spin up new experimental streams
- Can reallocate engineering resources between teams
- Can promote a research finding to "implement this now"
- Escalates to user ONLY if: the project beats the benchmark, or a catastrophic constraint is hit (e.g., 18GB is provably insufficient for any viable approach)

**Inputs (read every cycle):**
- `data/eval/scorecard.md` — where we stand vs Opus 4.6
- `data/research/synthesis_log.md` — Chief Scientist's latest synthesis
- `data/engineering/tech_plan.md` — Chief Engineer's status and blockers
- `data/engineering/memory_budget.md` — are we within 18GB?
- `data/contradictions.md` — unresolved cross-team conflicts
- `data/infra/thermal_log.md` — hardware health
- `data/infra/disk_budget.md` — storage pressure
- `data/shared_knowledge.md` — the full knowledge bus

**Outputs (written each cycle):**
- `data/roadmap.md` — updated priorities, resource allocation, active/killed streams
- `data/director_log.md` — decision journal: what was decided, why, what evidence
- `data/benchmark_tracker.md` — trend analysis, projected timeline
- Directives to Chief Scientist and Chief Engineer — specific asks for next cycle

**Director ↔ Chief Scientist:**
- Receives: research synthesis, pivot proposals, "this is ready for engineering" flags, resource requests
- Sends: priority rankings ("focus on X, deprioritize Y"), go/no-go on pivots, "investigate this benchmark gap" directives
- Conflict resolution: if Chief Scientist and Chief Engineer disagree on feasibility, Director arbitrates

**Director ↔ Chief Engineer:**
- Receives: feasibility reports, memory budget status, performance regressions, "this can't be built as designed" flags
- Sends: "implement this finding" directives with priority, resource allocation decisions, timeline pressure
- Engineering veto: if Chief Engineer says "this blows 18GB," Director must either adjust scope or kill the feature — not override

**Director ↔ Benchmark Lead:**
- Receives: scorecard after every eval run, regression alerts, victory declaration
- Sends: "run a focused eval on X" requests, benchmark priority adjustments
- Scorecard is the Director's north star — every decision traces back to "does this move the benchmarks?"

**Director ↔ Red Team:**
- Receives: critical failure reports, adversarial findings
- Sends: "stress test this specific capability" directives when a benchmark gap is unclear
- Red Team can escalate directly to Director, bypassing Benchmark Lead, for catastrophic findings

**Director ↔ Knowledge Agent:**
- Receives: contradiction alerts, knowledge gap analysis, "teams are duplicating work" warnings
- Sends: "resolve this contradiction, it's blocking progress" priority flags
- Knowledge Agent is the Director's peripheral vision — catches things the Chiefs might miss

**Director ↔ DevOps:**
- Receives: build health, thermal warnings, disk pressure alerts
- Sends: "pause low-priority agents, laptop is overheating" throttle commands
- DevOps thermal/disk alerts can trigger automatic Director intervention — no waiting for next cycle

**Decision cadence:**
- **Every cycle:** Read all inputs, update roadmap, issue directives
- **On-demand:** Respond to escalations (Red Team catastrophic, Engineering veto, thermal throttle)
- **Milestone events:** Major pivot decisions, research stream kill/create, resource reallocation
- **Victory:** Compile final report, notify user

**Autonomous pivot decision framework:**

1. **Signal:** Benchmark stall for N cycles, or team flags dead end, or Ablation shows no gains
2. **Evidence gathering:** Director requests Chief Scientist's analysis + Ablation data + Engineering feasibility
3. **Options:** Chief Scientist proposes 2-3 alternatives with tradeoffs
4. **Decision:** Director picks based on: benchmark impact potential, engineering cost, memory budget fit, time already invested
5. **Execution:** Reallocate agents, update roadmap, Knowledge Agent archives the dead stream's learnings
6. **Log:** Full reasoning written to `data/director_log.md` so future cycles can learn from past pivots

---

### Research Division (~10-13 agents)

#### Chief Scientist Agent

**Role:** Owns all research strategy. Translates Director's goals into concrete research hypotheses. Synthesizes findings across all research teams into architectural decisions.

**Responsibilities:**
- Breaks Director's milestones into research hypotheses
- Assigns hypotheses to the right team
- Identifies when findings from one team unlock progress in another
- Writes cycle synthesis for Director: what we learned, what it means, what to do next
- Proposes pivots when theory hits walls
- Decides when a research finding is mature enough to hand to Engineering

**State:**
- `data/research/strategy.md` — current research priorities and open questions
- `data/research/synthesis_log.md` — cross-team insights and connections

#### Math / Information Theory Team (2-3 agents)

- Optimal representation theory — most efficient way to encode knowledge in 18GB
- Compression bounds — theoretical limits on what's achievable at our scale
- Loss function design — from first principles, not borrowed from transformers
- Optimization theory — convergence guarantees for novel architectures
- Entropy and mutual information analysis for architecture decisions

#### Neuroscience Team (2-3 agents)

- Free Energy Principle — Friston's active inference as a learning framework
- Thousand Brains Theory — Hawkins' cortical columns, reference frames, voting
- Predictive Coding — hierarchical prediction error minimization
- Memory systems — hippocampal replay, complementary learning systems, sleep consolidation
- Attention and consciousness models — what's computationally useful vs philosophical fluff

#### Physics Team (2-3 agents)

- Thermodynamic computing — Landauer's principle, reversible computation, energy-efficient learning
- Statistical mechanics of learning — phase transitions in training, spin glass analogies
- Symmetry and invariance — what symmetries should the architecture respect
- Dynamical systems — attractors, stability, chaos as computational resource
- Quantum-inspired classical algorithms — tensor networks, amplitude encoding on classical hardware

#### Novel Compute Team (2-3 agents)

- Hyperdimensional computing — high-dimensional sparse representations
- Neuromorphic paradigms — spike-based computation adapted for Metal GPU
- Memory-centric architectures — exploiting M3 Pro's unified memory (CPU/GPU share same RAM)
- Sparse + structured computation — not dense matmuls, something fundamentally different
- Mixture/modular approaches — composable specialist modules vs monolithic models

#### Literature Review Agents (1-2 agents)

**Two mandates — assigned tasks AND proactive scanning every cycle:**

Assigned tasks:
- Complete specific research tasks from Chief Scientist or Director
- Research Opus 4.6 benchmark scores as targets

Proactive scanning (EVERY session):
- Continuous arxiv scanning via `arxiv_reader.py` + web search
- Quantization advances (TurboQuant, GPTQ, AWQ, QuIP#, AQLM, any era)
- Apple Silicon / Metal optimization techniques
- Sparse computation breakthroughs (structured sparsity, N:M, dynamic sparsity)
- Training efficiency (curriculum learning, data mixing, multi-token prediction, new optimizers)
- Small model techniques (distillation, pruning, weight sharing, parameter-efficient methods)
- MoE advances (routing, expert merging, load balancing, scaling laws)
- Predictive coding at scale (any PCN papers >100M params)
- Novel architectures that could outperform our current design
- Memory-efficient inference and training techniques
- New benchmarking tools and evaluation methods

Output:
- No era bias — search across ALL years. A 1990s neuroscience insight or 2010 information theory result could be as critical as a 2026 paper. Search for the BEST work, not the newest.
- PHASE-DEPENDENT FILTER: During research/exploration or pivot phases → search wide, bring alternative architectures and novel paradigms. During committed/engineering/training phases → search narrow, only surface findings that help the current approach. No shiny object syndrome when building. Check `roadmap.md` for current phase.
- Sole owner of `data/bibliography.md` — no other agent writes to it
- Digest papers into actionable summaries: what, why it matters for us, which team should know
- Flag URGENT breakthroughs that could change architecture or engineering to Director immediately
- Evaluate training data sources (Hugging Face, Common Crawl, Wikipedia, code/math datasets)
- Feed into `research_digest.py` for cross-stream synthesis

All research and engineering teams read `bibliography.md` before starting work each session.

---

### Engineering Division (~17 agents)

#### Chief Engineer Agent

**Role:** Owns all implementation. Translates research findings into working, tested, optimized code that runs within 18GB on M3 Pro.

**Responsibilities:**
- Receives "implement this" directives from research via Director
- Breaks architectural decisions into engineering tasks
- Owns code quality, performance, and memory budgets
- Maintains the technical debt log
- Signs off on every merge into the core codebase
- Reports back to Research: feasibility, performance characteristics, hard constraints

**State:**
- `data/engineering/tech_plan.md` — current implementation priorities
- `data/engineering/memory_budget.md` — who gets how many MB of the 18GB
- `data/engineering/perf_log.md` — benchmark results over time

#### Tensor / Compute Team (2-3 agents)

- Custom tensor library in C — allocation, indexing, broadcasting, views
- Memory pool management — zero-copy where possible, explicit lifetime tracking
- BLAS-level operations — matmul, elementwise, reductions, all tuned for M3
- Automatic differentiation engine — backward pass, gradient accumulation
- Data types — fp16, bf16, int8, custom quantization schemes

#### Metal Kernel Team (3 agents)

- GPU compute shaders for all hot-path operations
- Threadgroup optimization for M3 Pro's 18 GPU cores
- Unified memory exploitation — no CPU-GPU copies, shared pointers
- Custom kernels for whatever novel operations Research invents
- Profiling with Metal System Trace, optimizing occupancy
- Works tightly with Tensor team — kernels must match tensor memory layout

#### Architecture Team (4 agents)

- Model implementation — whatever Research designs, these agents build it
- Training loop — forward, backward, optimizer step, checkpointing
- Inference engine — optimized path for evaluation and deployment
- Tokenizer — custom, from scratch, designed for the specific architecture
- Data pipeline — chunked streaming: download chunk, preprocess, train, delete, next chunk
- iCloud checkpoint backup integration
- Memory-mapped data loading for chunks that fit, streaming for chunks that don't

#### Swift Bridge Team (2 agents)

- Metal-C interop via Swift
- Metal shader compilation and dispatch from C
- Device capability queries
- Eventually: macOS app wrapper for the assistant interface

#### Optimization Team (2-3 agents)

- Profiling — CPU, GPU, memory, thermal throttling via `hwmon.py`
- Quantization — post-training and quantization-aware training
- Pruning and sparsity — if the architecture supports it
- Fitting everything in 18GB — this team has veto power on architecture decisions that blow the budget
- Benchmarking via `benchmark.py` — tracks performance regressions
- Memory-mapped data loading, streaming training for large datasets

#### Code Review Team (2 agents)

- Reviews EVERY piece of code before it enters the codebase — has BLOCK power
- Memory safety: buffer overflows, use-after-free, double-free, uninitialized reads
- Undefined behavior: signed overflow, null dereference, alignment violations
- Memory leaks: every allocation has a matching free, pool lifetimes are correct
- Numerical correctness: floating point precision, quantization accuracy
- Silent breaks: code that compiles and runs but produces wrong results
- API contract violations: does the code match header contracts?
- Performance: unnecessary copies, missed zero-copy, suboptimal threadgroup sizes
- 18GB budget enforcement
- Builds with sanitizers (ASan + UBSan) and runs sanitizer-instrumented tests
- Uses `/coderabbit` for automated review + manual review for logic
- Nothing merges without Code Review approval

#### QA / Testing Team (1 agent)

- Integration tests: do tensor ops + Metal kernels + memory pool work together?
- End-to-end tests: can we forward-pass a dummy input through the full model?
- Sanitizer runs: ASan/UBSan/TSan on full test suite
- Regression testing: after every code change, verify nothing previously working is broken
- Stress tests: operations at memory limits (near 18GB), graceful behavior
- Numerical accuracy: compare C results against Python/SymPy reference implementations via `verify.py`
- Edge cases: empty tensors, zero-size dimensions, maximum tensor sizes, NaN/Inf
- Reports to Chief Engineer: test pass rate, sanitizer findings, regression count

---

### Evaluation Division (~5-7 agents)

#### Benchmark Lead Agent

**Role:** Owns the definition of "done." Single source of truth on how good the system is.

**Responsibilities:**
- Builds and maintains the benchmark harness — automated, reproducible, versioned
- Tracks Opus 4.6's known scores across all target benchmarks
- Runs evals after every significant model change
- Publishes scorecards visible to all agents
- Flags regressions
- Declares victory when all benchmarks are surpassed

**State:**
- `data/eval/scorecard.md` — latest scores vs Opus 4.6 targets
- `data/eval/regression_log.md` — what broke and when
- `data/eval/benchmark_history.db` — full history of every eval run

#### Red Team (2-3 agents)

- Adversarial testing — find inputs that break the model
- Prompt injection resistance, hallucination detection, edge cases
- Stress testing under memory pressure at 18GB limit
- Robustness testing — noise, distribution shift, out-of-domain inputs
- Every failure filed as structured report to the responsible team

#### Ablation Team (1-2 agents)

- After every architecture change: what's actually helping?
- Remove components one at a time, measure impact
- Prevents complexity bloat — if a module doesn't improve benchmarks, flagged for removal
- Tracks which Research ideas translated to real gains vs sounded good on paper
- Feeds back to Chief Scientist with honest "worth it?" analysis

---

### Infrastructure Division (2 agents)

#### DevOps Agent

**Role:** Keeps the machine running. Everyone else builds the AGI — this agent makes sure they can.

**Responsibilities:**
- Build system maintenance — Makefile, compilation, linking, Metal shader compilation
- CI pipeline — automated test runs on every code change
- Experiment tracking integrity — `experiments.db` stays clean
- Checkpoint management — versioned snapshots locally, backup to iCloud, disk budget enforcement
- Data pipeline management — chunked downloads, local cleanup after training, iCloud caching of processed data
- Thermal management — monitors `hwmon.py`, throttles workloads if laptop overheats
- Disk space monitoring — warn at 50GB free, alert Director at 20GB free
- iCloud sync monitoring — ensure backups complete before local deletion

**State:**
- `data/infra/build_status.md` — what compiles, what's broken
- `data/infra/disk_budget.md` — storage allocation and usage
- `data/infra/thermal_log.md` — overheating incidents and throttle events

#### Knowledge Agent

**Role:** The librarian. Owns the shared knowledge bus, ensures no insight gets lost.

**Responsibilities:**
- Maintains `data/shared_knowledge.md` — curated, organized, actionable
- Runs `research_digest.py` after every research cycle
- Maintains structured knowledge graph: concepts -> findings -> experiments -> code
- Deduplicates findings across teams
- Detects gaps and contradictions
- Archives stale knowledge
- Reads `bibliography.md` (owned by Literature team) to detect gaps and contradictions — flags missing papers back to Literature team, does NOT write to it directly

**State:**
- `data/shared_knowledge.md` — the bus, always current
- `data/knowledge_graph.md` — structured concept map
- `data/contradictions.md` — unresolved conflicts between teams
- `data/bibliography.md` — papers indexed by relevance and stream

---

## Cross-Communication System

### 1. Shared Knowledge Bus (passive, always-on)

- `data/shared_knowledge.md` — generated by `research_digest.py`
- Every agent reads this before starting any work session
- Updated after every significant finding from any team
- Contains: latest findings, open questions, active hypotheses, contradictions

### 2. Cross-Team Seminars (structured, periodic)

- After each research cycle, Chief Scientist runs a synthesis session
- Findings from all teams presented together
- Explicitly asks: "Does Team X's finding change anything for Team Y?"
- Output: updated shared knowledge + new cross-team hypotheses

### 3. Direct Collaboration Channels (ad-hoc, on-demand)

- Any agent can tag a finding as `cross-ref: [team]` in experiments.db
- Tagged team gets the finding in their next session briefing
- Two agents can be spun up as a pair to work a specific cross-domain question

### 4. Contradiction Board

- Dedicated section: "Things that don't add up yet"
- When Team A's finding conflicts with Team B's, it goes here
- Chief Scientist prioritizes resolving contradictions — often where breakthroughs hide

### 5. Shared Experiment Visibility

- All experiments in `experiments.db` visible to all agents
- Cross-reference table traces lineage of ideas across streams
- Any agent can run `research_digest.py` to see the full picture

### 6. Engineering-Research Feedback Loop

- Engineering reports feasibility constraints back to Research
- Research adjusts theory based on what's actually feasible on the hardware
- Continuous bidirectional dialogue, not one-way handoff

### 7. Eval-Everyone Loop

- Scorecard visible to all — everyone knows where we stand
- Red Team findings routed directly to responsible team
- Ablation results inform Director pivot decisions
- Research can challenge eval methodology
- Engineering provides hardware context for benchmark interpretation

### 8. Within-Division Communication

**Within Research:**
- All teams log experiments with stream tags and cross-references
- Chief Scientist reads all streams before writing synthesis
- "This changes everything" flag triggers emergency Director review
- Literature agents push papers directly to relevant teams

**Within Engineering:**
- Tensor and Metal Kernel teams share `data/engineering/tensor_metal_spec.md`
- Architecture team reads Tensor/Metal perf benchmarks before designing
- Optimization team reviews all code, can block merges on memory budget violations
- Swift Bridge syncs with Metal Kernel on dispatch patterns
- Code Review team reviews ALL code — has BLOCK power, nothing merges without approval
- QA team runs integration tests, sanitizers, regression tests after Code Review approves
- All engineering teams read `data/bibliography.md` before starting work for latest techniques
- Chief Engineer runs standups each cycle

**Within Evaluation:**
- Red Team finds what breaks, Ablation finds why
- All three share the scorecard as single scoreboard
- Benchmark Lead can block checkpoints on regression

---

## Agent Inventory

| Division | Role | Agent Count |
|----------|------|-------------|
| Director | Director | 1 |
| Research | Chief Scientist | 1 |
| Research | Math / Information Theory | 2 |
| Research | Neuroscience | 2 |
| Research | Physics | 2 |
| Research | Novel Compute | 2 |
| Research | Literature Review | 1 |
| Engineering | Chief Engineer | 1 |
| Engineering | Tensor / Compute | 2 |
| Engineering | Metal Kernel | 3 |
| Engineering | Architecture | 4 |
| Engineering | Swift Bridge | 2 |
| Engineering | Optimization | 2 |
| Engineering | Code Review | 2 |
| Engineering | QA / Testing | 1 |
| Evaluation | Benchmark Lead | 1 |
| Evaluation | Red Team | 2 |
| Evaluation | Ablation | 1 |
| Infrastructure | DevOps | 1 |
| Infrastructure | Knowledge Agent | 1 |
| **Total** | | **~38** |

All agents run Claude Opus 4.6 at max effort. No exceptions.

---

## Shared State / File System

```
data/
├── roadmap.md                    # Director — master plan
├── director_log.md               # Director — decision journal
├── benchmark_tracker.md          # Director — scores vs Opus 4.6
├── shared_knowledge.md           # Knowledge Agent — the bus
├── knowledge_graph.md            # Knowledge Agent — concept map
├── contradictions.md             # Knowledge Agent + Chief Scientist
├── bibliography.md               # Literature team (sole owner) — all agents read
├── experiments.db                # All agents — experiment tracker
├── research/
│   ├── strategy.md               # Chief Scientist — priorities
│   └── synthesis_log.md          # Chief Scientist — cross-team insights
├── engineering/
│   ├── tech_plan.md              # Chief Engineer — implementation priorities
│   ├── memory_budget.md          # Chief Engineer — 18GB allocation
│   ├── perf_log.md               # Chief Engineer — benchmarks over time
│   └── tensor_metal_spec.md      # Tensor + Metal Kernel shared spec
├── eval/
│   ├── scorecard.md              # Benchmark Lead — latest scores
│   ├── regression_log.md         # Benchmark Lead — what broke
│   ├── benchmark_history.db      # Benchmark Lead — full eval history
│   └── ablation_log.md           # Ablation Team — what works
├── infra/
│   ├── build_status.md           # DevOps — compilation status
│   ├── disk_budget.md            # DevOps — storage usage
│   └── thermal_log.md            # DevOps — overheating events
├── checkpoints/                  # Versioned model snapshots (latest N local, rest in iCloud)
├── digests/                      # Research digests
├── papers/                       # Cached arxiv PDFs
├── plots/                        # Generated visualizations
├── experiments/                  # Raw experimental data
│   └── exp_<ID>/                 # One directory per experiment — scripts, raw output, traces
├── agents/                       # Per-agent persistent state
│   ├── agents.json               # All agent role definitions for --agents flag
│   ├── director_prompt.md        # Director master prompt
│   └── <agent_name>/             # state.md + journal.md (or decisions.md) per agent
├── training/
│   ├── active/                   # Current chunk being trained on
│   ├── processed/                # Preprocessed chunks (cached, deletable)
│   └── manifest.md               # What chunks exist, where, status
└── icloud -> ~/Library/Mobile Documents/com~apple~CloudDocs/AGI/  # Symlink to iCloud
```

---

## Data Strategy

### Principles

- **18GB RAM is the hard constraint.** Disk and time are not.
- **Quality over speed.** If training on a larger, better dataset takes 10x longer, do it.
- **Chunked streaming.** No dataset needs to fit on disk all at once.

### Storage Tiers

| Tier | Location | Size | Speed | Use For |
|------|----------|------|-------|---------|
| Hot | Local SSD (~233GB) | Limited | Fast | Active training data, current checkpoint, working files |
| Warm | iCloud Drive | Expandable | Medium | Checkpoint backups, processed datasets, experiment archives |
| Cold | Internet (arxiv, HF, etc.) | Unlimited | Slow | Raw datasets, papers, pre-download staging |

### Chunked Data Pipeline

Training data is processed in chunks to work within disk constraints:

```
1. Download chunk N of dataset to data/training/active/
2. Preprocess and tokenize chunk N
3. Train on chunk N (multiple epochs if needed)
4. Save checkpoint to data/checkpoints/ (and backup to iCloud)
5. Delete chunk N from local disk
6. Download chunk N+1
7. Repeat until dataset exhausted
```

This means we can train on datasets of any size — terabytes if needed — just streaming through them over time. The Architecture team owns the data pipeline implementation. The DevOps agent manages disk budget to ensure we never fill the SSD.

### iCloud Integration

- **Checkpoint backups:** After every significant training milestone, copy checkpoint to iCloud. If local disk fills up, older local checkpoints can be deleted — they're safe in iCloud.
- **Processed data cache:** Once a dataset chunk is preprocessed, the processed version can be stored in iCloud and re-fetched faster than re-downloading + re-processing.
- **Experiment archives:** Old experiment data that's no longer actively needed but might be useful later gets moved to iCloud.
- **iCloud path:** `~/Library/Mobile Documents/com~apple~CloudDocs/AGI/` (or symlinked to `data/icloud/`)

### Disk Budget (updated)

The DevOps agent manages disk allocation:

| Component | Budget | Notes |
|-----------|--------|-------|
| Active training chunk | ~50 GB | Current chunk being trained on |
| Checkpoints (local) | ~50 GB | Last N checkpoints, rest in iCloud |
| Model + build artifacts | ~20 GB | Compiled code, metallib, etc. |
| Papers + bibliography | ~10 GB | Cached arxiv PDFs |
| Experiments + state | ~5 GB | SQLite DBs, agent state, logs |
| OS + headroom | ~50 GB | Never touch this |
| **Free buffer** | **~48 GB** | Emergency headroom |

DevOps warns at 50GB free, alerts Director at 20GB free.

### Data Sources (for Architecture/Literature teams to evaluate)

Potential training data sources to investigate:
- Hugging Face Hub (user authenticated as Drogon4231)
- arxiv papers (via arxiv_reader.py)
- Common Crawl subsets
- Wikipedia dumps
- Code datasets (The Stack, etc.)
- Math datasets (for GSM8K/MATH benchmark targeting)
- Reasoning datasets (for ARC-AGI targeting)

Literature team evaluates data quality. Architecture team designs the pipeline. Optimization team ensures chunks fit in memory during training.

---

## Research Verification Rule

No research hypothesis is "confirmed" based on theoretical reasoning alone:
1. Research team proposes hypothesis with theoretical analysis → status: **THEORIZED**
2. Engineering builds a minimal code test that measures the claim → status: **TESTED**
3. Only if code produces numbers matching the theory → status: **CONFIRMED**

Research reasoning sets direction. Only code produces truth.

## Micro-Experiment Rule

Before committing to full-scale implementation of any architecture or technique:
1. Build a tiny version (1M-10M params) and verify the entire pipeline end-to-end
2. Train on small dataset, measure actual loss curves, verify convergence
3. Only after micro-experiment produces real numbers → scale up

Applies to: HSPA architecture, iPC training, FEP routing, quantization. Fail fast and small.

---

## Decision Flow

1. Director sets cycle goals based on scorecard + roadmap
2. Chief Scientist breaks goals into research hypotheses, assigns to teams
3. Research teams investigate, log to experiments.db, surface findings to knowledge bus
4. Chief Scientist synthesizes, runs cross-team seminar, identifies connections
5. Mature findings get promoted: Chief Scientist -> Director -> Chief Engineer
6. Chief Engineer breaks into engineering tasks, assigns to teams
7. Engineering writes code
8. **Code review gate (BEFORE anything runs on new code)**: Code Review team reviews → QA runs sanitizers + integration tests → only THEN is code merged and available for experiments/training/eval. Nothing runs on unreviewed code.
9. Engineering reports feasibility back to Research
10. Eval runs benchmarks after every significant change (on reviewed, tested code only)
9. Scorecard updates -> Director adjusts priorities for next cycle
10. Dead ends trigger pivot: team flags -> Chief -> Director decides -> reallocation

**Pivot protocol:** When a research stream fails:
1. Team documents why it failed and what was learned
2. Chief Scientist proposes alternatives
3. Director evaluates: kill the stream, merge it into another, or try a different angle
4. Knowledge Agent archives the findings — dead ends are still knowledge
5. Freed agents get reallocated to highest-priority work

**Victory protocol:** When all benchmarks are surpassed:
1. Benchmark Lead declares victory with full scorecard
2. Red Team runs final adversarial suite
3. Ablation Team confirms no component is dead weight
4. Director compiles final report
5. Director notifies user (Harshith)

---

## Session Continuity System

### Core Principle

Every session is a fresh boot. No session assumes anything from a previous conversation. All state lives on disk. A session reads disk, figures out what to do, does work, writes to disk, dies. Next session does the same. The organization's progress is monotonically forward — sessions are disposable, the work is not.

### Atomic Work Units

All work is broken into small, atomic tasks that can complete within ~10% of a session's capacity. No agent accumulates large amounts of reasoning in memory without writing to disk.

```
BAD:  "Investigate Free Energy Principle" (huge, unrecoverable if interrupted)
GOOD: "Test hypothesis: FEP convergence under sparse representations with N=1000"
       → Result written to disk immediately on completion
```

### Progressive Journaling

Agents think on disk, not in context. Every agent writes a working journal as it goes:

```
data/agents/neuro_team_1/journal.md

## Task: FEP hypothesis #7
Started: 2026-04-15T03:22:00
Approach: Testing convergence of predictive coding with sparse codes
Step 1: Set up simulation parameters — DONE
  - Used 512-dim sparse vectors, sparsity 0.05
  - Result: initial loss 4.2
Step 2: Run convergence test for 1000 iterations — DONE
  - Converged at iteration 340, final loss 0.08
  - Surprising: converged faster with sparser representations
Step 3: Compare against dense baseline — IN PROGRESS
  - Dense baseline running...
  [SESSION DIED HERE]
```

Next session reads this journal, sees Step 3 was in progress, resumes from "run dense baseline comparison."

### Decision Ledgers

Every decision gets logged with full reasoning so future sessions understand not just what was decided but why:

```
data/agents/chief_scientist/decisions.md

## Decision: Prioritize FEP over Thousand Brains for cycle 3
Date: 2026-04-15
Reason: Math team's compression bounds (exp #42) suggest hierarchical
  prediction is more memory-efficient than voting-based architectures.
  At 18GB, we can't afford the redundancy Thousand Brains requires.
Alternatives considered:
  - Thousand Brains: rejected, memory cost too high
  - Hybrid: deferred to cycle 4 if FEP pans out
Reversible: yes, Thousand Brains stays in backlog
```

### Per-Agent State Files

Each agent role has a persistent state file that survives session boundaries:

```
data/agents/[role]/state.md

agent: neuroscience_team_1
status: IN_PROGRESS
current_task: "FEP hypothesis #7 — sparse convergence"
task_id: 47
journal: data/agents/neuro_team_1/journal.md
blocked_by: nothing
outputs_so_far:
  - experiments.db #47 (partial)
  - data/research/neuro/fep_h7_partial.md
next_step: "Run dense baseline comparison (Step 3 of journal)"
context_needed: "Read journal.md steps 1-2 for parameters and partial results"
```

### Orchestration State

The master state file that every session reads first:

```
data/session_state.md

cycle: 3
phase: RESEARCH
status: RUNNING
launched_agents:
  - math_team: COMPLETED (experiments #42-45)
  - physics_team: COMPLETED (experiments #46-48)
  - neuro_team_1: IN_PROGRESS (task #47, step 3)
  - neuro_team_2: NOT_STARTED (task #48)
  - novel_compute: NOT_STARTED
pending_after_research:
  - engineering: implement top finding
  - eval: benchmark new checkpoint
next_action: "Resume Neuroscience team hypothesis #7"
context_for_next_session: "FEP team was testing whether predictive coding
  converges under sparse representations. Preliminary results in
  data/research/neuro/fep_h7_partial.md. Continue from there."
```

### Cycle Queue

The prioritized work queue that persists across sessions:

```
data/cycle_queue.md

## Current Cycle: 3
Priority order (set by Director):
1. [IN PROGRESS] Research: complete all team investigations
2. [BLOCKED] Engineering: implement top research finding from cycle 2
3. [BLOCKED] Eval: run benchmark after engineering delivers
4. [PENDING] Director: cycle 3 review and cycle 4 planning

## Backlog
- Red Team: adversarial test on cycle 2 model checkpoint
- Literature: scan arxiv for sparse coding papers (flagged by Physics team)
```

### Raw Experiment Data Preservation

ALL agents must save raw experimental data, not just summaries. Every experiment gets its own directory:

```
data/experiments/exp_<ID>/
├── README.md          # What this experiment tested, parameters, conclusion
├── script.py          # The script/code that ran the experiment (if any)
├── raw_output.txt     # Full raw output
├── convergence.csv    # Convergence logs, training curves, etc.
├── profiling/         # Metal traces, memory profiles, timing data
└── figures/           # Any generated plots
```

The journal and `experiments.db` contain summaries that reference this directory. The raw data must be sufficient to reproduce the experiment. Nothing gets summarized away.

---

### Graceful Shutdown Protocol

The session monitors its own resource usage. When approaching limits:

1. Stop launching new subagents
2. Let running subagents finish their current atomic task (they're small by design)
3. Each subagent writes its final state to its state file and journal
4. Director writes updated `data/session_state.md`
5. Write exit reason to `data/session_exit.md`
6. Session exits cleanly

**Exit reason file (`data/session_exit.md`):**

The session writes why it's exiting so the wrapper script knows how long to wait:

```
reason: CONTEXT_FULL | RATE_LIMIT | GRACEFUL_CHECKPOINT
timestamp: 2026-04-15T03:22:00
details: "Context at 85%, completed 3 of 5 research tasks this cycle"
```

Exit reasons:
- `CONTEXT_FULL` — session used most of its context window. No rate limit issue. Restart immediately.
- `RATE_LIMIT` — session detected throttling or 429 responses. Need to wait and probe.
- `GRACEFUL_CHECKPOINT` — session completed a clean unit of work and exited on its own. Restart immediately for next unit.

If a hard kill happens (rate limit mid-execution):
- `session_exit.md` may not get written — wrapper treats missing file as unknown exit, uses probe
- Worst case: one atomic task's result is lost (not the whole session's work)
- The journal shows where it was, the state file shows what was completed
- Next session re-runs that one small task

### Resume Protocol

```
New session starts
  → Read data/session_state.md
  → If doesn't exist: first run, start from Director initial planning
  → If exists:
      → For each agent with status COMPLETED: skip, results on disk
      → For each agent with status IN_PROGRESS: read state.md + journal.md, resume from next_step
      → For each agent with status NOT_STARTED: launch normally
      → Continue the cycle
```

### Degradation Budget

What's lost per session interruption:
- At most one atomic task needs re-running (~5 minutes of work)
- No decisions are ever lost (ledgers capture all reasoning)
- No completed work is ever lost (results written incrementally)
- Cross-agent coordination resets but state files capture all dependencies

---

## Automation: Hands-Off Continuous Execution

### Architecture

A tmux session runs a bash wrapper script that launches Claude Code sessions in a loop. The user starts it once. It runs forever until victory.

```
┌─────────────────────────────────────────────┐
│ tmux session: agi-lab                       │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │ run_agi_lab.sh (loop forever)       │    │
│  │                                     │    │
│  │  1. Launch claude session           │    │
│  │  2. Claude reads session_state.md   │    │
│  │  3. Claude runs org cycle           │    │
│  │  4. Claude checkpoints to disk      │    │
│  │  5. Session exits (limit/complete)  │    │
│  │  6. Script checks for VICTORY       │    │
│  │  7. If not: wait for limit refresh  │    │
│  │  8. Go to 1                         │    │
│  └─────────────────────────────────────┘    │
│                                             │
└─────────────────────────────────────────────┘
```

### The Wrapper Script: `run_agi_lab.sh`

```bash
#!/bin/bash
# run_agi_lab.sh — Autonomous AGI lab runner
# Start once, runs until victory or manual stop

LOG_DIR="data/infra/session_logs"
mkdir -p "$LOG_DIR"

SESSION_NUM=0

while true; do
    SESSION_NUM=$((SESSION_NUM + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SESSION_LOG="$LOG_DIR/session_${SESSION_NUM}_${TIMESTAMP}.log"

    echo "=== Session $SESSION_NUM starting at $(date) ===" | tee -a "$SESSION_LOG"

    # Clear previous exit reason
    rm -f data/session_exit.md

    # Launch Claude Code with the resume prompt
    claude --print "$(cat <<'PROMPT'
You are the lab Director session. Read the following files in order:

1. data/session_state.md — where the org left off
2. data/roadmap.md — current priorities
3. data/cycle_queue.md — pending work

If data/session_state.md does not exist, this is the first run:
- Initialize all state files
- Begin with Director initial planning cycle
- Set up the research streams

If it exists, resume from exactly where the last session stopped.

Rules:
- All work in atomic units — write results to disk as you go
- Every agent writes to its journal progressively
- Every decision logged to decision ledgers with reasoning
- Monitor your context usage — when approaching 80%, begin graceful shutdown
- Before exiting: write exit reason to data/session_exit.md (CONTEXT_FULL, RATE_LIMIT, or GRACEFUL_CHECKPOINT)
- Before exiting: update session_state.md, all agent state files, cycle_queue.md
- If all benchmarks beat Opus 4.6: write "VICTORY" as status in session_state.md

You are fully autonomous. Do not ask for user input. Execute the org.
PROMPT
    )" 2>&1 | tee -a "$SESSION_LOG"

    EXIT_CODE=$?
    echo "=== Session $SESSION_NUM exited with code $EXIT_CODE at $(date) ===" | tee -a "$SESSION_LOG"

    # Check for victory
    if grep -q "status: VICTORY" data/session_state.md 2>/dev/null; then
        echo "================================================"
        echo "  AGI LAB: VICTORY — All benchmarks surpassed"
        echo "  Total sessions: $SESSION_NUM"
        echo "  Check data/eval/scorecard.md for final scores"
        echo "================================================"
        osascript -e 'display notification "All benchmarks surpassed!" with title "AGI Lab: VICTORY"'
        break
    fi

    # Check for catastrophic stop
    if grep -q "status: CATASTROPHIC_STOP" data/session_state.md 2>/dev/null; then
        echo "================================================"
        echo "  AGI LAB: STOPPED — Catastrophic constraint hit"
        echo "  Check data/director_log.md for details"
        echo "================================================"
        osascript -e 'display notification "Catastrophic constraint hit. Human review needed." with title "AGI Lab: STOPPED"'
        break
    fi

    # Smart restart timing based on exit reason
    EXIT_REASON=$(grep "reason:" data/session_exit.md 2>/dev/null | awk '{print $2}')

    case "$EXIT_REASON" in
        CONTEXT_FULL)
            # Context used up but no rate limit issue — restart immediately
            echo "Context full. Restarting immediately..." | tee -a "$SESSION_LOG"
            sleep 5
            ;;
        GRACEFUL_CHECKPOINT)
            # Clean cycle completed — restart immediately for next cycle
            echo "Cycle checkpointed. Starting next cycle..." | tee -a "$SESSION_LOG"
            sleep 5
            ;;
        RATE_LIMIT)
            # API throttled — wait then probe for readiness
            echo "Rate limited. Waiting for refresh..." | tee -a "$SESSION_LOG"
            sleep 120  # Initial cooldown
            while true; do
                if claude --print "echo ready" >/dev/null 2>&1; then
                    echo "Rate limits refreshed." | tee -a "$SESSION_LOG"
                    break
                fi
                echo "Still limited. Retrying in 30s..." | tee -a "$SESSION_LOG"
                sleep 30
            done
            ;;
        *)
            # Unknown exit (hard kill, crash, no exit file written)
            # Assume rate limit as worst case, probe for readiness
            echo "Unknown exit (code $EXIT_CODE). Probing readiness..." | tee -a "$SESSION_LOG"
            sleep 60
            while true; do
                if claude --print "echo ready" >/dev/null 2>&1; then
                    echo "Ready to resume." | tee -a "$SESSION_LOG"
                    break
                fi
                echo "Not ready. Retrying in 30s..." | tee -a "$SESSION_LOG"
                sleep 30
            done
            ;;
    esac
done
```

### How to Start (one time, once, ever)

```bash
# Make executable
chmod +x run_agi_lab.sh

# Launch in detached tmux session
tmux new-session -d -s agi-lab './run_agi_lab.sh'
```

### How to Monitor (optional, whenever you want)

```bash
# Watch live
tmux attach -t agi-lab

# Check status without attaching
cat data/session_state.md
cat data/eval/scorecard.md
cat data/director_log.md

# Check how many sessions have run
ls data/infra/session_logs/ | wc -l
```

### How it Stops

Only three ways:
1. **VICTORY** — all benchmarks beat Opus 4.6. macOS notification sent.
2. **CATASTROPHIC_STOP** — Director determines the goal is provably impossible under constraints. macOS notification sent.
3. **Manual** — `tmux kill-session -t agi-lab`

### Notifications

Uses macOS native `osascript` for notifications:
- Victory → "All benchmarks surpassed!"
- Catastrophic stop → "Human review needed"
- No cloud notification services needed — everything local

### Session Logs

Every session's full output is saved to `data/infra/session_logs/session_N_TIMESTAMP.log` for debugging and post-mortem analysis.

### Updated File System

```
data/
├── session_state.md              # Master resume state
├── cycle_queue.md                # Prioritized work queue
├── agents/                       # Per-agent persistent state
│   ├── director/
│   │   ├── state.md
│   │   └── decisions.md
│   ├── chief_scientist/
│   │   ├── state.md
│   │   └── decisions.md
│   ├── chief_engineer/
│   │   ├── state.md
│   │   └── decisions.md
│   ├── math_team_1/
│   │   ├── state.md
│   │   └── journal.md
│   ├── math_team_2/
│   │   ├── state.md
│   │   └── journal.md
│   ├── neuro_team_1/
│   │   ├── state.md
│   │   └── journal.md
│   ├── physics_team_1/
│   │   ├── state.md
│   │   └── journal.md
│   ├── novel_compute_1/
│   │   ├── state.md
│   │   └── journal.md
│   ├── literature_1/
│   │   ├── state.md
│   │   └── journal.md
│   ├── tensor_team_1/
│   │   ├── state.md
│   │   └── journal.md
│   ├── metal_kernel_1/
│   │   ├── state.md
│   │   └── journal.md
│   ├── architecture_1/
│   │   ├── state.md
│   │   └── journal.md
│   ├── swift_bridge_1/
│   │   ├── state.md
│   │   └── journal.md
│   ├── optimization_1/
│   │   ├── state.md
│   │   └── journal.md
│   ├── benchmark_lead/
│   │   ├── state.md
│   │   └── journal.md
│   ├── red_team_1/
│   │   ├── state.md
│   │   └── journal.md
│   ├── ablation_1/
│   │   ├── state.md
│   │   └── journal.md
│   ├── devops/
│   │   ├── state.md
│   │   └── journal.md
│   └── knowledge_agent/
│       ├── state.md
│       └── journal.md
├── roadmap.md
├── director_log.md
├── benchmark_tracker.md
├── shared_knowledge.md
├── knowledge_graph.md
├── contradictions.md
├── bibliography.md
├── experiments.db
├── research/
├── engineering/
├── eval/
├── infra/
│   ├── build_status.md
│   ├── disk_budget.md
│   ├── thermal_log.md
│   └── session_logs/            # Every session's full output
├── checkpoints/
├── digests/
├── papers/
├── plots/
└── training/
```
