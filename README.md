# AGI Lab Framework

> **A from-scratch autonomous ML-research lab that runs on a single 18 GB laptop**, a C17/Metal/Swift training engine with no ML framework (no PyTorch, no TensorFlow), a 31-role LLM research organization that self-directs from question to typeset paper, and a 4-layer hybrid-RAG memory that persists across months of work.

![License: MIT](https://img.shields.io/badge/license-MIT-green) ![Platform: macOS / Apple Silicon](https://img.shields.io/badge/platform-macOS%20%7C%20Apple%20Silicon-lightgrey) ![Languages: C17 / Metal / Python](https://img.shields.io/badge/languages-C17%20%7C%20Metal%20%7C%20Python-blue)

**Highlights** (measured, not estimated):

- **Scale:** ~74,200 LOC (raw `wc -l`), 283 development commits over ~7.1 weeks (this public repo is a curated snapshot of that work), including a ~19,600-LOC native engine (15,573 C / 2,740 headers / 1,047 Metal / 233 Swift) and ~19,000 lines of Python tooling.
- **Engine:** transformer training from scratch on Apple Silicon, Accelerate/CPU is the production path; a Metal GPU matmul path is built and tested but dormant.
- **Agent org:** 31 specialized roles (PI, Director, statisticians, engineers, reviewers) wielding 64 tools under a tmux runner, with pre-registration discipline and anti-forgery sign-offs.
- **Retrieval:** 4-layer hybrid-RAG (384-dim dense embeddings + BM25 + a token knowledge graph) with ~32 ms warm queries, fully local, zero cloud.
- **Tests:** 777 Python test functions + 269 C (Unity) cases.

**What's in this repo:** the full framework, native engine, 31-agent organization, 4-layer retrieval, build system, and test suite. See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the deep technical map.

**Honest framing:** the trained models are tiny, research-scale LMs (a 48M-param model benchmarks at/near random, expected at this size, never claimed as capability). The value is the *system* and the *rigorous method*: honest reporting of null results, formal decision rules, and a "beat-Opus" target the framework is built to test rather than assume.

---

An **autonomous AI agent organization** that runs ML research from scratch on a single laptop. 31 specialized agent roles (PI, Director, statisticians, engineers, reviewers, etc.) collaborate on multi-month research programs, self-directing from question → pre-registration → experiment → analysis → paper draft.

This repo is the **framework only**, code, agent role definitions, runner orchestration, memory architecture, retrieval system, build system. **No research output, no trained models, no accumulated agent memory.** Fork it, point it at your own scientific question, and let it run.

---

## Table of Contents

- [What this is](#what-this-is)
- [Architecture (deep technical map)](ARCHITECTURE.md)
- [Architecture](#architecture)
- [Hardware & API requirements](#hardware--api-requirements)
- [Setup](#setup)
- [Running the lab](#running-the-lab)
- [Repo layout](#repo-layout)
- [How a session works](#how-a-session-works)
- [The 31 agent roles](#the-31-agent-roles)
- [Memory architecture (4-tier)](#memory-architecture-4-tier)
- [The 15-phase research program lifecycle](#the-15-phase-research-program-lifecycle)
- [Retrieval system (4-layer hybrid)](#retrieval-system-4-layer-hybrid)
- [Customizing the framework](#customizing-the-framework)
- [Cost & operational notes](#cost--operational-notes)
- [Known caveats](#known-caveats)
- [Acknowledgments](#acknowledgments)

---

## What this is

A **self-directing research lab** built on top of Claude Code. Two co-equal leaders, a **PI** (scientific direction) and a **Director** (execution), collaborate via unanimous-compromise on program-level decisions. 28 specialist roles (across Scientific, Engineering, Optimization, Quality, Knowledge, Communication, and Meta layers) handle the work.

The lab operates autonomously: the Director runs in a continuous loop, claims work items from a queue, dispatches specialists with curated context, writes deliverables, commits decisions to a tiered memory, and exits cleanly so the next Director session has full state.

**Design principles:**

- **Think on disk, not in context**, agents write progressively to durable storage; conversation context is throwaway
- **Pre-registration discipline**, questions are locked in P1; experimental designs are locked in P6; outcomes pre-committed before any data is collected
- **Archive, never delete**, failed attempts, superseded artifacts, even forged signatures are preserved with breadcrumbs
- **Anti-forgery (D-109 pattern)**, PI/Director must personally edit their own sign-off blocks; orthogonal verifier checks integrity
- **One sized job per session**, atomic units of work that complete within ~10% of a session's context capacity

A research program runs end-to-end through the full 15-phase lifecycle described below, from locking a question to producing a typeset paper under formal multi-outcome decision rules.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Runner (run_agi_lab.sh)                                    │
│  - tmux-wrapped, caffeinated, restart-on-crash              │
│  - Pre-spawn gates: rate-limit, operator-review-pending     │
│  - Spawns: Director session via `claude --print`            │
│  - Post-session: post_director.py applies structured exits  │
│  - Reindexes: lab_memory (vector) + token graph + BM25      │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  Director session            │
              │  (claude-opus-4-7, max)      │
              │                              │
              │  1. Read session_brief.md    │
              │  2. Read context_brief.md    │
              │  3. Read current.md + INDEX  │
              │  4. (cond) Read substrate    │
              │     changelog (cursor-based) │
              │  5. Claim queue item         │
              │  6. Dispatch specialists     │
              │  7. Write deliverables       │
              │  8. memory.insert log entry  │
              │  9. Write session_exit.md    │
              │     (structured JSON)        │
              └────────────────┬─────────────┘
                               │
              ┌────────────────▼─────────────┐
              │  Dispatched specialist agents │
              │  (PI, statistician, red_team,│
              │   findings_curator, etc.)    │
              │                              │
              │  Each has:                   │
              │  - procedural.md (framework) │
              │  - episodic/ (history)       │
              │  - semantic.md (lessons)     │
              │  Returns structured result.  │
              └──────────────────────────────┘
```

**Three independent loops:**

1. **The runner**, bash, spawn-respawn forever
2. **The Director's per-session work**, bounded scope, atomic unit
3. **Specialist dispatches**, separate `claude --print` subprocesses for parallel work

The runner is the parent. Director and specialists are children. **Director must not kill its parent runner** (the prompt explicitly forbids this, there was an incident).

---

## Hardware & API requirements

| Item | Requirement |
|---|---|
| OS | macOS 14+ on Apple Silicon (M-series) |
| RAM | 16 GB minimum, 18 GB recommended |
| Disk | 100 GB+ free (data accumulates fast, training checkpoints + run logs) |
| API | Anthropic API key with **Opus access** + `claude-code` CLI installed |
| Python | 3.14+ |
| Other | `tmux`, `gh` (GitHub CLI, optional), `git`, `make` |

The lab depends on Apple's Metal kernels (in `src/`) for tensor ops, won't run on Linux/Windows without porting the GPU code.

---

## Setup

> ⚠️ **Do not put the repo (or its `.venv`) on a cloud-synced path**, macOS iCloud "Desktop & Documents", Dropbox, or OneDrive. Cloud sync *evicts* files to the cloud, so a cold `import torch` re-downloads ~2,100 files and can take **minutes** instead of <1s; it also *conflict-renames symlinks* (e.g. `.venv` → `.venv 2`), which silently breaks `.venv/bin/python` and takes the lab down. Keep everything under a **non-synced** directory like `~/code` or `~/dev` (the clone path below already does). `tools/preflight.py` warns if it detects a synced location.

```bash
# 1. Clone  (NOT under ~/Desktop or ~/Documents if iCloud sync is on)
git clone https://github.com/<your-user>/agi-lab-framework ~/code/AGI
cd ~/code/AGI

# 2. Install claude-code CLI (Anthropic)
#    See: https://docs.anthropic.com/en/docs/claude-code
#    Then: claude auth login (need Opus access)

# 3. Python venv + deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# (Also: pip install sentence-transformers sqlite-vec rank_bm25 networkx pyyaml pytest)

# 4. Build the C / Metal core
make all

# 5. Run the test suite
make test

# 6. Initialize the runtime directories (the framework share excludes these)
mkdir -p data/memories data/work_queue data/archives \
         data/checkpoints data/runs data/eval data/engineering \
         data/infra data/agents/_shared programs

# 7. Bootstrap the memory tier (write your first current.md, log.md, mission.md)
#    See data/agents/director/procedural.md for the canonical structure.

# 8. Edit CLAUDE.md to replace the example mission with YOUR mission
#    Then write your first program's question.md:
#      mkdir -p programs/program_1_<your_slug>
#      $EDITOR programs/program_1_<your_slug>/question.md

# 9. Start the autonomous lab
make lab-start            # attaches via tmux session 'agi-lab'
make lab-attach           # to view the live session
make lab-status           # quick status check
```

---

## Running the lab

```bash
make lab-start            # launches runner inside tmux, caffeinated
make lab-attach           # attach to live runner output
make lab-stop             # (manual): tmux kill-session -t agi-lab && pkill -f run_agi_lab.sh
make lab-status           # ps + last decision + queue depth
```

The runner is meant to run **continuously for days or weeks**. It self-handles:
- Rate-limit waits (sleeps until reset, no Director spawned)
- Watchdog kills for hung CLI processes
- Operator-review gates (operator-written `data/operator_review_pending.md` blocks Director spawn until you remove it)
- Stale claim reclaim (60-min timeout)
- Memory tier overage warnings

**Cost expectation:** $50-200/week of Anthropic API depending on research phase. Use `tools/cost_rollup.py` to track weekly.

---

## Repo layout

```
agi-lab-framework/
├── README.md                  ← you are here
├── CLAUDE.md                  ← project rules + mission (REPLACE WITH YOUR OWN)
├── Makefile                   ← build / test / lab-start targets
├── run_agi_lab.sh             ← the runner (~1100 lines bash)
├── requirements.txt           ← Python deps
│
├── src/                       ← C + Metal + Swift core
│   ├── tensor/                  tensor lib, BLAS, Metal kernels
│   ├── training/                backprop, iPC, optimizer
│   ├── eval/                    benchmark scorer
│   ├── inference/               from-scratch attention
│   └── tokenizer/               BPE
│
├── tools/                     ← Python tooling
│   ├── memory.py                Anthropic memory_20250818 protocol tool
│   ├── lab_memory.py            semantic search (sentence-transformers + sqlite-vec)
│   ├── retrieval/               4-layer hybrid retrieval (graph + BM25 + dense + reranker)
│   │   ├── graph.py               L0: token graph + Personalized PageRank
│   │   ├── bm25.py                L2: sparse retrieval (JSON-persisted)
│   │   ├── hybrid.py              L3: RRF fusion
│   │   ├── rerank.py              L4: cross-encoder reranker
│   │   ├── search.py              top-level orchestrator
│   │   └── concurrency.py         subprocess-isolated worker
│   ├── brief_assembler.py       pre-session context curation
│   ├── post_director.py         post-session structured-exit finalizer (RO-CO v1)
│   ├── work_queue.py            durable JSONL-backed queue
│   ├── queue_scanner.py         synthetic queue events from filesystem state
│   ├── dispatch_helper.py       agent dispatch + log_outcome
│   ├── cost_rollup.py           weekly cost analysis
│   ├── handler_schema.py        artifact_schema.yaml handlers
│   └── ... (~40 more modules)
│
├── tests/                     ← pytest suite
│
├── data/
│   └── agents/                ← agent role definitions (framework, not memory)
│       ├── _shared/             session_exit_schema, agent_contracts, dispatch helpers
│       ├── agents.json          roster of 31 roles
│       ├── director/procedural.md   ← read first
│       ├── pi/procedural.md         ← read second
│       ├── chief_scientist/procedural.md
│       ├── statistical_reviewer/procedural.md
│       ├── red_team/procedural.md
│       ├── findings_curator/procedural.md
│       ├── lab_architect/procedural.md
│       └── ... (31 roles total)
│
└── docs/
    ├── retrieval/architecture.md   ← 4-layer retrieval design doc
    └── superpowers/
        ├── specs/                    design docs for major substrate features
        └── plans/                    implementation plans the lab executes
```

**At runtime, the lab will create:**
- `data/memories/`, tiered wiki (current.md / log.md / mission.md / governance/ / programs/ / etc.)
- `data/work_queue/`, JSONL pending / claimed / completed
- `data/agents/<role>/episodic/` + `semantic.md`, accumulated per-agent memory
- `data/archives/<date>/`, preserved historical artifacts (Values §4: archive, never delete)
- `data/checkpoints/`, `data/runs/`, `data/eval/`, training artifacts
- `programs/program_N_<slug>/`, per-program deliverables
- `tools/lab_memory.db`, `tools/lab_graph.db`, `tools/lab_bm25.json`, built retrieval indexes
- `SUBSTRATE_CHANGELOG.md`, operator-authored changelog of framework changes

These are excluded from the framework share by `.gitignore`. They're your research output.

---

## How a session works

1. **Runner spawns Director** via `claude --print --model claude-opus-4-7 --effort max` with the procedural prompt at `data/agents/director/procedural.md`
2. **Director's first 3 tool calls** (always):
   - `memory.view session_brief.md` (runner-written pointer)
   - `memory.view INDEX.md`
   - `memory.view <relevant file>` (typically `current.md`)
3. **Conditional Step 4**, if `Substrate changelog: <N> unread > 0`, read the changelog + advance cursor
4. **Claim a work item** from `data/work_queue/pending.jsonl`
5. **Dispatch specialists** via the `Agent` tool with `subagent_type=<role>`, providing curated context
6. **Wait for specialists**, validate their structured returns
7. **Write deliverables** to `programs/<program>/...`
8. **Commit to log.md** via `memory.insert log.md --line 0 --text "<decision block>"`
9. **Patch current.md** via `memory.str_replace` (surgical edits only, wholesale rewrites blocked by Evaluator)
10. **Write `data/session_exit.md`**, structured JSON declaring status, next_action, current.md patches, deliverables, etc.
11. **Director exits.** Runner's `post_director.py` applies the exit, enqueues next_action, validates contracts, rotates logs.

A typical session is 5-20 minutes. The lab runs hundreds of these in series.

---

## The 31 agent roles

Defined in `data/agents/agents.json`. Each has a `procedural.md` (framework instructions). At runtime they accrue `episodic/` and `semantic.md` (per-role memory).

| Layer | Roles |
|---|---|
| **L1 Direction (3)** | `pi`, `director`, `unanimous_compromise_mediator` |
| **L2 Scientific (6)** | `chief_scientist`, `math_theorist`, `experimental_methodologist`, `hypothesis_generator`, `mechanism_extractor`, `measurement_theorist` |
| **L3 Engineering (5)** | `infrastructure_architect`, `implementation_engineer_c`, `sota_scout`, `tooling_engineer`, `reproducibility_engineer` |
| **L4 Optimization (3)** | `profiler`, `kernel_specialist`, `memory_optimizer` |
| **L5 Quality (5)** | `scientific_reviewer`, `statistical_reviewer`, `red_team`, `pre_reg_auditor`, `code_reviewer` |
| **L6 Knowledge (3)** | `literature_hunter`, `paper_digester`, `findings_curator` |
| **L7 Communication (2)** | `paper_writer`, `figure_generator` |
| **L8 Meta (4)** | `lab_architect`, `grant_reviewer`, `evaluator`, `consolidator` |

The Director and PI are co-equal. Specialists are dispatched and return structured results. The `findings_curator` runs the **KM-Closeout** at every phase close (archive deliverables, rotate logs, update bibliographies). The `lab_architect` runs structural retros after each program closes.

---

## Memory architecture (4-tier)

The lab's persistent memory follows a deliberate tiering inspired by the [CoALA](https://arxiv.org/abs/2309.02427) framework:

```
┌──────────────────────────────────────────────────────────────┐
│  Tier 0, Pre-session brief (runner-written, ephemeral)      │
│    session_brief.md (last decision, rate-limit, substrate)   │
│    context_brief.md (curated 10-30 KB, classified by type)   │
├──────────────────────────────────────────────────────────────┤
│  Tier 1, Hot wiki (current.md, log.md), 40 KB cap each    │
│    Updated every session via memory.str_replace / .insert    │
├──────────────────────────────────────────────────────────────┤
│  Tier 2, Wiki (mission.md, governance/, programs/, etc.)    │
│    50 KB total cap. Read on demand, edited rarely.           │
├──────────────────────────────────────────────────────────────┤
│  Tier 3, Per-agent CoALA memory                              │
│    data/agents/<role>/procedural.md  (framework, immutable)  │
│    data/agents/<role>/episodic/<date>_<topic>.md (what I did)│
│    data/agents/<role>/semantic.md (lessons learned)          │
├──────────────────────────────────────────────────────────────┤
│  Tier 4, Vector + token graph + BM25 retrieval               │
│    On-demand semantic search across all of the above         │
└──────────────────────────────────────────────────────────────┘
```

**Cap enforcement:** When a tier exceeds its cap, the next `findings_curator` KM-Closeout archives overflow to `data/archives/<date>/<topic>.md` (Values §4: archive-never-delete). Breadcrumbs are left at the original location.

**Anti-corruption:** Evaluator check #16 blocks wholesale (>80% line change) rewrites of tier files in git diff at phase close. Forces surgical edits.

---

## The 15-phase research program lifecycle

Every research program follows the same 15 phases (~3-6 months of lab time):

```
P1 question         lock the primary research question (blob hash binding)
P2 lit scan         literature_hunter + paper_digester survey prior work
P3 theory           mechanism / mathematical framing
P4 hypotheses       hypothesis_generator produces falsifiable claims
P5 design           experimental_methodologist drafts experimental_design.md
                    + red_team adversarial review rounds
P6 pre-register     LOCK the design (PI+Director unanimous gate)
                    decision rules + outcome map + paper titles pre-committed
P7 apparatus        infrastructure_architect + implementation_engineer_c build
                    the actual training/eval code
P8 execute          run training (factorial sweep, N seeds per arm, etc.)
P9 analyze          statistical_reviewer runs the pre-registered tests
P10 mechanism       mechanism_extractor produces "why" narrative
P11 measurement     benchmark scoring against locked decision rules
P12 peer review     red_team + scientific_reviewer adversarial check
P13 draft           paper_writer produces V1 of the paper
P14 revise          iterate to V_final with PI signoff
P15 close           closure_memo + findings_curator KM-Closeout
                    + optional publication opt-in
```

**Unanimous gates:** Opening a program, locking a question (P1), locking a pre-reg (P6), closing a phase, killing a program, approving a paper, all require PI + Director unanimous agreement. The `unanimous_compromise_mediator` role handles deadlocks.

**Programs in flight have a `programs/program_N_<slug>/` directory.** When you start the lab, you create `programs/program_1_<your_slug>/question.md` to seed your first program. The lab takes it from there.

---

## Retrieval system (4-layer hybrid)

The lab's accumulated corpus (decisions, programs, agent memories, code reviews) is searchable via a 4-layer hybrid retrieval system. Full design doc at `docs/retrieval/architecture.md`.

```
┌──────────────────────────────────────────────────────────────┐
│  L4: Cross-encoder reranker (bge-reranker-v2-m3, 568 MB)     │ ← top-30 → top-5
├──────────────────────────────────────────────────────────────┤
│  L3: RRF fusion (k=60)                                        │ ← merge BM25 + dense
├────────────────────────────┬─────────────────────────────────┤
│  L2: BM25 (rank_bm25)      │  L1: Dense vector (MiniLM 384)  │
│  - JSON-persisted          │  - sqlite-vec storage           │
│  - Token-preserving (D-N,  │  - Hugging Face cache           │
│     P-*, program IDs)      │                                 │
├────────────────────────────┴─────────────────────────────────┤
│  L0: Token graph (regex-built from canonical vocabulary)     │
│    Nodes: D-N decisions, P-* carry-forwards, programs,       │
│           phases, roles, files                                │
│    Edges: cites / precedes / resolves / raises / in_program  │
│    Queried via Personalized PageRank (networkx)              │
└──────────────────────────────────────────────────────────────┘
```

**Why this shape:**
- Lab's canonical token vocabulary (D-N decisions, P-* carry-forwards) makes graph construction trivial, regex parse, no LLM extraction needed
- BM25 catches exact-token matches (e.g. "D-420") that dense retrieval drops
- RRF fuses rankings without needing score calibration
- Reranker is cheap enough to add at top of retrieval

**CLI:**

```bash
# Hybrid (default)
.venv/bin/python tools/lab_memory.py search "P-D417 dedup gap" --top-k 5

# Sparse-only (when query is identifier-dominated)
.venv/bin/python -m tools.retrieval.bm25 search "D-420" --top-k 3

# Token graph relational query
.venv/bin/python -m tools.retrieval.graph ppr --seed D-420 --top-k 10

# Legacy pure-vector path
.venv/bin/python tools/lab_memory.py search "..." --legacy
```

Indexes auto-refresh after every Director exit (post-session hook in `run_agi_lab.sh`).

---

## Customizing the framework

**Replace the mission.** Edit `CLAUDE.md` to remove the example mission and put your own. Same for any references to "Program 1/2/3" specifics that don't apply.

**Define your first program.** Create `programs/program_1_<your_slug>/question.md` with your primary research question. The Director will pick it up when you start the lab.

**Edit role prompts.** Each role's procedural.md is a customizable prompt. The Director's prompt at `data/agents/director/procedural.md` controls the most behavior, see its "First three tool calls" and "Do NOT" sections.

**Add new roles.** Add a row to `data/agents/agents.json` and create `data/agents/<new_role>/procedural.md`. The dispatch infrastructure picks it up automatically.

**Add new work-item types.** Edit `tools/work_queue.py` priority validation + add a handler in `tools/queue_scanner.py` if you want synthetic events for it.

**Modify the runner.** `run_agi_lab.sh` is bash. Major substrate changes should be documented in `SUBSTRATE_CHANGELOG.md` (the runner reads its mtime/cursor to surface unread changes to the next Director session).

---

## Cost & operational notes

**Anthropic API costs:** Opus is expensive. Typical week:
- Director-only steady state: ~$30-50/week
- Active dispatch (multiple specialists per session): $100-200/week
- Apparatus build phase (many code-review iterations): $150-250/week

The `tools/cost_rollup.py` produces a weekly markdown report, runs automatically after each Director exit.

**Rate limits:** Anthropic has 5-hour rolling TPM limits + weekly TPM budgets. The runner has a pre-spawn gate that sleeps until rate-limit reset (no useless Director spawns during long waits). This was added after an incident where the lab burned 159 no-op sessions over 13 hours.

**Disk:** Training checkpoints can grow to 30+ GB per program. Run logs accumulate. Plan for 100 GB+ of free space if you run the lab seriously.

**Tmux:** The lab runs inside a tmux session named `agi-lab`. The wrapper uses `caffeinate -dis` so macOS doesn't sleep mid-experiment.

---

## Known caveats

- **macOS only.** Metal shaders in `src/` require Apple Silicon. No Linux/Windows port.
- **Opus only.** The framework assumes `claude-opus-4-7` for Director sessions. Other models will not produce the same reasoning depth.
- **No retries on bad agent dispatches.** If a dispatched specialist returns garbage, the Director must catch it. There's `verifier_pairs.json` for some critical roles but it's incomplete.
- **iCloud Desktop sync conflicts.** If your lab dir lives under `~/Desktop` on macOS with Desktop & Documents iCloud sync enabled, deleted state files can be restored from snapshots. Put the lab outside iCloud-synced directories (this README assumes `~/code/AGI`).
- **Memory tier overages are aspirational.** The lab tracks 40 KB / 30 KB / 50 KB caps on hot/log/wiki tiers but enforcement is reactive (KM-Closeout) not preventive. Files can grow past cap between closes.
- **Sandboxed Python.** Some pip installs require running from inside `.venv/` not system Python.

---

## Acknowledgments

Built on:
- [Anthropic Claude Code](https://docs.anthropic.com/en/docs/claude-code), the CLI that hosts the Director sessions
- [CoALA](https://arxiv.org/abs/2309.02427), the agent memory taxonomy (working / episodic / semantic / procedural)
- [HippoRAG2](https://arxiv.org/abs/2405.14831), Personalized PageRank over a token graph
- [Microsoft GraphRAG](https://arxiv.org/abs/2404.16130), relational retrieval (though we use canonical tokens instead of LLM extraction)
- [BM25Okapi](https://github.com/dorianbrown/rank_bm25), sparse retrieval
- [sentence-transformers](https://www.sbert.net/), MiniLM-L6-v2 for dense embeddings
- [sqlite-vec](https://github.com/asg017/sqlite-vec), vector storage
- [`networkx`](https://networkx.org/), graph algorithms (PPR)
- [`rank_bm25`](https://github.com/dorianbrown/rank_bm25), BM25 implementation

Inspired by but does not reuse code from:
- AI-Scientist (Sakana, 2024), autonomous research agent prior art
- Letta / MemGPT (Packer et al., 2023), agent memory paging
- HIVE-Claude (internal pattern, 2026), substrate-awareness mechanism

---

## License

MIT. See `LICENSE`. (Add one if you fork to a public repo.)

---

## Questions?

This is research software shared as-is. Read `CLAUDE.md` first, then `data/agents/director/procedural.md`, then `run_agi_lab.sh`. Most "how does X work" questions are answered in those three files. For substrate questions, check `docs/superpowers/specs/`.
