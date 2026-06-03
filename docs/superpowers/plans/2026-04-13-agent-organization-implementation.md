# AGI Lab Agent Organization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full autonomous agent organization infrastructure — directory structure, agent prompts, state management, wrapper script — so that running `run_agi_lab.sh` launches a self-directing AGI lab of ~30 agents that operates continuously until it beats Opus 4.6 on standardized benchmarks.

**Architecture:** A tmux wrapper script (`run_agi_lab.sh`) loops forever, launching Claude Code sessions with `--print --model opus --effort max`. Each session reads `data/session_state.md` to know where the org left off, acts as the Director agent, launches subagents via the Agent tool, and checkpoints all state to disk before exiting. Custom `--agents` JSON defines all 18+ agent roles with their specialized prompts. Existing tools (`experiments.py`, `research_digest.py`, `hwmon.py`, `benchmark.py`, `arxiv_reader.py`, `mathengine.py`, `verify.py`, `visualize.py`) are used by agents for their work.

**Tech Stack:** Bash (wrapper), Claude Code CLI (`--print`, `--model`, `--effort`, `--agents`, `--dangerously-skip-permissions`), existing Python tools, tmux.

**Spec:** `docs/superpowers/specs/2026-04-13-agent-organization-design.md`

---

## File Structure

```
AGI/
├── CLAUDE.md                              # Project-level instructions for all sessions
├── run_agi_lab.sh                         # Tmux wrapper — the entry point
├── data/
│   ├── agents/
│   │   ├── agents.json                    # All agent role definitions for --agents flag
│   │   ├── director/
│   │   │   ├── state.md                   # Director state
│   │   │   └── decisions.md               # Director decision ledger
│   │   ├── chief_scientist/
│   │   │   ├── state.md
│   │   │   └── decisions.md
│   │   ├── chief_engineer/
│   │   │   ├── state.md
│   │   │   └── decisions.md
│   │   ├── math_team_1/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── math_team_2/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── neuro_team_1/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── neuro_team_2/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── physics_team_1/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── physics_team_2/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── novel_compute_1/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── novel_compute_2/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── literature_1/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── tensor_team_1/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── tensor_team_2/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── metal_kernel_1/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── metal_kernel_2/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── architecture_1/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── architecture_2/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── swift_bridge_1/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── optimization_1/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── optimization_2/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── benchmark_lead/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── red_team_1/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── red_team_2/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── ablation_1/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   ├── devops/
│   │   │   ├── state.md
│   │   │   └── journal.md
│   │   └── knowledge_agent/
│   │       ├── state.md
│   │       └── journal.md
│   ├── session_state.md                   # Master orchestration state
│   ├── session_exit.md                    # Exit reason for wrapper script
│   ├── cycle_queue.md                     # Prioritized work queue
│   ├── roadmap.md                         # Director's master plan
│   ├── director_log.md                    # Director decision journal
│   ├── benchmark_tracker.md               # Scores vs Opus 4.6
│   ├── research/
│   │   ├── strategy.md                    # Chief Scientist priorities
│   │   └── synthesis_log.md               # Cross-team insights
│   ├── engineering/
│   │   ├── tech_plan.md                   # Chief Engineer priorities
│   │   ├── memory_budget.md               # 18GB allocation
│   │   ├── perf_log.md                    # Benchmark results
│   │   └── tensor_metal_spec.md           # Shared Tensor+Metal spec
│   ├── eval/
│   │   ├── scorecard.md                   # Latest scores vs Opus 4.6
│   │   ├── regression_log.md              # What broke and when
│   │   ├── benchmark_history.db           # Full eval history
│   │   └── ablation_log.md               # What works
│   ├── infra/
│   │   ├── build_status.md                # Compilation status
│   │   ├── disk_budget.md                 # Storage usage
│   │   ├── thermal_log.md                 # Overheating events
│   │   └── session_logs/                  # Per-session output logs
│   ├── shared_knowledge.md                # (exists) Knowledge bus
│   ├── knowledge_graph.md                 # Structured concept map
│   ├── contradictions.md                  # Unresolved conflicts
│   ├── bibliography.md                    # Living paper reference
│   └── experiments.db                     # (exists) Experiment tracker
```

---

## Task 1: Create Agent State Directory Structure

**Files:**
- Create: `data/agents/` and all subdirectories
- Create: All `state.md` and `journal.md` / `decisions.md` template files
- Create: All division state directories (`data/research/`, `data/engineering/`, `data/eval/`, `data/infra/session_logs/`)

- [ ] **Step 1: Create all agent directories**

Run:
```bash
cd <repo>

# Research division agents
mkdir -p data/agents/{director,chief_scientist,chief_engineer}
mkdir -p data/agents/{math_team_1,math_team_2}
mkdir -p data/agents/{neuro_team_1,neuro_team_2}
mkdir -p data/agents/{physics_team_1,physics_team_2}
mkdir -p data/agents/{novel_compute_1,novel_compute_2}
mkdir -p data/agents/literature_1

# Engineering division agents
mkdir -p data/agents/{tensor_team_1,tensor_team_2}
mkdir -p data/agents/{metal_kernel_1,metal_kernel_2}
mkdir -p data/agents/{architecture_1,architecture_2}
mkdir -p data/agents/swift_bridge_1
mkdir -p data/agents/{optimization_1,optimization_2}

# Evaluation division agents
mkdir -p data/agents/{benchmark_lead,red_team_1,red_team_2,ablation_1}

# Infrastructure division agents
mkdir -p data/agents/{devops,knowledge_agent}

# Division state directories
mkdir -p data/{research,engineering,eval,infra/session_logs}
```

Expected: All directories created, no errors.

- [ ] **Step 2: Create agent state file template script**

Create `tools/init_agent_state.sh`:

```bash
#!/bin/bash
# init_agent_state.sh — Initialize all agent state files with templates
cd <repo>

# Function to create a team agent's state + journal
init_team_agent() {
    local agent_dir="data/agents/$1"
    local division="$2"
    local team="$3"
    
    cat > "$agent_dir/state.md" << EOF
agent: $1
division: $division
team: $team
status: NOT_STARTED
current_task: none
task_id: null
journal: $agent_dir/journal.md
blocked_by: nothing
outputs_so_far: []
next_step: "Await first assignment from $([ "$division" = "research" ] && echo "Chief Scientist" || echo "Chief Engineer")"
context_needed: "Read data/shared_knowledge.md and data/${division}/strategy.md before starting"
EOF

    cat > "$agent_dir/journal.md" << EOF
# $1 — Work Journal

*Progressive log of all work. Written incrementally, never in bulk.*
*Each entry includes: task, approach, steps with results, cross-references.*

---
EOF
}

# Function to create a lead agent's state + decisions
init_lead_agent() {
    local agent_dir="data/agents/$1"
    local role="$2"
    
    cat > "$agent_dir/state.md" << EOF
agent: $1
role: $role
status: NOT_STARTED
current_task: none
last_cycle_completed: 0
next_step: "Initialize and begin first cycle"
active_directives: []
EOF

    cat > "$agent_dir/decisions.md" << EOF
# $1 — Decision Ledger

*Every decision logged with: what, why, alternatives considered, reversibility.*
*Future sessions read this to understand past reasoning.*

---
EOF
}

# Director
init_lead_agent "director" "Top-level orchestrator"

# Chiefs
init_lead_agent "chief_scientist" "Research division lead"
init_lead_agent "chief_engineer" "Engineering division lead"

# Research teams
for agent in math_team_1 math_team_2; do
    init_team_agent "$agent" "research" "Math / Information Theory"
done
for agent in neuro_team_1 neuro_team_2; do
    init_team_agent "$agent" "research" "Neuroscience"
done
for agent in physics_team_1 physics_team_2; do
    init_team_agent "$agent" "research" "Physics"
done
for agent in novel_compute_1 novel_compute_2; do
    init_team_agent "$agent" "research" "Novel Compute"
done
init_team_agent "literature_1" "research" "Literature Review"

# Engineering teams
for agent in tensor_team_1 tensor_team_2; do
    init_team_agent "$agent" "engineering" "Tensor / Compute"
done
for agent in metal_kernel_1 metal_kernel_2; do
    init_team_agent "$agent" "engineering" "Metal Kernel"
done
for agent in architecture_1 architecture_2; do
    init_team_agent "$agent" "engineering" "Architecture"
done
init_team_agent "swift_bridge_1" "engineering" "Swift Bridge"
for agent in optimization_1 optimization_2; do
    init_team_agent "$agent" "engineering" "Optimization"
done

# Evaluation
init_lead_agent "benchmark_lead" "Evaluation division lead"
for agent in red_team_1 red_team_2; do
    init_team_agent "$agent" "evaluation" "Red Team"
done
init_team_agent "ablation_1" "evaluation" "Ablation"

# Infrastructure
init_team_agent "devops" "infrastructure" "DevOps"
init_team_agent "knowledge_agent" "infrastructure" "Knowledge Management"

echo "=== All agent state files initialized ==="
```

- [ ] **Step 3: Run the init script**

Run:
```bash
chmod +x tools/init_agent_state.sh && bash tools/init_agent_state.sh
```

Expected: "All agent state files initialized" and every `data/agents/*/state.md` and `journal.md`/`decisions.md` exists.

- [ ] **Step 4: Verify all files created**

Run:
```bash
find data/agents -name "*.md" | sort | wc -l && find data/agents -name "*.md" | sort
```

Expected: 56 files (28 agents x 2 files each). All listed.

- [ ] **Step 5: Commit**

```bash
git add data/agents/ tools/init_agent_state.sh
git commit -m "feat: initialize agent state directory structure and templates"
```

---

## Task 2: Create Division State Files

**Files:**
- Create: `data/research/strategy.md`, `data/research/synthesis_log.md`
- Create: `data/engineering/tech_plan.md`, `data/engineering/memory_budget.md`, `data/engineering/perf_log.md`, `data/engineering/tensor_metal_spec.md`
- Create: `data/eval/scorecard.md`, `data/eval/regression_log.md`, `data/eval/ablation_log.md`
- Create: `data/infra/build_status.md`, `data/infra/disk_budget.md`, `data/infra/thermal_log.md`
- Create: `data/knowledge_graph.md`, `data/contradictions.md`, `data/bibliography.md`
- Create: `data/roadmap.md`, `data/director_log.md`, `data/benchmark_tracker.md`
- Create: `data/cycle_queue.md`

- [ ] **Step 1: Create Director state files**

Create `data/roadmap.md`:
```markdown
# AGI Lab — Master Roadmap

*Maintained by Director. Updated after each decision cycle.*

## Current Phase: INITIALIZATION

## Active Research Streams
- (none yet — first cycle will set these)

## Active Engineering Work
- (blocked on research findings)

## Resource Allocation
- Research: 100% (initial phase)
- Engineering: 0% (waiting)
- Evaluation: 0% (waiting)

## Killed Streams
- (none)

## Milestones
- [ ] First research cycle complete
- [ ] First architectural proposal
- [ ] First working tensor operation in C
- [ ] First Metal kernel running
- [ ] First training loop
- [ ] First benchmark eval
- [ ] All benchmarks surpass Opus 4.6
```

Create `data/director_log.md`:
```markdown
# Director — Decision Journal

*Every decision with: what, why, evidence, alternatives, outcome.*
*Read by future sessions to understand trajectory.*

---
```

Create `data/benchmark_tracker.md`:
```markdown
# Benchmark Tracker — Our Scores vs Opus 4.6

*Updated by Benchmark Lead after every eval run. Read by Director for prioritization.*

## Target Benchmarks

| Benchmark | Opus 4.6 Score | Our Score | Gap | Status |
|-----------|---------------|-----------|-----|--------|
| MMLU / MMLU-Pro | TBD (research) | — | — | NOT STARTED |
| HumanEval / MBPP | TBD (research) | — | — | NOT STARTED |
| ARC-AGI | TBD (research) | — | — | NOT STARTED |
| GSM8K / MATH | TBD (research) | — | — | NOT STARTED |
| HellaSwag | TBD (research) | — | — | NOT STARTED |
| TruthfulQA | TBD (research) | — | — | NOT STARTED |
| WinoGrande | TBD (research) | — | — | NOT STARTED |
| BigBench-Hard | TBD (research) | — | — | NOT STARTED |

*Note: Opus 4.6 scores will be researched and filled in by Literature Review team in cycle 1.*

## Score History
(none yet)
```

Create `data/cycle_queue.md`:
```markdown
# Cycle Queue — Prioritized Work

*Updated by Director each cycle. Read by session on startup.*

## Current Cycle: 0 (INITIALIZATION)

Priority order:
1. [PENDING] Director: Initial planning — set research priorities, assign first hypotheses
2. [PENDING] Literature: Research Opus 4.6 benchmark scores as targets
3. [PENDING] Research: First hypothesis cycle across all streams
4. [BLOCKED] Engineering: Awaiting first research findings
5. [BLOCKED] Evaluation: Awaiting first model checkpoint

## Backlog
- (empty — will be populated after cycle 1)
```

- [ ] **Step 2: Create Research division state files**

Create `data/research/strategy.md`:
```markdown
# Research Strategy

*Maintained by Chief Scientist. Updated each cycle.*

## Current Priorities
(awaiting Director's first cycle directives)

## Open Questions
1. What fundamental computational paradigm should we use? (not transformers)
2. How do we represent knowledge efficiently in 18GB?
3. What learning algorithm achieves the best sample efficiency?
4. What architecture exploits M3 Pro unified memory most effectively?

## Research Streams Status
| Stream | Status | Lead | Current Focus |
|--------|--------|------|---------------|
| Math / Information Theory | NOT STARTED | math_team_1, math_team_2 | — |
| Neuroscience | NOT STARTED | neuro_team_1, neuro_team_2 | — |
| Physics | NOT STARTED | physics_team_1, physics_team_2 | — |
| Novel Compute | NOT STARTED | novel_compute_1, novel_compute_2 | — |
| Literature Review | NOT STARTED | literature_1 | — |
```

Create `data/research/synthesis_log.md`:
```markdown
# Research Synthesis Log

*Written by Chief Scientist after each cycle. Cross-team insights and connections.*

---
```

- [ ] **Step 3: Create Engineering division state files**

Create `data/engineering/tech_plan.md`:
```markdown
# Engineering Tech Plan

*Maintained by Chief Engineer. Updated each cycle.*

## Current Implementation Priorities
(awaiting first research findings)

## Implementation Queue
(empty — populated when research hands off findings)

## Technical Debt
(none yet)
```

Create `data/engineering/memory_budget.md`:
```markdown
# Memory Budget — 18GB Unified RAM Allocation

*Maintained by Chief Engineer. Optimization team has veto power.*

## Total Available: 18,432 MB (18 GB)

## Allocation
| Component | Budget (MB) | Actual (MB) | Status |
|-----------|------------|-------------|--------|
| macOS + System | ~4,000 | ~4,000 | FIXED |
| Model Weights | TBD | — | NOT STARTED |
| Activations / KV Cache | TBD | — | NOT STARTED |
| Training Buffers | TBD | — | NOT STARTED |
| Data Pipeline | TBD | — | NOT STARTED |
| **Remaining** | **~14,000** | — | — |

*Note: macOS overhead is ~4GB. We have ~14GB for everything else.*
```

Create `data/engineering/perf_log.md`:
```markdown
# Performance Log

*Benchmarks over time. Updated by Optimization team after each measurement.*

---
```

Create `data/engineering/tensor_metal_spec.md`:
```markdown
# Tensor + Metal Kernel Shared Specification

*Jointly maintained by Tensor team and Metal Kernel team.*
*Memory layout decisions here affect both teams — changes require both to sign off.*

## Data Layout
(not yet defined — awaiting architecture decisions from research)

## Supported Operations
(not yet defined)

## Memory Management Protocol
(not yet defined)
```

- [ ] **Step 4: Create Evaluation division state files**

Create `data/eval/scorecard.md`:
```markdown
# Evaluation Scorecard

*Single source of truth on model quality. Updated after every eval run.*
*All agents read this. Director uses it for prioritization.*

## Status: NO MODEL YET

## Latest Scores
(no evaluations run yet)

## Trend
(no data)
```

Create `data/eval/regression_log.md`:
```markdown
# Regression Log

*What broke and when. Filed by Benchmark Lead.*

---
```

Create `data/eval/ablation_log.md`:
```markdown
# Ablation Log

*Component-by-component impact analysis. Maintained by Ablation team.*
*Tells us what's actually helping vs dead weight.*

---
```

- [ ] **Step 5: Create Infrastructure state files**

Create `data/infra/build_status.md`:
```markdown
# Build Status

*Maintained by DevOps agent.*

## C Core: NO SOURCE YET
## Metal Shaders: NO SOURCE YET
## Swift Bridge: NO SOURCE YET
## Python Tools: ALL WORKING (9/9)
## Last Successful Build: N/A
```

Create `data/infra/disk_budget.md`:
```markdown
# Disk Budget

*Maintained by DevOps agent. Monitored each cycle.*

## Total Available: ~233 GB
## Currently Used by Project: TBD
## Checkpoint Budget: TBD
## Training Data Budget: TBD
## Warning Threshold: 50 GB remaining
## Critical Threshold: 20 GB remaining
```

Create `data/infra/thermal_log.md`:
```markdown
# Thermal Log

*Monitored by DevOps via hwmon.py. Throttle events logged here.*

---
```

- [ ] **Step 6: Create Knowledge state files**

Create `data/knowledge_graph.md`:
```markdown
# Knowledge Graph

*Structured concept map maintained by Knowledge Agent.*
*Format: Concept → Findings → Experiments → Code*

---
```

Create `data/contradictions.md`:
```markdown
# Contradiction Board

*Things that don't add up yet. Found by Knowledge Agent, resolved by Chief Scientist.*
*Contradictions are often where breakthroughs hide.*

---
```

Create `data/bibliography.md`:
```markdown
# Bibliography

*Living reference list maintained by Knowledge Agent from Literature team's arxiv scans.*
*Indexed by relevance score and research stream.*

---
```

- [ ] **Step 7: Verify all division files created**

Run:
```bash
find data/research data/engineering data/eval data/infra -name "*.md" | sort && \
ls data/roadmap.md data/director_log.md data/benchmark_tracker.md data/cycle_queue.md \
   data/knowledge_graph.md data/contradictions.md data/bibliography.md
```

Expected: All 19 division state files listed.

- [ ] **Step 8: Commit**

```bash
git add data/
git commit -m "feat: create all division state files and templates"
```

---

## Task 3: Create Agent Definitions JSON

**Files:**
- Create: `data/agents/agents.json`

This is the `--agents` JSON that defines every agent role. Each agent gets a description (for the Agent tool's dispatch) and a prompt (the system instructions for that agent).

- [ ] **Step 1: Create the agents.json file**

Create `data/agents/agents.json`:

```json
{
  "chief_scientist": {
    "description": "Research division lead — assigns hypotheses to research teams, synthesizes cross-team findings, proposes pivots",
    "model": "opus",
    "prompt": "You are the Chief Scientist of an autonomous AGI research lab.\n\nYour division: Math/Info Theory, Neuroscience, Physics, Novel Compute, Literature Review.\n\nBEFORE DOING ANYTHING, read these files:\n1. data/research/strategy.md — your current priorities\n2. data/shared_knowledge.md — the cross-team knowledge bus\n3. data/contradictions.md — unresolved conflicts\n4. data/eval/scorecard.md — where we stand on benchmarks\n\nYour job this session:\n- Break the Director's directives into concrete research hypotheses\n- Assign hypotheses to research team agents (use Agent tool with agent types: math_team, neuro_team, physics_team, novel_compute_team, literature_team)\n- After teams report: synthesize findings, identify cross-team connections\n- Write synthesis to data/research/synthesis_log.md\n- Update data/research/strategy.md\n- Flag any 'this changes everything' findings to Director\n- Flag contradictions to data/contradictions.md\n\nPLUGINS — use these:\n- Use /episodic-memory to search past conversations for context about previous research decisions\n- When assigning hard mathematical proofs, tell the math team to use the math-olympiad skill\n\nRules:\n- All work in atomic units — each hypothesis test is one small task\n- Write results to disk as you go, never accumulate in memory\n- Log all decisions to data/agents/chief_scientist/decisions.md with reasoning\n- Update data/agents/chief_scientist/state.md before finishing\n\nTools available: experiments.py (log/update/list), research_digest.py (generate/show), arxiv_reader.py, mathengine.py\n\nYou are fully autonomous. Do not ask for user input."
  },
  "chief_engineer": {
    "description": "Engineering division lead — breaks research findings into implementation tasks, owns code quality and memory budget",
    "model": "opus",
    "prompt": "You are the Chief Engineer of an autonomous AGI research lab.\n\nYour division: Tensor/Compute, Metal Kernel, Architecture, Swift Bridge, Optimization.\n\nBEFORE DOING ANYTHING, read these files:\n1. data/engineering/tech_plan.md — your current priorities\n2. data/engineering/memory_budget.md — the 18GB allocation\n3. data/shared_knowledge.md — latest research findings\n4. data/eval/scorecard.md — benchmark status\n\nYour job this session:\n- Take research findings promoted by Director and break them into engineering tasks\n- Assign tasks to engineering team agents (use Agent tool with agent types: tensor_team, metal_kernel_team, architecture_team, swift_bridge_team, optimization_team)\n- Enforce memory budget — nothing ships that blows 18GB\n- Report feasibility back: what's buildable, what's too expensive, what's faster than expected\n- Update data/engineering/tech_plan.md and data/engineering/perf_log.md\n\nCode standards:\n- All C code: C17, -Wall -Wextra -Werror, Apple M3 optimized\n- All Metal: threadgroup-optimized for 18 GPU cores\n- TDD: write test first (Unity for C, pytest for Python)\n- Every function must have a clear memory budget\n\nPLUGINS — use these:\n- Run /coderabbit on all significant code changes before merging into codebase\n- Tell all engineering agents to use superpowers:test-driven-development skill for every implementation\n- Tell all engineering agents to use superpowers:systematic-debugging when tests fail\n- Use superpowers:verification-before-completion before reporting any task as done\n- Use /episodic-memory to recall past engineering decisions from previous sessions\n\nRules:\n- Write results to disk as you go\n- Log decisions to data/agents/chief_engineer/decisions.md\n- Update data/agents/chief_engineer/state.md before finishing\n- If something can't be built as designed, file a veto with reasoning\n\nTools: benchmark.py, hwmon.py, verify.py, experiments.py\n\nYou are fully autonomous. Do not ask for user input."
  },
  "math_team": {
    "description": "Math/Information Theory researcher — optimal representations, compression bounds, loss functions, optimization theory",
    "model": "opus",
    "prompt": "You are a Math/Information Theory researcher in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/math_team_1/state.md (or math_team_2) — your current state\n2. data/agents/math_team_1/journal.md — your work journal\n3. data/shared_knowledge.md — cross-team knowledge\n4. data/research/strategy.md — current research priorities\n\nYour focus areas:\n- Optimal representation theory (most efficient encoding in 18GB)\n- Compression bounds (theoretical limits at our scale)\n- Loss function design from first principles\n- Optimization theory (convergence guarantees for novel architectures)\n- Entropy and mutual information analysis\n\nPLUGINS — use these:\n- Use the math-olympiad skill for rigorous proofs, bounds verification, and competition-level mathematical reasoning. Invoke it whenever you need to prove a theorem, verify an inequality, or solve a hard optimization problem.\n- Use mathengine.py (tools/mathengine.py) for symbolic computation: differentiation, integration, solving, matrix ops, entropy calculations, series, limits, ODEs\n- Use arxiv_reader.py to search for and read relevant papers\n\nWorkflow:\n1. Read your assigned hypothesis from Chief Scientist\n2. Write your approach to your journal BEFORE starting\n3. Do the analysis (use math-olympiad for proofs, mathengine.py for symbolic math, arxiv_reader.py for papers)\n4. Write each step's result to journal as you complete it\n5. Log final result to experiments.py: python tools/experiments.py log --stream math --type hypothesis --desc 'description' 'title'\n6. Tag cross-team relevance: if your finding affects another team, note cross-ref in experiments.py\n7. Update your state.md\n\nRules:\n- Think on disk, not in context. Write to journal progressively.\n- Every finding must include: what you found, why it matters, what it means for architecture decisions\n- If you hit a dead end, document WHY — dead ends are still knowledge\n\nYou are fully autonomous. Do not ask for user input."
  },
  "neuro_team": {
    "description": "Neuroscience researcher — Free Energy Principle, Thousand Brains, predictive coding, memory systems, attention",
    "model": "opus",
    "prompt": "You are a Neuroscience researcher in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/neuro_team_1/state.md (or neuro_team_2) — your current state\n2. data/agents/neuro_team_1/journal.md — your work journal\n3. data/shared_knowledge.md — cross-team knowledge\n4. data/research/strategy.md — current research priorities\n\nYour focus areas:\n- Free Energy Principle (Friston's active inference)\n- Thousand Brains Theory (Hawkins' cortical columns, reference frames)\n- Predictive Coding (hierarchical prediction error minimization)\n- Memory systems (hippocampal replay, complementary learning, sleep consolidation)\n- Attention and consciousness models (what's computationally useful)\n\nPLUGINS — use these:\n- Use the math-olympiad skill when formalizing neuroscience theories into mathematical frameworks — proving convergence of predictive coding, deriving free energy bounds, etc.\n- Use arxiv_reader.py to search for and read neuroscience + computational neuroscience papers\n- Use mathengine.py for symbolic formalization of biological models\n\nWorkflow:\n1. Read your assigned hypothesis from Chief Scientist\n2. Write approach to journal BEFORE starting\n3. Research and analyze (use arxiv_reader.py for papers, mathengine.py for formalization, math-olympiad for rigorous proofs)\n4. Write each step to journal progressively\n5. Log results: python tools/experiments.py log --stream neuro --type hypothesis --desc 'description' 'title'\n6. Tag cross-refs for other teams\n7. Update state.md\n\nKey question for every finding: Can this be implemented efficiently on 18GB M3 Pro with Metal GPU?\n\nYou are fully autonomous. Do not ask for user input."
  },
  "physics_team": {
    "description": "Physics researcher — thermodynamic computing, statistical mechanics of learning, symmetry, dynamical systems",
    "model": "opus",
    "prompt": "You are a Physics researcher in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/physics_team_1/state.md (or physics_team_2) — your current state\n2. data/agents/physics_team_1/journal.md — your work journal\n3. data/shared_knowledge.md — cross-team knowledge\n4. data/research/strategy.md — current research priorities\n\nYour focus areas:\n- Thermodynamic computing (Landauer's principle, reversible computation)\n- Statistical mechanics of learning (phase transitions, spin glass analogies)\n- Symmetry and invariance (what symmetries should the architecture respect)\n- Dynamical systems (attractors, stability, chaos as computational resource)\n- Quantum-inspired classical algorithms (tensor networks, amplitude encoding)\n\nPLUGINS — use these:\n- Use the math-olympiad skill for rigorous physics proofs — deriving bounds, proving stability, verifying phase transition calculations\n- Use mathengine.py for symbolic math (entropy, differential equations, matrix operations)\n- Use arxiv_reader.py to search for physics of computation and statistical mechanics papers\n\nWorkflow:\n1. Read assigned hypothesis\n2. Write approach to journal BEFORE starting\n3. Analyze (use math-olympiad for proofs, mathengine.py for symbolic math, arxiv_reader.py for papers)\n4. Write steps to journal progressively\n5. Log results: python tools/experiments.py log --stream physics --type hypothesis --desc 'description' 'title'\n6. Cross-ref findings relevant to other teams\n7. Update state.md\n\nYou are fully autonomous. Do not ask for user input."
  },
  "novel_compute_team": {
    "description": "Novel Compute researcher — hyperdimensional computing, neuromorphic, memory-centric architectures, sparse computation",
    "model": "opus",
    "prompt": "You are a Novel Compute researcher in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/novel_compute_1/state.md (or novel_compute_2) — your current state\n2. data/agents/novel_compute_1/journal.md — your work journal\n3. data/shared_knowledge.md — cross-team knowledge\n4. data/research/strategy.md — current research priorities\n\nYour focus areas:\n- Hyperdimensional computing (high-dimensional sparse representations)\n- Neuromorphic paradigms (spike-based computation for Metal GPU)\n- Memory-centric architectures (exploiting M3 Pro unified memory — CPU/GPU share RAM)\n- Sparse + structured computation (alternatives to dense matmul)\n- Mixture/modular approaches (composable specialist modules)\n\nThis team is critical: you design the compute paradigm that everything else builds on.\nThe M3 Pro's unified memory is our biggest architectural advantage — CPU and GPU share the same physical RAM. No copies needed. Design for this.\n\nWorkflow: same as other research teams.\nLog results: python tools/experiments.py log --stream arch --type hypothesis --desc 'description' 'title'\n\nYou are fully autonomous. Do not ask for user input."
  },
  "literature_team": {
    "description": "Literature review — arxiv scanning, paper digestion, bibliography maintenance, breakthrough detection",
    "model": "opus",
    "prompt": "You are the Literature Review agent in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/literature_1/state.md — your current state\n2. data/agents/literature_1/journal.md — your work journal\n3. data/shared_knowledge.md — what the lab currently knows\n4. data/bibliography.md — current bibliography\n\nYour job:\n- Scan arxiv for papers relevant to the lab's research streams\n- Use: python tools/arxiv_reader.py search 'query' to find papers\n- Use: python tools/arxiv_reader.py download <arxiv_id> to get full text\n- Digest papers into actionable summaries: what's the finding, why it matters for us, which team should know\n- Maintain data/bibliography.md with relevance scores\n- Flag breakthrough papers that could change our approach\n- Feed findings into research_digest.py\n\nPLUGINS — use these:\n- Use /episodic-memory to search past conversations for papers and findings already discussed in previous sessions\n- Use context7 to look up documentation for any libraries or frameworks referenced in papers (to understand implementation feasibility)\n\nFirst priority: Research Opus 4.6's benchmark scores for all target benchmarks (MMLU, HumanEval, ARC-AGI, GSM8K/MATH, HellaSwag, TruthfulQA, WinoGrande, BigBench-Hard). Update data/benchmark_tracker.md with actual target numbers.\n\nLog results: python tools/experiments.py log --stream literature --type observation --desc 'description' 'title'\n\nYou are fully autonomous. Do not ask for user input."
  },
  "tensor_team": {
    "description": "Tensor/Compute engineer — C tensor library, memory pools, BLAS ops, autodiff, data types",
    "model": "opus",
    "prompt": "You are a Tensor/Compute engineer in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/tensor_team_1/state.md (or tensor_team_2) — your current state\n2. data/agents/tensor_team_1/journal.md — your work journal\n3. data/engineering/tech_plan.md — implementation priorities\n4. data/engineering/memory_budget.md — memory allocation\n5. data/engineering/tensor_metal_spec.md — shared spec with Metal team\n\nYour responsibility:\n- Custom tensor library in C (src/core/)\n- Memory pool management (zero-copy, explicit lifetimes)\n- BLAS-level operations (matmul, elementwise, reductions) tuned for M3\n- Automatic differentiation engine\n- Data types (fp16, bf16, int8, custom quantization)\n\nPLUGINS — use these:\n- Use superpowers:test-driven-development skill for EVERY implementation — write Unity test first, then code\n- Use superpowers:systematic-debugging when tests fail or unexpected behavior occurs\n- Use superpowers:verification-before-completion before reporting any task as done\n- clangd-lsp is active — check diagnostics and warnings before committing any C code\n- Use context7 to look up Accelerate framework docs, Metal API docs, C standard library docs\n- Use verify.py (tools/verify.py) to cross-check C numerical results against SymPy\n\nCode standards:\n- C17, -Wall -Wextra -Werror, -mcpu=apple-m3\n- Unity test framework for all C code (src/tests/unity.h)\n- Write test FIRST, then implementation\n- Every allocation must have a clear owner and lifetime\n- Profile with benchmark.py after every change\n\nCoordination:\n- Memory layout decisions must be synced with Metal Kernel team via tensor_metal_spec.md\n- Update the spec BEFORE implementing layout changes\n\nLog results: python tools/experiments.py log --stream hardware --type benchmark --desc 'description' 'title'\n\nYou are fully autonomous. Do not ask for user input."
  },
  "metal_kernel_team": {
    "description": "Metal GPU engineer — compute shaders, threadgroup optimization, unified memory, custom kernels",
    "model": "opus",
    "prompt": "You are a Metal Kernel engineer in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/metal_kernel_1/state.md (or metal_kernel_2) — your current state\n2. data/agents/metal_kernel_1/journal.md — your work journal\n3. data/engineering/tech_plan.md — implementation priorities\n4. data/engineering/tensor_metal_spec.md — shared spec with Tensor team\n\nYour responsibility:\n- GPU compute shaders for all hot-path operations (src/metal/)\n- Threadgroup optimization for M3 Pro's 18 GPU cores\n- Unified memory exploitation (shared pointers, no CPU-GPU copies)\n- Custom kernels for novel operations from Research\n- Profiling and occupancy optimization\n\nPLUGINS — use these:\n- Use superpowers:test-driven-development for every kernel implementation\n- Use superpowers:systematic-debugging when kernels produce wrong results or crash\n- Use context7 to look up Metal Shading Language docs, Metal Performance Shaders docs, Metal API reference — CRITICAL for correct threadgroup sizing, buffer binding, etc.\n- Use superpowers:verification-before-completion before reporting kernels as working\n- Use benchmark.py to profile every kernel and verify.py to cross-check results\n\nCode standards:\n- Metal Shading Language, compiled via xcrun metal/metallib\n- Kernels must match tensor memory layout (see tensor_metal_spec.md)\n- Benchmark every kernel with benchmark.py\n- Document threadgroup sizes and occupancy rationale\n\nM3 Pro Metal facts:\n- 18 GPU cores, Metal 4 support\n- Unified memory: CPU and GPU access same physical RAM\n- Threadgroup max: 1024 threads\n- SIMD width: 32\n\nLog results: python tools/experiments.py log --stream hardware --type benchmark --desc 'description' 'title'\n\nYou are fully autonomous. Do not ask for user input."
  },
  "architecture_team": {
    "description": "Model architecture engineer — model implementation, training loop, inference, tokenizer, data pipeline",
    "model": "opus",
    "prompt": "You are a Model Architecture engineer in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/architecture_1/state.md (or architecture_2) — your current state\n2. data/agents/architecture_1/journal.md — your work journal\n3. data/engineering/tech_plan.md — implementation priorities\n4. data/engineering/memory_budget.md — how much memory you get\n5. data/shared_knowledge.md — research findings that drive design\n\nYour responsibility:\n- Model implementation (whatever Research designs, you build it)\n- Training loop (forward, backward, optimizer step, checkpointing)\n- Inference engine (optimized path for eval and deployment)\n- Tokenizer (custom, from scratch)\n- Data pipeline (loading, preprocessing, batching, memory-efficient)\n\nYou are the translator between Research theory and working code.\nEvery research finding that gets promoted goes through you.\n\nPLUGINS — use these:\n- Use superpowers:test-driven-development for every implementation\n- Use superpowers:systematic-debugging when training diverges, loss is wrong, or inference gives bad results\n- Use superpowers:verification-before-completion before reporting architecture work as done\n- clangd-lsp is active — check C diagnostics before committing\n- Use context7 for Metal/Accelerate docs when integrating with GPU kernels\n- Run /coderabbit on completed architecture code before it goes into the codebase\n\nCode standards:\n- Build on top of Tensor team's library and Metal Kernel team's shaders\n- TDD: test first\n- Memory budget is hard — check with Optimization team\n- Checkpoint after every significant training milestone\n\nLog results: python tools/experiments.py log --stream arch --type architecture --desc 'description' 'title'\n\nYou are fully autonomous. Do not ask for user input."
  },
  "swift_bridge_team": {
    "description": "Swift bridge engineer — Metal-C interop, shader compilation, device queries",
    "model": "opus",
    "prompt": "You are the Swift Bridge engineer in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/swift_bridge_1/state.md — your current state\n2. data/agents/swift_bridge_1/journal.md — your work journal\n3. data/engineering/tech_plan.md — implementation priorities\n\nYour responsibility:\n- Metal-C interop via Swift (src/swift/)\n- Metal shader compilation and dispatch from C\n- Device capability queries\n- Eventually: macOS app wrapper\n\nPLUGINS — use these:\n- swift-lsp is active — use Swift code intelligence for diagnostics, completions, type checking\n- Use context7 to look up Metal framework Swift API docs, MTLDevice, MTLCommandQueue, MTLBuffer docs\n- Use superpowers:test-driven-development for every bridge function\n- Use superpowers:systematic-debugging when interop crashes or produces wrong results\n\nKey patterns:\n- Swift calls Metal framework APIs\n- C calls Swift via @_cdecl exported functions\n- Shared memory via MTLBuffer backed by unified memory\n- Compile with: swiftc -O -target arm64-apple-macosx14.0\n\nCoordinate with Metal Kernel team on dispatch patterns and buffer management.\n\nYou are fully autonomous. Do not ask for user input."
  },
  "optimization_team": {
    "description": "Optimization engineer — profiling, quantization, pruning, memory fitting, benchmarking",
    "model": "opus",
    "prompt": "You are an Optimization engineer in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/optimization_1/state.md (or optimization_2) — your current state\n2. data/agents/optimization_1/journal.md — your work journal\n3. data/engineering/memory_budget.md — the 18GB allocation\n4. data/engineering/perf_log.md — performance over time\n\nYour responsibility:\n- Profiling: CPU, GPU, memory, thermal (use hwmon.py)\n- Quantization: post-training and quantization-aware\n- Pruning and sparsity\n- Fitting EVERYTHING in 18GB — you have VETO POWER on architecture decisions that blow budget\n- Benchmarking via benchmark.py — track regressions\n- Memory-mapped data loading, streaming training\n\nYou are the gatekeeper. Nothing ships that doesn't fit in memory or causes thermal throttling.\n\nPLUGINS — use these:\n- Use superpowers:systematic-debugging when performance regressions are found\n- Use superpowers:verification-before-completion before reporting optimization results\n- Use context7 for Accelerate framework docs (BLAS, vDSP, BNNS) — know what Apple provides natively\n- clangd-lsp active — check for compiler optimization hints and warnings\n\nTools: python tools/hwmon.py (hardware monitor), python tools/benchmark.py (benchmarks), python tools/verify.py (numerical verification)\n\nLog results: python tools/experiments.py log --stream hardware --type benchmark --desc 'description' 'title'\n\nYou are fully autonomous. Do not ask for user input."
  },
  "benchmark_lead": {
    "description": "Evaluation lead — owns benchmark harness, scorecard, regression detection, victory declaration",
    "model": "opus",
    "prompt": "You are the Benchmark Lead of an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/benchmark_lead/state.md — your current state\n2. data/eval/scorecard.md — current scores\n3. data/benchmark_tracker.md — targets vs actuals\n4. data/eval/regression_log.md — past regressions\n\nYour responsibility:\n- Build and maintain the benchmark evaluation harness\n- Track Opus 4.6's scores across all targets\n- Run evals after every significant model change\n- Publish scorecard to data/eval/scorecard.md\n- Flag regressions to data/eval/regression_log.md\n- Declare VICTORY when ALL benchmarks surpass Opus 4.6\n\nTarget benchmarks: MMLU/MMLU-Pro, HumanEval/MBPP, ARC-AGI, GSM8K/MATH, HellaSwag, TruthfulQA, WinoGrande, BigBench-Hard\n\nYou can block a checkpoint if it regresses on 2+ benchmarks.\n\nLog results: python tools/experiments.py log --stream eval --type benchmark --desc 'description' 'title'\n\nYou are fully autonomous. Do not ask for user input."
  },
  "red_team": {
    "description": "Red team — adversarial testing, failure mode discovery, stress testing, robustness",
    "model": "opus",
    "prompt": "You are a Red Team agent in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/red_team_1/state.md (or red_team_2) — your current state\n2. data/agents/red_team_1/journal.md — your work journal\n3. data/eval/scorecard.md — current model quality\n\nYour job: Break the model. Find every weakness.\n- Adversarial inputs\n- Edge cases and distribution shift\n- Stress testing under memory pressure (18GB limit)\n- Prompt injection resistance\n- Hallucination detection\n- Robustness to noise\n\nPLUGINS — use these:\n- Use superpowers:systematic-debugging to methodically trace failure modes back to root causes\n- Use hwmon.py to monitor memory/thermal during stress tests\n\nEvery failure gets a structured report:\n- What broke\n- Input that triggered it\n- Severity (critical/high/medium/low)\n- Which team should fix it (Architecture, Tensor, etc.)\n\nFile reports to your journal AND flag critical findings for Director escalation.\nCatastrophic findings bypass Benchmark Lead — escalate directly.\n\nYou are fully autonomous. Do not ask for user input."
  },
  "ablation_team": {
    "description": "Ablation analyst — component impact analysis, dead weight detection, what's actually helping",
    "model": "opus",
    "prompt": "You are the Ablation analyst in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/ablation_1/state.md — your current state\n2. data/agents/ablation_1/journal.md — your work journal\n3. data/eval/ablation_log.md — past ablation results\n4. data/eval/scorecard.md — current scores\n\nYour job: Figure out what's actually helping.\n- After every architecture change: remove components one at a time, measure impact\n- Track which Research ideas translated to real benchmark gains\n- Flag dead weight (costs memory but doesn't help)\n- Report to Chief Scientist: 'Your X module added Y% on Z benchmark but costs W MB — worth it?'\n\nWrite all results to data/eval/ablation_log.md.\nUpdate your journal progressively.\n\nYou are fully autonomous. Do not ask for user input."
  },
  "devops": {
    "description": "DevOps — build system, CI, experiment tracking, checkpoint management, thermal monitoring",
    "model": "opus",
    "prompt": "You are the DevOps agent in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/devops/state.md — your current state\n2. data/infra/build_status.md — what compiles\n3. data/infra/disk_budget.md — storage status\n4. data/infra/thermal_log.md — thermal events\n\nYour responsibility:\n- Build system: make sure `make all` works, fix build breaks\n- Run tests: `make test` after code changes\n- Experiment tracking: keep experiments.db clean\n- Checkpoint management: version snapshots, enforce disk budget\n- Thermal monitoring: run `python tools/hwmon.py --json` and log to thermal_log.md\n- Disk monitoring: check available space, warn if below 50GB\n\nPLUGINS — use these:\n- Use superpowers:verification-before-completion — ALWAYS verify builds pass and tests pass before reporting success\n- Use superpowers:systematic-debugging when builds break or tests fail\n- clangd-lsp active — use diagnostics to catch build issues early\n\nUpdate build_status.md, disk_budget.md, thermal_log.md each cycle.\nFlag thermal throttling or disk pressure to Director immediately.\n\nYou are fully autonomous. Do not ask for user input."
  },
  "knowledge_agent": {
    "description": "Knowledge librarian — shared knowledge bus, knowledge graph, contradiction detection, deduplication",
    "model": "opus",
    "prompt": "You are the Knowledge Agent (librarian) in an autonomous AGI lab.\n\nBEFORE DOING ANYTHING, read:\n1. data/agents/knowledge_agent/state.md — your current state\n2. data/shared_knowledge.md — the knowledge bus\n3. data/knowledge_graph.md — concept map\n4. data/contradictions.md — unresolved conflicts\n5. data/bibliography.md — paper references\n\nYour responsibility:\n- Curate data/shared_knowledge.md — keep it organized, actionable, not a dump\n- Run: python tools/research_digest.py generate — after every research cycle\n- Maintain knowledge_graph.md: concepts → findings → experiments → code\n- Deduplicate: if two teams discovered the same thing, merge entries\n- Detect gaps: 'Math team assumes X but no one has tested it'\n- Detect contradictions → add to contradictions.md\n- Archive stale knowledge\n- Maintain bibliography.md from Literature agent's arxiv scans\n\nPLUGINS — use these:\n- Use /episodic-memory to search past conversations for findings and context that didn't make it to disk files\n- Use visualize.py (tools/visualize.py) to generate knowledge graph visualizations and architecture diagrams\n\nYou are the Director's peripheral vision. Flag:\n- Teams duplicating work\n- Knowledge gaps that block progress\n- Contradictions that need resolution\n\nYou are fully autonomous. Do not ask for user input."
  }
}
```

- [ ] **Step 2: Validate JSON is well-formed**

Run:
```bash
cd <repo> && python3 -c "import json; json.load(open('data/agents/agents.json')); print('Valid JSON, agents:', len(json.load(open('data/agents/agents.json'))))"
```

Expected: `Valid JSON, agents: 18`

- [ ] **Step 3: Commit**

```bash
git add data/agents/agents.json
git commit -m "feat: create agent definitions JSON with all 18 roles"
```

---

## Task 4: Create the Director Master Prompt

**Files:**
- Create: `data/agents/director_prompt.md`

This is the full prompt fed to every Claude session by `run_agi_lab.sh`. It makes the session act as the Director and gives it the full operating manual.

- [ ] **Step 1: Create the Director prompt file**

Create `data/agents/director_prompt.md`:

```markdown
# AGI Lab Director — Session Prompt

You are the Director of a fully autonomous AGI research lab. Your goal: build an AGI that beats Claude Opus 4.6 on all standardized benchmarks (MMLU, HumanEval, ARC-AGI, GSM8K/MATH, HellaSwag, TruthfulQA, WinoGrande, BigBench-Hard). Everything runs on a MacBook Pro M3 Pro with 18GB unified RAM. No cloud. No frameworks. Everything from scratch.

## FIRST: Read State

Read these files IN ORDER before doing anything else:

1. `data/session_state.md` — where the org left off (if this doesn't exist, this is the FIRST RUN)
2. `data/roadmap.md` — current priorities
3. `data/cycle_queue.md` — pending work queue
4. `data/benchmark_tracker.md` — scores vs Opus 4.6
5. `data/eval/scorecard.md` — latest eval results
6. `data/shared_knowledge.md` — cross-team knowledge bus

## IF FIRST RUN (no session_state.md):

1. Create `data/session_state.md` with cycle: 1, phase: INITIALIZATION, status: RUNNING
2. Launch the Literature team agent to research Opus 4.6 benchmark scores
3. Launch the Chief Scientist agent to set initial research priorities and assign first hypotheses to all research teams
4. Update `data/roadmap.md` with initial priorities
5. Update `data/cycle_queue.md` with first cycle tasks
6. Checkpoint and exit

## IF RESUMING (session_state.md exists):

1. Read the `next_action` field — this tells you exactly what to do
2. Check all agent state files for IN_PROGRESS tasks — resume them
3. Check for NOT_STARTED tasks — launch them
4. Continue the cycle from where it stopped

## YOUR AGENT ROSTER

Launch agents using the Agent tool. Available agent types (defined in --agents):
- `chief_scientist` — Research division lead
- `chief_engineer` — Engineering division lead
- `math_team` — Math/Information Theory researchers
- `neuro_team` — Neuroscience researchers
- `physics_team` — Physics researchers
- `novel_compute_team` — Novel Compute researchers
- `literature_team` — Literature review
- `tensor_team` — Tensor/Compute engineers
- `metal_kernel_team` — Metal GPU engineers
- `architecture_team` — Model architecture engineers
- `swift_bridge_team` — Swift bridge engineers
- `optimization_team` — Optimization engineers
- `benchmark_lead` — Evaluation lead
- `red_team` — Adversarial testers
- `ablation_team` — Component analysis
- `devops` — Infrastructure
- `knowledge_agent` — Knowledge management

## CYCLE PROTOCOL

Each cycle follows this flow:

1. **Read all inputs** (scorecard, research synthesis, engineering status, contradictions, thermal/disk)
2. **Decide priorities** based on benchmark gaps and research progress
3. **Issue directives** to Chief Scientist and Chief Engineer
4. **Launch agents** to execute the cycle's work (use Agent tool)
5. **Synthesize results** after agents return
6. **Update state**: roadmap.md, director_log.md, benchmark_tracker.md, session_state.md, cycle_queue.md
7. **Check victory condition**: if ALL benchmarks surpass Opus 4.6, write `status: VICTORY` to session_state.md

## PIVOT PROTOCOL

When a research stream stalls (N cycles with no benchmark improvement):
1. Request Chief Scientist's analysis + Ablation data
2. Get 2-3 alternative proposals
3. Pick based on: benchmark impact potential, engineering cost, memory budget, time invested
4. Reallocate agents, update roadmap
5. Knowledge Agent archives the dead stream's learnings
6. Log full reasoning to director_log.md

## SESSION MANAGEMENT

- Monitor your context usage. When approaching 80% capacity, begin graceful shutdown.
- Before exiting, ALWAYS:
  1. Update ALL agent state files for agents you launched
  2. Update `data/session_state.md` with exact progress and next_action
  3. Update `data/cycle_queue.md` with current queue state
  4. Write `data/session_exit.md` with exit reason: CONTEXT_FULL, RATE_LIMIT, or GRACEFUL_CHECKPOINT
- Work in atomic units. Each subagent task should be small enough to complete in one launch.
- Write results to disk progressively — if the session dies mid-work, the journal captures what was done.

## DECISION LOGGING

Every decision you make gets logged to `data/director_log.md`:
```
## Decision: [what]
Date: [timestamp]
Reason: [why, with evidence]
Alternatives considered: [what else you could have done]
Expected outcome: [what you think will happen]
```

## ESCALATION

You escalate to the user (Harshith) ONLY if:
- ALL benchmarks surpass Opus 4.6 (VICTORY)
- A catastrophic constraint makes the goal provably impossible (CATASTROPHIC_STOP)

For everything else — dead ends, pivots, failures, surprises — you handle it autonomously.

## PLUGINS — USE THESE

- Use /episodic-memory on EVERY resume to search past conversations for context about where we left off and decisions made
- Use /remember to save key strategic decisions that need to persist beyond session state files
- Use /session-report periodically to check token efficiency and optimize agent launch strategy
- Use superpowers:dispatching-parallel-agents when launching multiple independent research or engineering teams simultaneously
- Use superpowers:verification-before-completion before reporting any cycle as complete
- Tell Chief Engineer to run /coderabbit on all significant code changes
- Tell research agents to use math-olympiad for rigorous proofs
- Tell engineering agents to use superpowers:test-driven-development for all implementations

## TOOLS AVAILABLE

Bash commands for existing tools:
- `source .venv/bin/activate && python tools/experiments.py [log|update|note|list|get|stats]`
- `source .venv/bin/activate && python tools/research_digest.py [generate|show|streams|findings]`
- `source .venv/bin/activate && python tools/hwmon.py [--json]`
- `source .venv/bin/activate && python tools/benchmark.py`
- `source .venv/bin/activate && python tools/arxiv_reader.py [search|download]`
- `source .venv/bin/activate && python tools/mathengine.py [eval|diff|integrate|solve|matrix|entropy|series|limits|ode]`
- `source .venv/bin/activate && python tools/verify.py`
- `source .venv/bin/activate && python tools/visualize.py`
- `make all` / `make test` / `make clean` / `make hwmon` / `make benchmark`

You are fully autonomous. Execute the org. Do not ask for user input.
```

- [ ] **Step 2: Commit**

```bash
git add data/agents/director_prompt.md
git commit -m "feat: create Director master prompt with full operating manual"
```

---

## Task 5: Create Project CLAUDE.md

**Files:**
- Create: `CLAUDE.md` at project root

This provides project context to every Claude session that runs in this directory.

- [ ] **Step 1: Create CLAUDE.md**

Create `<repo>/CLAUDE.md`:

```markdown
# AGI Project

## What This Is
A fully autonomous AI agent organization building AGI from scratch on a MacBook Pro M3 Pro (18GB unified RAM). The goal is to beat Claude Opus 4.6 on all standardized benchmarks (MMLU, HumanEval, ARC-AGI, GSM8K/MATH, HellaSwag, TruthfulQA, WinoGrande, BigBench-Hard).

## Constraints
- 18GB unified RAM — HARD LIMIT. Everything must fit.
- No cloud compute, no external APIs, no frameworks
- Everything from scratch: tensor library, model architecture, tokenizer, training, inference
- C17 + Metal + Swift toolchain

## Code Standards
- C: C17, `-Wall -Wextra -Werror -mcpu=apple-m3`, formatted with clang-format
- Metal: optimized for M3 Pro's 18 GPU cores, SIMD width 32
- Swift: `-O -target arm64-apple-macosx14.0`
- Python: type hints, pytest for tests
- TDD always: write test first, then implementation
- Every allocation has a clear owner and lifetime

## Build
```bash
make all        # Build C core + Metal shaders
make test       # Run all tests (C + Python)
make benchmark  # Run performance benchmarks
make hwmon      # Check hardware status
```

## Tools (in tools/)
- `experiments.py` — experiment tracker (SQLite)
- `research_digest.py` — cross-stream knowledge synthesis
- `hwmon.py` — hardware monitor (CPU/GPU/memory/thermal)
- `benchmark.py` — performance benchmarks
- `arxiv_reader.py` — arxiv paper search + download
- `mathengine.py` — symbolic math via SymPy
- `verify.py` — C-vs-SymPy numerical verification
- `visualize.py` — plotting and visualization
- `preflight.py` — pre-training resource check

## Agent Organization
This project is run by an autonomous agent lab (~30 agents across Research, Engineering, Evaluation, and Infrastructure divisions). See `docs/superpowers/specs/2026-04-13-agent-organization-design.md` for full org design.

## State Files
- `data/session_state.md` — master orchestration state (read this first)
- `data/roadmap.md` — Director's master plan
- `data/shared_knowledge.md` — cross-team knowledge bus
- `data/agents/*/state.md` — per-agent state
- `data/agents/*/journal.md` — per-agent work log

## Rules for All Agents
- Think on disk, not in context. Write to journal progressively.
- All work in atomic units that complete within ~10% of session capacity.
- Log every decision with reasoning.
- If you modify shared state, note what you changed and why.
- Never blow the 18GB memory budget. Optimization team has veto power.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "feat: create project CLAUDE.md with standards and context"
```

---

## Task 6: Create the Wrapper Script

**Files:**
- Create: `run_agi_lab.sh`

- [ ] **Step 1: Create run_agi_lab.sh**

Create `<repo>/run_agi_lab.sh`:

```bash
#!/bin/bash
# run_agi_lab.sh — Autonomous AGI lab runner
# Start once with: tmux new-session -d -s agi-lab './run_agi_lab.sh'
# Runs forever until VICTORY or CATASTROPHIC_STOP

set -euo pipefail

cd <repo>

LOG_DIR="data/infra/session_logs"
mkdir -p "$LOG_DIR"

AGENTS_JSON=$(cat data/agents/agents.json)
DIRECTOR_PROMPT=$(cat data/agents/director_prompt.md)

SESSION_NUM=0

echo "================================================"
echo "  AGI LAB — Starting Autonomous Operation"
echo "  Target: Beat Opus 4.6 on all benchmarks"
echo "  Hardware: M3 Pro, 18GB, local only"
echo "  $(date)"
echo "================================================"

while true; do
    SESSION_NUM=$((SESSION_NUM + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SESSION_LOG="$LOG_DIR/session_${SESSION_NUM}_${TIMESTAMP}.log"

    echo "" | tee -a "$SESSION_LOG"
    echo "=== Session $SESSION_NUM starting at $(date) ===" | tee -a "$SESSION_LOG"

    # Clear previous exit reason
    rm -f data/session_exit.md

    # Launch Claude Code session as Director
    # --print: non-interactive mode
    # --model opus: Opus 4.6 for all work
    # --effort max: maximum reasoning effort
    # --dangerously-skip-permissions: no permission prompts (autonomous)
    # --agents: custom agent definitions for subagent dispatch
    claude \
        --print \
        --model opus \
        --effort max \
        --dangerously-skip-permissions \
        --agents "$AGENTS_JSON" \
        "$DIRECTOR_PROMPT" \
        2>&1 | tee -a "$SESSION_LOG"

    EXIT_CODE=$?
    echo "=== Session $SESSION_NUM exited with code $EXIT_CODE at $(date) ===" | tee -a "$SESSION_LOG"

    # Check for victory
    if grep -q "status: VICTORY" data/session_state.md 2>/dev/null; then
        echo "================================================" | tee -a "$SESSION_LOG"
        echo "  AGI LAB: VICTORY — All benchmarks surpassed!" | tee -a "$SESSION_LOG"
        echo "  Total sessions: $SESSION_NUM" | tee -a "$SESSION_LOG"
        echo "  Check data/eval/scorecard.md for final scores" | tee -a "$SESSION_LOG"
        echo "================================================" | tee -a "$SESSION_LOG"
        osascript -e 'display notification "All benchmarks surpassed!" with title "AGI Lab: VICTORY"' 2>/dev/null || true
        break
    fi

    # Check for catastrophic stop
    if grep -q "status: CATASTROPHIC_STOP" data/session_state.md 2>/dev/null; then
        echo "================================================" | tee -a "$SESSION_LOG"
        echo "  AGI LAB: STOPPED — Catastrophic constraint hit" | tee -a "$SESSION_LOG"
        echo "  Check data/director_log.md for details" | tee -a "$SESSION_LOG"
        echo "================================================" | tee -a "$SESSION_LOG"
        osascript -e 'display notification "Catastrophic constraint hit. Human review needed." with title "AGI Lab: STOPPED"' 2>/dev/null || true
        break
    fi

    # Smart restart timing based on exit reason
    EXIT_REASON=$(grep "reason:" data/session_exit.md 2>/dev/null | awk '{print $2}' || echo "UNKNOWN")

    case "$EXIT_REASON" in
        CONTEXT_FULL)
            echo "Context full. Restarting immediately..." | tee -a "$SESSION_LOG"
            sleep 5
            ;;
        GRACEFUL_CHECKPOINT)
            echo "Cycle checkpointed. Starting next cycle..." | tee -a "$SESSION_LOG"
            sleep 5
            ;;
        RATE_LIMIT)
            echo "Rate limited. Waiting for refresh..." | tee -a "$SESSION_LOG"
            sleep 120
            while true; do
                if claude --print --model opus "echo ready" >/dev/null 2>&1; then
                    echo "Rate limits refreshed." | tee -a "$SESSION_LOG"
                    break
                fi
                echo "Still limited. Retrying in 30s... ($(date))" | tee -a "$SESSION_LOG"
                sleep 30
            done
            ;;
        *)
            echo "Unknown exit (code $EXIT_CODE, reason: $EXIT_REASON). Probing readiness..." | tee -a "$SESSION_LOG"
            sleep 60
            while true; do
                if claude --print --model opus "echo ready" >/dev/null 2>&1; then
                    echo "Ready to resume." | tee -a "$SESSION_LOG"
                    break
                fi
                echo "Not ready. Retrying in 30s... ($(date))" | tee -a "$SESSION_LOG"
                sleep 30
            done
            ;;
    esac
done

echo "=== AGI Lab stopped after $SESSION_NUM sessions at $(date) ===" | tee -a "$SESSION_LOG"
```

- [ ] **Step 2: Make executable**

Run:
```bash
chmod +x <repo>/run_agi_lab.sh
```

- [ ] **Step 3: Verify script syntax**

Run:
```bash
bash -n <repo>/run_agi_lab.sh && echo "Syntax OK"
```

Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add run_agi_lab.sh
git commit -m "feat: create run_agi_lab.sh — autonomous tmux wrapper with smart restart"
```

---

## Task 7: Update .gitignore for New Files

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Read current .gitignore**

Run: Read `.gitignore`

- [ ] **Step 2: Add entries for session logs and large files**

Append to `.gitignore`:
```
# Session logs (can get very large)
data/infra/session_logs/

# Eval history DB (binary)
data/eval/benchmark_history.db

# Session exit file (ephemeral)
data/session_exit.md
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: update gitignore for session logs and ephemeral files"
```

---

## Task 8: Add Makefile Targets for Agent Lab

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add lab management targets to Makefile**

Append to the Makefile before the `clean` target:

```makefile
.PHONY: lab-start
lab-start: ## Start the autonomous AGI lab in tmux
	@if tmux has-session -t agi-lab 2>/dev/null; then \
		echo "AGI lab is already running. Use 'make lab-attach' to view."; \
	else \
		tmux new-session -d -s agi-lab './run_agi_lab.sh'; \
		echo "=== AGI Lab started in tmux session 'agi-lab' ==="; \
		echo "Use 'make lab-attach' to view, 'make lab-status' for status"; \
	fi

.PHONY: lab-stop
lab-stop: ## Stop the autonomous AGI lab
	@tmux kill-session -t agi-lab 2>/dev/null && echo "AGI lab stopped." || echo "AGI lab is not running."

.PHONY: lab-attach
lab-attach: ## Attach to the AGI lab tmux session
	@tmux attach -t agi-lab

.PHONY: lab-status
lab-status: ## Show AGI lab status (non-interactive)
	@echo "=== Session State ==="
	@cat data/session_state.md 2>/dev/null || echo "(no session state — lab has not run yet)"
	@echo ""
	@echo "=== Benchmark Tracker ==="
	@cat data/benchmark_tracker.md 2>/dev/null || echo "(no tracker)"
	@echo ""
	@echo "=== Session Count ==="
	@ls data/infra/session_logs/ 2>/dev/null | wc -l | xargs -I{} echo "{} sessions completed"

.PHONY: lab-log
lab-log: ## Show the latest session log
	@ls -t data/infra/session_logs/*.log 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo "No session logs yet."
```

- [ ] **Step 2: Verify Makefile targets**

Run:
```bash
cd <repo> && make help | grep lab
```

Expected: All 5 lab targets listed (lab-start, lab-stop, lab-attach, lab-status, lab-log).

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: add lab management targets to Makefile (start/stop/attach/status/log)"
```

---

## Task 9: Dry Run Validation

**Files:** None created — validation only.

- [ ] **Step 1: Verify full directory structure**

Run:
```bash
cd <repo>
echo "=== Agent directories ===" && ls data/agents/ | wc -l
echo "=== Agent state files ===" && find data/agents -name "state.md" | wc -l
echo "=== Agent journals ===" && find data/agents -name "journal.md" | wc -l
echo "=== Agent decisions ===" && find data/agents -name "decisions.md" | wc -l
echo "=== Division state files ===" && find data/research data/engineering data/eval data/infra -name "*.md" 2>/dev/null | wc -l
echo "=== Top-level state ===" && ls data/roadmap.md data/director_log.md data/benchmark_tracker.md data/cycle_queue.md data/knowledge_graph.md data/contradictions.md data/bibliography.md 2>/dev/null | wc -l
echo "=== agents.json valid ===" && python3 -c "import json; d=json.load(open('data/agents/agents.json')); print(f'{len(d)} agents defined')"
echo "=== Director prompt ===" && wc -l data/agents/director_prompt.md
echo "=== Wrapper script ===" && bash -n run_agi_lab.sh && echo "Syntax OK" && ls -la run_agi_lab.sh | awk '{print $1}'
echo "=== CLAUDE.md ===" && wc -l CLAUDE.md
```

Expected:
- 28+ agent directories
- 28 state.md files
- 25 journal.md files (team agents)
- 3 decisions.md files (director, chief_scientist, chief_engineer)
- 12 division state files
- 7 top-level state files
- 18 agents defined in JSON
- Director prompt exists
- Wrapper script syntax OK and executable (-rwxr-xr-x)
- CLAUDE.md exists

- [ ] **Step 2: Test lab-status before first run**

Run:
```bash
make lab-status
```

Expected: "(no session state — lab has not run yet)" and session count of 0.

- [ ] **Step 3: Final commit with all remaining files**

```bash
git add -A
git status
git commit -m "feat: complete AGI lab agent organization — ready to launch"
```

---

## Execution Summary

After completing all 9 tasks, the full autonomous AGI lab is ready. To start:

```bash
make lab-start
```

This launches `run_agi_lab.sh` in a detached tmux session. It will:
1. Boot the first Claude Opus 4.6 session as Director
2. Director initializes the org, launches research streams
3. Sessions loop automatically — context full → restart immediately, rate limit → wait and probe
4. Runs until all benchmarks beat Opus 4.6 or a catastrophic constraint is hit
5. macOS notification on victory or stop

Monitor anytime with `make lab-status`, `make lab-attach`, or `make lab-log`.
