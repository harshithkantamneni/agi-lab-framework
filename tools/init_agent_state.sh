#!/bin/bash
# init_agent_state.sh — Initialize all agent state files with templates
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

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
