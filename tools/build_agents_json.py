#!/usr/bin/env python3
"""Sync agents.json prompts from each role's procedural.md (source of truth).

The runner registers every subagent from data/agents/agents.json (run_agi_lab.sh
`--agents "$(cat agents.json)"`), so the `prompt` field in agents.json is the
*operative* system prompt each role actually runs with. The per-role
procedural.md files are the maintained source of truth, so agents.json must be
regenerated from them whenever a procedural.md changes — otherwise the live
roster silently runs stale instructions.

This script SYNCS rather than rebuilds from scratch:
  * `model` and `description` are preserved from the existing agents.json, so the
    per-role tier assignments (opus / sonnet / haiku) are NOT flattened and the
    full roster (e.g. `consolidator`, which has no entry in ROLES_META) is kept.
  * `prompt` is refreshed from data/agents/<role>/procedural.md.
  * a role that has a procedural.md but no agents.json entry yet is added (model
    defaults to claude-opus-4-7; override afterwards if it belongs in a lower
    tier); a role with an agents.json entry but no procedural.md keeps its
    existing prompt and is reported.

Run from the repo root: `python3 tools/build_agents_json.py`.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "agents"
AGENTS_JSON = BASE / "agents.json"
DEFAULT_MODEL = "claude-opus-4-7"

# Fallback descriptions, used only for a role that has a procedural.md but is not
# yet present in agents.json. Existing agents.json descriptions always win.
ROLES_META = {
    "pi": "Principal Investigator — scientific direction, program selection, kill calls, paper approval; unanimous-compromise with Director",
    "director": "Execution lead — agent dispatch, phase orchestration, resource allocation; unanimous-compromise with PI",
    "unanimous_compromise_mediator": "Mediator — runs only on PI-vs-Director deadlocks; proposes compromise, does not decide",
    "chief_scientist": "L2 lead — synthesizes scientific sub-specialists into coherent program narrative",
    "math_theorist": "Info theory + optimization theory + scaling laws + bounds",
    "experimental_methodologist": "Experimental design + controls + confounds + sample size + ablation plan",
    "hypothesis_generator": "Divergent hypothesis generation + formalization + falsifiability check + ranking",
    "mechanism_extractor": "First-principles explanations of observed experimental results",
    "measurement_theorist": "Metric validity + construct validity + external validity",
    "infrastructure_architect": "Apparatus design for current program — data pipeline, tooling needs, resource layout",
    "implementation_engineer_c": "C17 implementation + TDD + Unity tests + ASan/UBSan compliance",
    "sota_scout": "Continuous literature scan for applicable SOTA techniques; always-on",
    "tooling_engineer": "Program-specific measurement/profiling/analysis tools",
    "reproducibility_engineer": "Seeds + versioning + data provenance + config locks",
    "profiler": "Bottleneck identification + roofline analysis",
    "kernel_specialist": "Metal/AMX/CPU hot-path kernel optimization",
    "memory_optimizer": "Fit model + training + eval into 18GB; quantization, activation mem, gradient checkpointing",
    "scientific_reviewer": "Experimental design soundness review (P5 gate)",
    "statistical_reviewer": "Analysis validity review (P9 gate)",
    "red_team": "Adversarial reviewer; alternative explanations; stress-tests conclusions",
    "pre_reg_auditor": "Pre-registration gate (P6) + post-hoc compliance audit (P12)",
    "code_reviewer": "Correctness + TDD compliance for C/Python code",
    "literature_hunter": "Finds relevant papers (arxiv, Semantic Scholar, citation chains)",
    "paper_digester": "Deep-reads papers → per-paper digests",
    "findings_curator": "Lab semantic memory + bibliography + cross-program synthesis + lab_memory.py ingestion",
    "paper_writer": "Paper outline → prose → structure → revision",
    "figure_generator": "Publication-quality plots + diagrams + tables",
    "lab_architect": "Org health audit every 3 programs — role promotion/retirement proposals",
    "grant_reviewer": "Skeptical outsider review every 5 programs",
    "evaluator": "Per-phase rigor audit; verdict gates phase closure",
}


def main():
    existing = {}
    if AGENTS_JSON.exists():
        existing = json.loads(AGENTS_JSON.read_text())

    roster = {}
    refreshed, kept, added = [], [], []

    # 1) Preserve existing roster order; refresh prompts from procedural.md.
    for role, entry in existing.items():
        proc_path = BASE / role / "procedural.md"
        new_entry = {
            "description": entry.get("description") or ROLES_META.get(role, ""),
            "model": entry.get("model", DEFAULT_MODEL),
        }
        if proc_path.exists():
            new_entry["prompt"] = proc_path.read_text()
            refreshed.append(role)
        else:
            new_entry["prompt"] = entry.get("prompt", "")
            kept.append(role)
        roster[role] = new_entry

    # 2) Add any role that has a procedural.md but no agents.json entry yet.
    for proc_path in sorted(BASE.glob("*/procedural.md")):
        role = proc_path.parent.name
        if role in roster:
            continue
        roster[role] = {
            "description": ROLES_META.get(role, ""),
            "model": DEFAULT_MODEL,
            "prompt": proc_path.read_text(),
        }
        added.append(role)

    AGENTS_JSON.write_text(
        json.dumps(roster, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {AGENTS_JSON} with {len(roster)} roles")
    print(f"  refreshed prompt from procedural.md: {len(refreshed)}")
    if kept:
        print(f"  kept existing prompt (no procedural.md): {kept}")
    if added:
        print(f"  added new role(s) from procedural.md: {added}")


if __name__ == "__main__":
    main()
