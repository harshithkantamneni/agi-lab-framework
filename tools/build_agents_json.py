#!/usr/bin/env python3
"""Build the new agents.json from 30 seed role procedural.md files."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "agents"

ROLES_META = [
    # (role, description)
    ("pi", "Principal Investigator — scientific direction, program selection, kill calls, paper approval; unanimous-compromise with Director"),
    ("director", "Execution lead — agent dispatch, phase orchestration, resource allocation; unanimous-compromise with PI"),
    ("unanimous_compromise_mediator", "Mediator — runs only on PI-vs-Director deadlocks; proposes compromise, does not decide"),
    ("chief_scientist", "L2 lead — synthesizes scientific sub-specialists into coherent program narrative"),
    ("math_theorist", "Info theory + optimization theory + scaling laws + bounds"),
    ("experimental_methodologist", "Experimental design + controls + confounds + sample size + ablation plan"),
    ("hypothesis_generator", "Divergent hypothesis generation + formalization + falsifiability check + ranking"),
    ("mechanism_extractor", "First-principles explanations of observed experimental results"),
    ("measurement_theorist", "Metric validity + construct validity + external validity"),
    ("infrastructure_architect", "Apparatus design for current program — data pipeline, tooling needs, resource layout"),
    ("implementation_engineer_c", "C17 implementation + TDD + Unity tests + ASan/UBSan compliance"),
    ("sota_scout", "Continuous literature scan for applicable SOTA techniques; always-on"),
    ("tooling_engineer", "Program-specific measurement/profiling/analysis tools"),
    ("reproducibility_engineer", "Seeds + versioning + data provenance + config locks"),
    ("profiler", "Bottleneck identification + roofline analysis"),
    ("kernel_specialist", "Metal/AMX/CPU hot-path kernel optimization"),
    ("memory_optimizer", "Fit model + training + eval into 18GB; quantization, activation mem, gradient checkpointing"),
    ("scientific_reviewer", "Experimental design soundness review (P5 gate)"),
    ("statistical_reviewer", "Analysis validity review (P9 gate)"),
    ("red_team", "Adversarial reviewer; alternative explanations; stress-tests conclusions"),
    ("pre_reg_auditor", "Pre-registration gate (P6) + post-hoc compliance audit (P12)"),
    ("code_reviewer", "Correctness + TDD compliance for C/Python code"),
    ("literature_hunter", "Finds relevant papers (arxiv, Semantic Scholar, citation chains)"),
    ("paper_digester", "Deep-reads papers → per-paper digests"),
    ("findings_curator", "Lab semantic memory + bibliography + cross-program synthesis + lab_memory.py ingestion"),
    ("paper_writer", "Paper outline → prose → structure → revision"),
    ("figure_generator", "Publication-quality plots + diagrams + tables"),
    ("lab_architect", "Org health audit every 3 programs — role promotion/retirement proposals"),
    ("grant_reviewer", "Skeptical outsider review every 5 programs"),
    ("evaluator", "Per-phase rigor audit; verdict gates phase closure"),
]

def main():
    roster = {}
    for role, description in ROLES_META:
        proc_path = BASE / role / "procedural.md"
        if not proc_path.exists():
            print(f"WARNING: missing {proc_path}")
            continue
        prompt_text = proc_path.read_text()
        roster[role] = {
            "description": description,
            "model": "claude-opus-4-7",
            "prompt": prompt_text,
        }
    out_path = BASE / "agents.json"
    with open(out_path, "w") as f:
        json.dump(roster, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path} with {len(roster)} roles")

if __name__ == "__main__":
    main()
