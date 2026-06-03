# Agent Tier Audit — 2026-05-04

Per spec at `docs/superpowers/specs/2026-05-04-lab-architecture-refresh.md`.

Source-of-truth read: `data/agents/agents.json`. Total agents: 30.

## Tier assignments

| Role | Current | Proposed | Rationale |
|------|---------|----------|-----------|
| chief_scientist | claude-opus-4-7 | A (claude-opus-4-7) | synthesizes all scientific sub-specialists into coherent program narrative — judgment-tier synthesis consumed by PI/Director |
| code_reviewer | claude-opus-4-7 | A (claude-opus-4-7) | issues APPROVED/BLOCKED verdicts on C/Python code; a subtle miss (UBSan, memory ownership, TDD gap) propagates silently into results |
| director | claude-opus-4-7 | A (claude-opus-4-7) | execution lead + unanimous-compromise decisions + phase dispatch — co-equal with PI; judgment-tier |
| evaluator | claude-opus-4-7 | A (claude-opus-4-7) | per-phase rigor audit; verdict gates phase closure; output consumed by PI/Director for go/no-go |
| experimental_methodologist | claude-opus-4-7 | A (claude-opus-4-7) | designs controls, ablations, sample sizes, confound mitigation — a bad design contaminates entire program results |
| figure_generator | claude-opus-4-7 | B (claude-sonnet-4-6) | creates plots + captions from clean data; interpretive captioning warrants sonnet, but output reviewed by paper_writer/PI before publication |
| findings_curator | claude-opus-4-7 | B (claude-sonnet-4-6) | cross-program knowledge synthesis + lab_memory.py ingestion; substantive synthesis but output re-processed by PI/chief_scientist; mistakes surfaced in review |
| grant_reviewer | claude-opus-4-7 | A (claude-opus-4-7) | hostile outsider review every 5–10 programs; issues fundability verdicts; explicitly adversarial — subtle optimism would undermine the role's purpose |
| hypothesis_generator | claude-opus-4-7 | A (claude-opus-4-7) | generates and ranks falsifiable hypotheses; scientific taste required; bad hypotheses waste entire programs |
| implementation_engineer_c | claude-opus-4-7 | B (claude-sonnet-4-6) | writes C17 code with TDD + Unity tests; mistakes caught by code_reviewer + ASan/UBSan; SWE-bench gap between sonnet and opus is small |
| infrastructure_architect | claude-opus-4-7 | B (claude-sonnet-4-6) | designs data pipeline and apparatus manifest (spec, not implementation); output reviewed by Director/code_reviewer before execution |
| kernel_specialist | claude-opus-4-7 | A (claude-opus-4-7) | issues PERF_APPROVED after roofline analysis; Metal/AMX/CPU trade-offs require nuanced judgment; wrong optimization direction wastes significant cycles |
| lab_architect | claude-opus-4-7 | A (claude-opus-4-7) | org health audit + role promotion/retirement proposals every 3 programs; output consumed by PI/Director for structural decisions |
| literature_hunter | claude-opus-4-7 | B (claude-sonnet-4-6) | finds and ranks papers by relevance; explicitly does not digest deeply (paper_digester does that); relevance ranking is substantive but recoverable |
| math_theorist | claude-opus-4-7 | A (claude-opus-4-7) | works in proofs, bounds, and scaling laws; mathematical rigor requires opus-tier; an incorrect bound would mislead program direction |
| measurement_theorist | claude-opus-4-7 | A (claude-opus-4-7) | validates whether metrics measure what we claim; issues verdicts on construct validity; a wrong pass here corrupts all downstream conclusions |
| mechanism_extractor | claude-opus-4-7 | A (claude-opus-4-7) | first-principles mechanistic explanation of experimental results; scientific depth required; shallow mechanism explanations accepted as truth by paper_writer |
| memory_optimizer | claude-opus-4-7 | B (claude-sonnet-4-6) | fits models into 18GB via quantization / activation mem / gradient checkpointing; implementation-level with hard-constraint validation (OOM = caught immediately) |
| paper_digester | claude-opus-4-7 | B (claude-sonnet-4-6) | deep-reads papers → per-paper digests (claim / method / evidence / relevance / limitations); interpretive but output consumed and re-evaluated by chief_scientist / PI |
| paper_writer | claude-opus-4-7 | A (claude-opus-4-7) | writes publication-quality prose for ICLR/NeurIPS workshop tier; paper quality is a lab output deliverable; humans + peer reviewers consume directly |
| pi | claude-opus-4-7 | A (claude-opus-4-7) | scientific direction, program selection, kill calls, paper approval — top of hierarchy; unanimous-compromise with Director |
| pre_reg_auditor | claude-opus-4-7 | A (claude-opus-4-7) | gates P6 on pre-registration sufficiency + post-hoc compliance audit at P12; verdict role; a weak pass enables HARKing and invalidates the program's scientific claim |
| profiler | claude-opus-4-7 | B (claude-sonnet-4-6) | bottleneck identification + roofline analysis; produces structured reports consumed by kernel_specialist; explicitly does not optimize; analysis recoverable via kernel_specialist review |
| red_team | claude-opus-4-7 | A (claude-opus-4-7) | adversarial reviewer; finds alternative explanations and failure modes; a lenient red team defeats the purpose — needs full adversarial judgment |
| reproducibility_engineer | claude-opus-4-7 | B (claude-sonnet-4-6) | locks seeds, versions configs, hashes data files; primarily procedural engineering; compliance verifiable structurally (hash matches, config diffs) |
| scientific_reviewer | claude-opus-4-7 | A (claude-opus-4-7) | experimental design soundness gate at P5; "would a reviewer accept the method?" — gate verdict; subtle confound missed here runs unchecked through P6–P12 |
| sota_scout | claude-opus-4-7 | B (claude-sonnet-4-6) | continuous literature scan for applicable techniques; finds and flags; judgment on applicability is substantive but output re-evaluated by chief_scientist / PI before adoption |
| statistical_reviewer | claude-opus-4-7 | A (claude-opus-4-7) | analysis validity gate at P9; appropriate test selection, effect sizes, CI, multiple-testing corrections — a wrong pass corrupts published results |
| tooling_engineer | claude-opus-4-7 | B (claude-sonnet-4-6) | builds small program-specific measurement and profiling tools; implementation work; mistakes caught by tests + downstream users of the tool output |
| unanimous_compromise_mediator | claude-opus-4-7 | A (claude-opus-4-7) | runs only on PI-vs-Director deadlocks; proposes compromise; nuanced mediation of two fully-reasoned positions requires highest judgment tier |

## Tier distribution
- Tier A (opus-4-7): 17 roles
- Tier B (sonnet-4-6): 13 roles
- Tier C (haiku-4-5): 0 roles

## Notes

### Borderline flags — operator should confirm before Task 10 applies

**figure_generator (assigned B — could argue C or A):**
The role generates matplotlib/mermaid plots and markdown tables. Purely mechanical plotting from clean structured data could run on haiku. However, the prompt mandates self-contained captions and interpretive figure design. If figures go into papers directly, a subtle misrepresentation in a caption is a paper-quality error. Operator call: if figure captions are always reviewed by paper_writer before inclusion, downgrade to C is reasonable. If figures sometimes go to publications with minimal review, keep at B.

**findings_curator (assigned B — could argue A):**
The role synthesizes knowledge across programs and ingests it into lab_memory.db. The synthesis verb in the prompt is "identify key findings across programs and distill" — this is interpretive. If the curator misclassifies a finding (marks a null result as positive), that incorrect label propagates through lab memory. Conservative argument for A. Assigned B because the output is consumed and re-validated by PI/chief_scientist before affecting program direction. Operator can upgrade to A if lab memory is treated as authoritative without re-review.

**kernel_specialist (assigned A — could argue B):**
The role issues "PERF_APPROVED" after roofline analysis. This is a gate verdict on performance optimization. Assigned A because an incorrect PERF_APPROVED wastes significant engineering time on a dead-end kernel path. However, the profiler (B) produces the analysis the kernel_specialist acts on, so a Tier B kernel_specialist with a Tier A code_reviewer reviewing the optimization could work. Operator can downgrade to B if profiler reports are thorough and code_reviewer covers performance review.

**profiler (assigned B — could argue A):**
Roofline analysis and arithmetic intensity calculations require genuine technical depth. Assigned B because the output (structured bottleneck reports) is consumed by kernel_specialist who re-interprets it. Downgrade to C is not safe — roofline misreads would send kernel_specialist chasing wrong bottlenecks.

**sota_scout (assigned B — could argue A):**
Always-on literature scan with applicability judgments. The scout decides what techniques are "applicable to our stack" — this requires understanding of the current program and architecture. Assigned B conservatively (higher than pure fetching) because adoption decisions are re-validated by chief_scientist/PI. If the scout's applicability flags are acted on without re-review, upgrade to A.

### No Tier C assignments

No role was classified Tier C. The lab's smallest roles (profiler, reproducibility_engineer, tooling_engineer, literature_hunter) all require substantive technical judgment or synthesis that exceeds haiku's validated mechanical-execution profile. The lab has no pure file-ops or schema-extraction roles in agents.json. If a data-fetcher or archiver role is added in future, that would be the natural Tier C candidate.
