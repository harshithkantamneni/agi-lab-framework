# Ablation Team — Pre-Staged Dispatch Brief (Cycle 30 if re-gate fails)

**Authored:** 2026-04-17 (Cycle 29 close, D-097)
**Trigger:** Phase B re-gate PID 38002 endpoint FAILS the 4-condition gate (H ≥ 0.6 AND load-range ≤ 5:1 AND H-slope ≥ 0 AND best_ppl < 500) — i.e., a SECOND consecutive C-severe collapse under LFB + Plan B + entropy-penalty + τ-anneal.
**Author:** Director (Cycle 29)
**Dispatching:** lab_architect retro_29.md §8 P3 (ACCEPTED). This brief shelves if re-gate passes.

---

## Situation (set this context explicitly when dispatching)

Two full-50M training runs have collapsed in the 50M K=8 k=2 architecture despite escalating remediation:
- **Cycle 26** (D-094): LFB-only, `--balance-h-floor` silent no-op — killed at step ~980, entropy 0.39.
- **Cycle 28** (D-096): LFB + Plan B (Default MoE / Dense Backprop) — killed at step ~825 under scenario C-severe (H<0.45, load-range 400:1 via proxy + layer 3 = 16.84:1 directly from EMA). Plan B alone sharpened per-token more than it rescued dormant experts because LFB's bias rotation prevented persistent dormancy (the lever Plan B's mechanism (b) depends on).
- **Cycle 29** (D-097): LFB + Plan B + entropy-penalty (β_H=0.15, H_target=0.90) + τ-anneal (1.4→1.0 over 500 steps) — [UPDATE: FAILED, endpoint readings here].

Cycle 29's Rev-2 design was the first mathematically-dominant entropy countermeasure per chief_scientist's gradient-magnitude arithmetic (117% of Plan B sharpening at H=0.3 vs Rev-1's 36%). Its failure would mean the entropy-penalty mechanism itself is insufficient at this scale, OR the mechanism is sufficient but the 50M K=8 k=2 config is mal-specified. **This is the question the ablation team resolves.**

Per-lab values §6 (Question the Direction), two consecutive full 50M collapses under two independent mechanistic framings (Plan B alone; Plan B + entropy penalty) is evidence that the PROBLEM is not in the mechanism alone but may also be in the architecture. Evaluator Cycle 28 §7 risk #3 explicitly flagged this.

---

## Your job (ablation_team)

Rank 4 candidate remediations by **expected entropy-stability lift** and **implementation LOC cost** to produce a Director-actionable recommendation.

### Candidate remediations

| # | Remediation | Hypothesis | Expected lift | Est. LOC | Re-gate wall-clock |
|---|-------------|-----------|---------------|----------|---------------------|
| A | **Wider d_ff** (e.g., d_ff=2048 → 3072 per expert) | Expert capacity at d_ff=2048 is too narrow; top-1 concentration is a capacity-floor symptom, not a gating symptom | Medium — raises active-param capacity, may not prevent collapse | ~5 LOC config change | ~3.5h |
| B | **Smaller K** (K=4 k=2 at 50M) | At K=8 k=2 dormancy rescue cannot fire (LFB keeps experts "just active enough"); at K=4 k=2 the 50% active ratio changes the Plan B dormancy-lever dynamics | High (changes gradient regime) | ~5-10 LOC | ~3.5h |
| C | **Hard routing floor** (per-expert min load 0.5/K enforced, not just biased) | Soft mechanisms (bias + entropy penalty) lose to softmax sharpening at high load imbalance; hard constraint removes the sharpening escape hatch | High — structurally eliminates collapse | ~40-80 LOC new router path | ~3.5h |
| D | **Dense 50M baseline** (K=1 equivalent, or disable MoE entirely) | Drop MoE complexity; train dense 50M on the 5-corpus mix; establish a non-MoE anchor to bound the "what is MoE buying us" question | Very high (eliminates collapse by construction) | ~5 LOC config change | ~3.5h |

### Required output

Produce `data/findings/ablation_team_cycle30_ranking.md` containing:

1. **Ranked recommendation** (1-4 ordered, best first) with one-paragraph justification each. Explicit reasoning about WHY the ranking, not just WHAT.
2. **Gradient-magnitude or dynamics argument** for why the top-ranked remediation should not suffer the same per-token sharpening dominance Cycle 28 revealed. Cite chief_scientist's two-effect decomposition in `data/engineering/entropy_penalty_temp_anneal_design.md` §Revision 2.
3. **Known-prior evidence** from literature (arxiv, if reachable) or lab history (cite `data/killed_ideas.md` + archived shared_knowledge). Has anyone in the field faced this exact dilemma at ~50M K=8 k=2? If so, what did they choose?
4. **Integration risk** per candidate — does C (hard floor) interfere with Plan B's dense-gradient path? Does B (K=4) invalidate the LFB bias-rotation mechanism? 
5. **Budget recommendation**: pick ONE remediation for Cycle 30 to try, or recommend a split (e.g., baseline D + primary B). Director will dispose.

Include gradient math wherever possible. Do not recommend "try all of them" — that's the failure mode this brief exists to prevent.

### Data sources you may read

- `data/engineering/entropy_penalty_temp_anneal_design.md` (chief_scientist §Revision 2 — full two-effect decomposition)
- `data/findings/chief_scientist_cycle29_csevere_revisit.md` (gradient-magnitude arithmetic)
- `data/killed_ideas.md` (Cycle 13 pin-at-log(K) prior, Cycle 28 Plan B alone)
- `data/shared_knowledge.md` (current-cycle status) + `data/archives/shared_knowledge/2026-04-17_cycle_29.md` (pruned history)
- `data/evaluator_report.md` (Cycle 29 verdict; Cycle 28 archived)
- Lit scan via arxiv tools IF you think field prior exists; do not burn cycles chasing marginal hits.
- `data/experiments/cycle_29_phaseb_regate_rev2/training.log` (post-failure log — read per-layer `[load]` SUMMARY to identify WHICH layer failed first and HOW it failed; step-by-step trajectory informs ranking)
- `data/checkpoints/cycle29_phaseb_regate_rev2/*.ckpt` (per-500-step saves — the last-saved ckpt before kill is the forensic ground truth)

### Your guardrails

- **Do not modify source files** — this is a design-ranking dispatch, not an implementation dispatch.
- **Do not recommend adding new mechanisms on top of Rev-2**. The ranking is for MECHANISM REPLACEMENT or ARCHITECTURAL CHANGE, not further stacking. Rev-2 is the test of whether stacking works; if it failed, stacking more is unlikely to resolve.
- Cite file:line evidence for every architectural claim. Director expects evaluator-compatible rigor.
- Length target: 2-3 pages (~300-500 lines). No padding.

---

## Why this brief exists

Cycle 28 evaluator Top-3 risk #3 (two-consecutive collapses → architectural revisit) + lab_architect retro_29.md §8 P3. If Cycle 30 walks into a re-gate failure without this brief, it will scramble across ~2h of same-cycle design work while wall-clock is ticking. With this brief, Cycle 30 dispatches ablation_team same-cycle and pivots efficiently.

**If re-gate PASSES: shelve this brief.** It incurs zero cost and remains in the archive for any future scenario where the same question arises.
