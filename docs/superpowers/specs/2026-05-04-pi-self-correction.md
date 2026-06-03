# PI Self-Correction Mechanisms

**Date:** 2026-05-04
**Status:** Approved (all 3 mechanisms, with mitigation on #3)
**Source:** Self-correction patterns from Reflexion (Shinn et al. 2023), calibration research (OpenAI Sept 2025 hallucination paper), and hard-budget patterns from agent literature.

## Goal

Add three structured self-correction mechanisms to PI's workflow that *deepen the organic-learning property* of the lab without prescribing decisions. Each mechanism captures judgment for feedback rather than replacing judgment with rules.

## The three mechanisms

### #1 Reflexion-style program retrospective

**What:** At every program close, PI fills out a structured retrospective template answering: (a) Would I run this program again with hindsight? (b) What's the highest-leverage transferable lesson? (c) What pattern transfers to the next program?

**Why organic:** Doesn't prescribe what to conclude. Captures PI's judgment in a structured form so it accumulates in `data/agents/pi/semantic.md` as transferable lessons. Over many programs, PI carries forward what worked.

**Implementation:**
- Create `data/agents/_shared/program_retrospective_template.md` (the 3-question template)
- Modify `data/agents/pi/procedural.md` — add requirement to fill out template at program close, file output at `data/agents/pi/episodic/<YYYY-MM-DD>_<program>_retrospective.md`, summarize transferable lessons into `pi/semantic.md`
- Test: `tests/test_pi_retrospective_template.py` validates the template exists + PI procedural references it

### #2 Calibrated confidence on gate verdicts

**What:** At every PI gate verdict (program selection, phase-close approval, paper approval, sub-step approval), PI attaches `confidence: 0.XX` (a probability) and `calibration_reasoning: <why this number>`. Verifier (lab_architect) checks calibration is present and isn't obviously miscalibrated relative to evidence.

**Why most-organic:** PI's calibration improves *through feedback*. Telemetry records each (decision_id, claim, confidence) pair. When outcomes are known later (program closes successfully, paper accepted/rejected, sub-step holds or breaks), the actual hit rate at each confidence band gets compared to claimed. PI gets monthly calibration reports — "you said 70% confident on these 10 things, 5 hit = 50% actual." PI adjusts its claims downward if overconfident, upward if underconfident, all without external prescription.

**Implementation:**
- Modify `data/agents/pi/procedural.md` — add "calibrated confidence" section
- Modify `data/agents/_shared/verifier_pairs.json` — add note in `pi` entry that verifier must check calibration is present
- Create `tools/calibration_logger.py` — `log_calibration(decision_id, claim, confidence, reasoning)` appends to `data/infra/calibration_telemetry.jsonl`
- Create `tools/calibration_rollup.py` — when outcomes are known (read from log.md), produces per-confidence-band actual hit rate; flags "consistently overconfident" or "consistently underconfident"
- Test: `tests/test_calibration_logger.py` validates schema; integration test validates rollup math

### #3 Hard cycle budgets per program (with mitigation)

**What:** Each program has `max_cycles_without_close` (default 12) and `max_total_cycles` (default 30). When tripped, PI must respond at next gate with one of: (a) propose phase close, (b) propose pivot, (c) propose documented extension with explicit reason. Runner injects a notice into Director's brief when over budget.

**Mitigation against prescriptive feel:** PI can extend the budget any time with `extend_reason` logged to `data/programs/<name>/budget_overrides.md`. The tripwire forces a *decision moment*, not a closure decision. PI judgment is still in charge; the budget just says "this can't drift indefinitely without explicit choice."

**Implementation:**
- Create `data/programs/budgets.json` — top-level config of default budgets, can be overridden per-program
- Modify `tools/brief_assembler.py` — add a "budget status" section when over budget (so Director's context_brief includes the notice)
- Modify `data/agents/pi/procedural.md` — add "budget tripwire" workflow (close vs pivot vs extend)
- Test: `tests/test_program_budgets.py` validates schema + brief_assembler surfaces the notice when over budget

## Architecture interaction

These mechanisms layer on top of the existing 7-pattern architecture:

- **#1 (Reflexion)** complements the verifier loops (Stage 5) — verifier catches confident-wrong-output on the deliverable; retrospective catches strategic-wrong-direction over the program.
- **#2 (Calibration)** is the missing piece in Pattern 7 (telemetry). We track escalation + verifier-rejection rates per role; now we also track PI's calibration accuracy over programs. Three signals → richer organic learning.
- **#3 (Budget)** complements skip-when-stable (Stage 1+ tuning C). Skip-when-stable says "don't fire Director when nothing's happening"; budget says "force a decision when something's been happening for too long without resolution."

## Out of scope (this spec)

- **External-signal monitoring** (literature scout pinging on each program's question). `literature_hunter` already does periodic scans; tighter integration is future work.
- **Strike rule on tactical loops.** Already in place via `_phase_just_ended`, holding-pattern detector, PI's existing 3-strike rule for unanimous decisions.
- **Confidence calibration across all roles.** This spec scopes calibration to PI only (where program-level decisions live). Could extend to other judgment roles (code_reviewer, paper_writer) later if useful.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Retrospectives become rote ("everything went fine") | Template asks specific questions; verifier (lab_architect) flags rote answers |
| PI gives ritualistic confidence numbers ("70% on everything") | Calibration rollup tracks accuracy at each confidence band; consistent miscalibration flagged |
| Hard budget feels prescriptive | Mitigation built in: PI can extend with reason. Tripwire = decision moment, not decision. |
| All three together = procedural bloat | Reflexion is 1 template + 1 procedural section. Calibration is 1 procedural section + 1 telemetry call. Budget is 1 config + 1 procedural section. Net ~150 lines added across 3 files. |

## Success criteria

- Retrospectives produce ≥1 transferable lesson per program (measured: lines added to `pi/semantic.md` after each program)
- Calibration accuracy reaches ±10% within 3 months (claimed 70% → actual 60-80% hit rate at the 70% confidence band)
- No program runs >30 cycles without an explicit close/pivot/extension decision (measured: no `max_total_cycles` violations without a `budget_overrides.md` entry)

## Migration

Stage-gated implementation:
- **Stage P1**: #1 Reflexion (smallest, lowest risk) — ~1h
- **Stage P2**: #3 Hard budgets (medium, includes mitigation) — ~1.5h
- **Stage P3**: #2 Calibration (most substantive, includes telemetry + rollup) — ~2.5h

Each stage commits + reviewable independently. After P3, full self-correction loop is live; calibration accuracy data accumulates over the next month.
