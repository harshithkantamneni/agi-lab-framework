# Training Digest Observer (TDO)

**Date:** 2026-05-05
**Status:** Drafted, bug-checked
**Inspired by:** Aswinkumar's Taskflow-AGENTS DigestObserver pattern (LinkedIn, 2026-05-05). Same architectural pattern — structured diagnostics that bridge LLM-generated work and deterministic downstream consumers — applied to our training-run observability.

## Problem

mechanism_extractor's P10 report (D-313) read all 12 phase-3 stdout.log files (~2.4 MB raw text) plus run_index.json plus close memos to extract per-cell mechanism flags, factorial outliers, and anomaly catalogues. Same parsing work happens repeatedly:

- Director monitoring during P8 EXECUTE (every session reads run_index + tails stdout)
- statistical_reviewer at P9 (re-derives summary metrics from raw stdout)
- mechanism_extractor at P10 (parses 12 stdouts in full)
- PI verification at P10 close (anti-forgery requires re-derivation from raw)
- paper_writer at P13 (cites training-time signals)

Every consumer pays the parsing cost. The raw stdout has structure (column-aligned step rows, `[load] step=N` MoE traces, `[entpen]` traces, summary blocks at every 500 steps, atexit summary), but each consumer re-implements the parse.

## Goal

A pure-Python heuristic parser that runs once at cell complete, emits a compact digest (~1 KB markdown + JSON sidecar) per cell plus a phase aggregate (~5 KB) per phase. Subsequent consumers read digests; raw stdout becomes fallback evidence.

## Why this is the structurally right answer

The training code (C, under our control) emits structured lines deterministically. A heuristic parser is testable, free, fast (<1s per cell). The digest schema becomes a contract: future training-code changes that break the schema fail the parser tests, forcing intentional schema migration rather than silent drift.

The architectural pattern is the same as `tools/post_director.py`: produce structured artifacts that downstream consumers can act on without re-parsing prose. Same shape, different domain (training observability instead of session close-out).

## Architecture

### 1. `tools/training_digest.py` — per-cell parser

**Inputs:**
- `data/runs/<cell_id>/stdout.log` (~200 KB)
- `data/checkpoints/<phase>/run_index.json` (cell metadata block)

**Outputs:**
- `data/digests/training/<cell_id>.md` (~1 KB human/Director-readable narrative)
- `data/digests/training/<cell_id>.json` (~3 KB structured sidecar for programmatic consumption)

**CLI:**
```bash
python3 tools/training_digest.py <cell_id> [--phase <phase>] [--regen]
```

Auto-discovers `--phase` from the cell directory if omitted. `--regen` forces re-parse even if digest mtime > stdout.log mtime.

### 2. Per-cell digest schema (JSON sidecar)

```json
{
  "schema_version": "1.0",
  "cell_id": "A42",
  "phase": "phase3_factorial",
  "config": {
    "model": "dense50m",
    "lr": 0.001,
    "seed": 42,
    "steps_planned": 5000,
    "steps_completed": 5000,
    "expected_eta_hours": 11.8,
    "actual_hours": 12.586,
    "eta_overshoot_pct": 6.7
  },
  "outcome": {
    "rc": 0,
    "atexit_clean": true,
    "fatal": 0,
    "error": 0,
    "warning": 0,
    "verdict_line": "CONVERGING -- LM loss well below random baseline (58.8% of random)",
    "best_ppl": 2.12,
    "best_loss": 0.7528,
    "recent_avg_lm": 6.1159,
    "initial_ppl": 57601.60,
    "tokens_trained": 10240000
  },
  "loss_trajectory": {
    "shape_class": "monotone_decreasing | plateau_after_N | oscillating | collapsed",
    "shape_class_evidence": "best_ppl monotone over [step_500..step_5000]; no >2x regression event",
    "step_summaries": [
      {"step": 500,  "avg_lm": 7.2595, "best_ppl": 198.35, "tokens_k": 1026, "nan": 0},
      {"step": 1000, "avg_lm": 6.3102, "best_ppl": 56.17,  "tokens_k": 2050, "nan": 0},
      ... (every 500 steps from stdout >>> Step N summary <<< lines)
    ],
    "first_below_baseline_step": 1500,
    "best_ppl_first_seen_step": 4000,
    "convergence_pct_of_random": 58.8
  },
  "moe_balance": {
    "applicable": false,
    "regime_class": "n/a | uniform | collapsed | oscillating",
    "per_layer_load_ema_summary": [],
    "gate_hit_total": 0,
    "kill_hit_total": 0,
    "imbalance_step_ranges": []
  },
  "entropy_penalty_trace": {
    "applicable": false,
    "tau_anneal_observed_step_range": [null, null],
    "beta_H_eff_observed_range": [null, null],
    "H_observed_range": [null, null]
  },
  "nan_incidents": {
    "raw_count_run_index": 9,
    "step_summary_cumulative": 0,
    "classification": "false_positive",
    "classification_reason": "raw_count > 0 AND fatal=0 AND rc=0 (D-291 false-positive class)",
    "step_ranges_observed": []
  },
  "stream_diagnostics": {
    "WT103_bytes": 2359171,
    "OWT_bytes": 20184043,
    "Python_bytes": 11402085,
    "GSM8K_bytes": 3014537,
    "MATH_bytes": 3473062,
    "total_bytes": 40432898
  },
  "anomalies": [
    {"line_no": 1893, "level": "FATAL", "text": "..."},
    ...
  ],
  "log_path": "data/runs/phase3_A42/stdout.log",
  "log_size_bytes": 207360,
  "stdout_mtime": "2026-04-28T10:11:59Z",
  "digest_generated_at": "2026-05-05T17:30:00Z"
}
```

### 3. Per-cell digest schema (markdown narrative)

The markdown is a one-page human-and-Director-readable summary derived from the JSON. Director / mechanism_extractor reads this first. Schema:

```markdown
# Cell A42 — Phase 3 (dense50m, lr=1e-3, seed=42)

**Outcome:** ✅ CONVERGING (58.8% of random); rc=0 atexit_clean
**Best PPL:** 2.12 (first seen step 4000)
**Wall:** 12.59h (vs ETA 11.8h, +6.7%)

## Loss trajectory
- **Shape class:** monotone_decreasing
- **First below random baseline:** step 1500
- **Final avg_lm:** 6.116 (recent), best 0.7528

## MoE balance
n/a (dense model)

## NaN incidents
- raw_count=9 / step_summary_cumulative=0
- **Classification:** false_positive (D-291 class — fatal=0, rc=0, training trajectory unaffected)

## Stream diagnostics
WT103=2.4 MB · OWT=20.2 MB · Python=11.4 MB · GSM8K=3.0 MB · MATH=3.5 MB · TOT=40.4 MB

## Anomalies
None.

---
*Generated by tools/training_digest.py from data/runs/phase3_A42/stdout.log (203 KB) on 2026-05-05.*
```

### 4. `tools/training_digest_aggregate.py` — phase-level rollup

**Inputs:**
- All `data/digests/training/<cell_id>.json` for cells in the phase
- `data/checkpoints/<phase>/run_index.json` (cell list)
- `programs/<program>/<phase_prereg>.md` (factorial structure if applicable)

**Outputs:**
- `data/digests/training/<phase>_summary.md` (~5 KB human/Director-readable)
- `data/digests/training/<phase>_summary.json` (~10 KB structured)

**CLI:**
```bash
python3 tools/training_digest_aggregate.py <phase> [--regen]
```

Phase aggregate contents (markdown):
- Phase header (cell count, total wall, factorial structure if known)
- Cell-by-cell table (one row per cell, key metrics: best_ppl, avg_lm, wall, rc, classification)
- Outliers vs cell-mates (e.g., C42's best_ppl 3.18 vs C43=2.61, C44=2.43)
- Cross-cell anomaly catalogue
- Phase-level NaN classification summary
- Forward-carry candidates (step ranges or patterns that warrant P10 mechanism investigation)

**Explicitly NOT in aggregate:**
- ANOVA tables, F statistics, p-values (P9 statistical_reviewer's job)
- Hypothesis tests, decision-rule firing (P9 owns)
- Mechanism explanations (mechanism_extractor's job at P10)
- Benchmark scores (P11 measurement_theorist's job)

### 5. Wire-in points

**Orchestrator (post-cell-complete hook):**
```bash
# In tools/run_phase3_factorial.py or scale_experiment atexit handler equivalent
# After run_index.json updated for the just-completed cell:
python3 tools/training_digest.py "$CELL_ID" --phase "$PHASE" >> "$LOG" 2>&1 || \
    echo "training_digest failed for $CELL_ID (non-fatal); see $LOG" >&2
```

After the last cell of a phase completes:
```bash
python3 tools/training_digest_aggregate.py "$PHASE" >> "$LOG" 2>&1 || true
```

**mechanism_extractor procedural:**
Add a "Read digests first" step at the top of the input-reading checklist:

> 1. Read `data/digests/training/<phase>_summary.md` (phase aggregate, ~5 KB).
> 2. For each cell flagged as outlier or anomalous in the aggregate, read `data/digests/training/<cell_id>.md` (~1 KB each).
> 3. Raw `data/runs/<cell>/stdout.log` is fallback evidence — read only when:
>    - The aggregate or cell digest flags an anomaly without a stdout line ref
>    - You need exact verbatim text for a quote
>    - You suspect digest staleness (mtime check)

**statistical_reviewer procedural** (optional v1, recommended):
Add same "read digests first" pointer; statistical_reviewer's ANOVA work consumes per-cell metrics from JSON sidecars rather than re-parsing stdout.

### 6. Tests

- `test_training_digest_parser.py`: 12 tests
  - Header column parsing (matches C source line 363 schema)
  - Step row parsing (8 columns)
  - `[load] step=N` line parsing (per-layer + SUMMARY)
  - `[entpen] step=N` line parsing
  - `>>> Step N summary` block parsing
  - Final summary block parsing
  - Verdict line parsing
  - ATEXIT marker detection
  - Anomaly detection (FATAL / ERROR / WARNING with line refs)
  - Loss trajectory shape classification (4 classes — fixture for each)
  - MoE regime classification (3 classes — fixture for each)
  - NaN classification (false_positive vs hard_nan)
- `test_training_digest_e2e.py`: 4 tests
  - Real phase3_A42 stdout.log → expected digest match
  - Missing stdout.log → graceful error
  - Mtime staleness check (regen flag works)
  - Digest output schema matches JSON schema spec
- `test_training_digest_aggregate.py`: 6 tests
  - All 12 phase-3 cells aggregated → expected summary
  - Missing per-cell digest → aggregator regenerates it
  - Outlier detection (C42 best_ppl outlier flagged)
  - Phase prereg discovery (auto-finds factorial structure)
  - Aggregator does NOT replicate ANOVA fields
  - Phase summary mtime tracks per-cell digest mtimes

## Out of scope (v1)

- **Real-time mid-cell digest update.** Fires only on cell complete (atexit). Live monitoring stays at the run_index + ps + tail level.
- **LLM-based unusual-pattern detection.** Heuristic parsing only. If a stdout has unusual content the parser doesn't recognize, it gets recorded as an anomaly with line ref — operator/mechanism_extractor reads raw stdout for that specific case.
- **Cross-phase aggregation.** v1 produces per-phase aggregates only. Phase 2 stdouts don't even exist at the standard path; cross-phase trends would need a separate spec.
- **Charts / visualizations.** Text-only. `tools/visualize.py` covers chart needs.
- **C-source-side parser config.** Schema is hard-coded to current `scale_experiment.c:363` column layout. If C source changes column order, parser tests fail and someone updates both.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| C source emits new line types parser doesn't recognize | Parser captures unrecognized FATAL/ERROR/WARNING lines as anomalies with line refs. Other unrecognized lines are silently skipped (intentional — most non-step lines are environment / library noise). |
| Schema drift across phases | Per-cell digest includes `schema_version`. Aggregator refuses to mix versions; on mismatch, refuses to aggregate and prints "regenerate cell digests with current parser version". |
| Stdout.log truncated mid-write (e.g., Director restart while training) | Parser detects missing ATEXIT marker → marks `outcome.atexit_clean=false` and `parser_warning=truncated_log`. Aggregator surfaces as anomaly. |
| `data/digests/training/` namespace collision with future tools | Path explicit in spec; documented in `data/digests/README.md` (created in v1) listing each subdir's owner. |
| Mechanism_extractor reads stale digest after stdout.log re-write | Parser checks `stdout_mtime > digest_generated_at` and refuses to read; emits "regen needed". CLI `--regen` forces. |
| NaN classification gets D-291 wrong | Parser uses run_index.json's `fatal/rc` AND step-summary cumulative AND raw count to classify. Three-way agreement required for `false_positive`; otherwise `unclassified`. |
| Aggregator duplicates P9 statistical work | Spec § "Explicitly NOT in aggregate" enumerates excluded fields. Test asserts no ANOVA / F / p-value keys in JSON. |

## Success criteria

- mechanism_extractor session reading 1 cell's digest takes <2 KB context (vs 200 KB raw stdout). 12-cell phase context: <50 KB total (vs 2.4 MB).
- All 12 phase-3 cells produce identical digests across two parser runs (deterministic).
- Backfill of phase-3 cells at v1 deployment matches the manual mechanism_extractor read at D-313 — i.e., the digest captures every claim the human-driven mechanism report made about training-time signals.
- Anomalies surface earlier: future P8 EXECUTE Director session can answer "is anything weird?" by reading aggregate digest, not by tailing stdout.

## Migration

Stage-gated, each commits independently:

- **Stage D1**: `tools/training_digest.py` parser + per-cell schema + tests (~3h). Backfills phase-3 cells as test corpus.
- **Stage D2**: `tools/training_digest_aggregate.py` + phase format + tests (~2h). Produces phase3_factorial_summary.{md,json}.
- **Stage D3**: Orchestrator wire-in (run digester on cell complete) + mechanism_extractor procedural update (~1h).
- **Stage D4**: Validation — backfill all 12 phase-3 cells; cross-check digest content against `phase10_mechanism_report.md` claims; verify every quantitative claim in the report is reproducible from the digest (~1h).

Total effort: ~7-8h via subagent-driven-development. Schema-stable after D1.

## Out-of-scope follow-ups (future specs)

- Multi-phase digest comparison (phase2 vs phase3 baselines, etc.)
- LLM-based anomaly classification (haiku call for unusual-pattern detection)
- Live mid-cell digest (every 500 steps emit incremental digest)
- C-source-side digest emission (training code emits structured diagnostics directly, eliminating parser layer)
