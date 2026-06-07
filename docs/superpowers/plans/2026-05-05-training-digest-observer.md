# Training Digest Observer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a heuristic parser that converts training stdout.log + run_index.json into compact LLM-targeted digests so mechanism_extractor reads ~10 KB instead of 2.4 MB.

**Architecture:** `tools/training_digest.py` (per-cell parser, emits markdown + JSON sidecar to `data/digests/training/`) + `tools/training_digest_aggregate.py` (phase rollup) + orchestrator wire-in + mechanism_extractor procedural update.

**Tech Stack:** Python 3 (stdlib only — re, json, dataclasses, pathlib) + pytest. No external deps.

**Spec:** `docs/superpowers/specs/2026-05-05-training-digest-observer.md`

---

## Stage D1: Per-Cell Parser

**Goal:** `tools/training_digest.py` reads one cell's stdout.log + run_index entry, emits markdown + JSON sidecar.

### Task 1: Test fixture — A42 stdout sample (truncated to first/last 50 lines)

**Files:**
- Create: `tests/fixtures/training_digest/phase3_A42_sample.log`
- Create: `tests/fixtures/training_digest/phase3_A42_run_index_entry.json`

- [ ] **Step 1: Capture a representative truncated sample**

```bash
cd <repo>
mkdir -p tests/fixtures/training_digest
# First 50 lines (config + initial steps) + last 50 lines (final summary + atexit)
{
  head -50 data/runs/phase3_A42/stdout.log
  echo "..."
  echo "[middle truncated for fixture brevity — real test reads full log]"
  echo "..."
  tail -50 data/runs/phase3_A42/stdout.log
} > tests/fixtures/training_digest/phase3_A42_sample.log

# Extract A42 entry from run_index
python3 -c "
import json
d = json.load(open('data/checkpoints/phase3_factorial/run_index.json'))
json.dump(d['A42'], open('tests/fixtures/training_digest/phase3_A42_run_index_entry.json', 'w'), indent=2)
"
```

- [ ] **Step 2: Commit fixtures**

```bash
git add tests/fixtures/training_digest/
git commit -m "test(fixture): phase3_A42 stdout + run_index sample for training_digest tests"
```

---

### Task 2: training_digest.py — module skeleton + main()

**Files:**
- Create: `tools/training_digest.py`
- Create: `tests/test_training_digest_skeleton.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_training_digest_skeleton.py
"""Validates tools/training_digest.py module skeleton."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_module_imports():
    import tools.training_digest as td
    assert hasattr(td, "main")
    assert hasattr(td, "REPO")


def test_main_no_args_prints_usage():
    import tools.training_digest as td
    rc = td.main(argv=[])
    assert rc != 0
```

- [ ] **Step 2: Run to verify fail**

```bash
cd <repo> && source .venv/bin/activate
python -m pytest tests/test_training_digest_skeleton.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement skeleton**

```python
# tools/training_digest.py
"""Training run digest parser — converts stdout.log + run_index.json into
compact LLM-targeted digests at data/digests/training/<cell_id>.{md,json}.

Per spec: docs/superpowers/specs/2026-05-05-training-digest-observer.md.

CLI:
    python3 tools/training_digest.py <cell_id> [--phase <phase>] [--regen]
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SCHEMA_VERSION = "1.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate training digest for a cell.")
    parser.add_argument("cell_id", nargs="?", help="Cell ID (e.g., A42)")
    parser.add_argument("--phase", help="Phase name (e.g., phase3_factorial). Auto-detected if omitted.")
    parser.add_argument("--regen", action="store_true", help="Force regen even if digest is fresh.")
    args = parser.parse_args(argv)

    if not args.cell_id:
        parser.print_usage(sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify pass**

- [ ] **Step 5: Commit**

```bash
git add tools/training_digest.py tests/test_training_digest_skeleton.py
git commit -m "scaffold: tools/training_digest.py module + skeleton tests"
```

---

### Task 3: training_digest.py — parse_step_rows()

**Files:**
- Modify: `tools/training_digest.py`
- Create: `tests/test_training_digest_parser.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_training_digest_parser.py
"""Validates tools/training_digest.py heuristic parsers against C source schema.

Step row format from src/training/scale_experiment.c:363-371:
    step | loss (pred + bal + lm) | ppl | gnorm | ent | vn_d | ep | ms
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_parse_step_row_canonical_8_columns():
    from tools.training_digest import parse_step_row
    line = "  820  |   7.3117 ( 0.000 + 0.0000 +  7.312) |   1497.68 |  7.762 | 0.00000 | 0.0000 |  0 | 8683"
    r = parse_step_row(line)
    assert r is not None
    assert r["step"] == 820
    assert abs(r["loss_total"] - 7.3117) < 1e-3
    assert abs(r["loss_pred"] - 0.000) < 1e-3
    assert abs(r["loss_bal"] - 0.0000) < 1e-3
    assert abs(r["loss_lm"] - 7.312) < 1e-3
    assert abs(r["ppl"] - 1497.68) < 1e-2
    assert abs(r["gnorm"] - 7.762) < 1e-3
    assert abs(r["ent"] - 0.00000) < 1e-3
    assert abs(r["vn_d"] - 0.0000) < 1e-3
    assert r["ep"] == 0
    assert r["ms"] == 8683.0


def test_parse_step_row_returns_none_for_non_match():
    from tools.training_digest import parse_step_row
    assert parse_step_row("[load] step=500 layer=0 K=1 ema=[1.0]") is None
    assert parse_step_row("--- Training Complete ---") is None
    assert parse_step_row("") is None
```

- [ ] **Step 2: Run failing tests**

- [ ] **Step 3: Implement `parse_step_row`**

```python
import re

# Step row schema per src/training/scale_experiment.c:363-371:
#   step | loss_total (loss_pred + loss_bal + loss_lm) | ppl | gnorm | ent | vn_d | ep | ms
_STEP_ROW_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*"            # step
    r"([\d.\-]+)\s*"                # loss_total
    r"\(\s*([\d.\-]+)\s*\+\s*"      # loss_pred
    r"([\d.\-]+)\s*\+\s*"           # loss_bal
    r"([\d.\-]+)\s*\)\s*\|\s*"      # loss_lm
    r"([\d.\-]+)\s*\|\s*"           # ppl
    r"([\d.\-]+)\s*\|\s*"           # gnorm
    r"([\d.\-]+)\s*\|\s*"           # ent
    r"([\d.\-]+)\s*\|\s*"           # vn_d
    r"(\d+)\s*\|\s*"                # ep
    r"([\d.\-]+)\s*$"               # ms
)


def parse_step_row(line: str) -> dict | None:
    m = _STEP_ROW_RE.match(line)
    if not m:
        return None
    return {
        "step": int(m.group(1)),
        "loss_total": float(m.group(2)),
        "loss_pred": float(m.group(3)),
        "loss_bal": float(m.group(4)),
        "loss_lm": float(m.group(5)),
        "ppl": float(m.group(6)),
        "gnorm": float(m.group(7)),
        "ent": float(m.group(8)),
        "vn_d": float(m.group(9)),
        "ep": int(m.group(10)),
        "ms": float(m.group(11)),
    }
```

- [ ] **Step 4: Run tests pass**

- [ ] **Step 5: Commit**

```bash
git add tools/training_digest.py tests/test_training_digest_parser.py
git commit -m "feat: training_digest.parse_step_row() — 8-column row parser per scale_experiment.c:363"
```

---

### Task 4: training_digest.py — parse_step_summary() (the >>> Step N summary <<< blocks)

**Files:**
- Modify: `tools/training_digest.py`
- Modify: `tests/test_training_digest_parser.py`

- [ ] **Step 1: Write failing tests**

```python
def test_parse_step_summary_canonical():
    from tools.training_digest import parse_step_summary
    line = "  >>> Step 4500 summary: avg_lm=5.4874, best_ppl=2.12, epochs=0, tokens=9218K, NaN=0 <<<"
    r = parse_step_summary(line)
    assert r is not None
    assert r["step"] == 4500
    assert abs(r["avg_lm"] - 5.4874) < 1e-3
    assert abs(r["best_ppl"] - 2.12) < 1e-2
    assert r["epochs"] == 0
    assert r["tokens_k"] == 9218
    assert r["nan"] == 0


def test_parse_step_summary_returns_none_for_non_match():
    from tools.training_digest import parse_step_summary
    assert parse_step_summary("step | loss") is None
    assert parse_step_summary("") is None
```

- [ ] **Step 2-4: Implement, run, verify pass**

```python
_STEP_SUMMARY_RE = re.compile(
    r">>>\s*Step\s+(\d+)\s+summary:\s+"
    r"avg_lm=([\d.\-]+),\s*"
    r"best_ppl=([\d.\-]+),\s*"
    r"epochs=(\d+),\s*"
    r"tokens=(\d+)K,\s*"
    r"NaN=(\d+)\s*<<<"
)


def parse_step_summary(line: str) -> dict | None:
    m = _STEP_SUMMARY_RE.search(line)
    if not m:
        return None
    return {
        "step": int(m.group(1)),
        "avg_lm": float(m.group(2)),
        "best_ppl": float(m.group(3)),
        "epochs": int(m.group(4)),
        "tokens_k": int(m.group(5)),
        "nan": int(m.group(6)),
    }
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: training_digest.parse_step_summary() — >>> Step N summary <<< block parser"
```

---

### Task 5: training_digest.py — parse_load_lines() (MoE per-layer EMA + SUMMARY)

**Files:**
- Modify: `tools/training_digest.py`
- Modify: `tests/test_training_digest_parser.py`

- [ ] **Step 1: Write failing tests**

```python
def test_parse_load_per_layer():
    from tools.training_digest import parse_load_per_layer
    line = "[load] step=820 layer=2 K=1 ema=[1.00000] min=1.00000 max=1.00000 ratio=1.00"
    r = parse_load_per_layer(line)
    assert r is not None
    assert r["step"] == 820
    assert r["layer"] == 2
    assert r["K"] == 1
    assert r["ema"] == [1.0]
    assert abs(r["min_load"] - 1.0) < 1e-5
    assert abs(r["max_load"] - 1.0) < 1e-5
    assert abs(r["ratio"] - 1.0) < 1e-3


def test_parse_load_per_layer_multi_expert():
    from tools.training_digest import parse_load_per_layer
    line = "[load] step=820 layer=2 K=8 ema=[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125] min=0.12500 max=0.12500 ratio=1.00"
    r = parse_load_per_layer(line)
    assert r is not None
    assert r["K"] == 8
    assert len(r["ema"]) == 8
    assert all(abs(v - 0.125) < 1e-4 for v in r["ema"])


def test_parse_load_summary():
    from tools.training_digest import parse_load_summary
    line = "[load] step=820 SUMMARY worst_ratio=1.00 worst_layer=0 gate_hit=0 kill_hit=0"
    r = parse_load_summary(line)
    assert r["step"] == 820
    assert abs(r["worst_ratio"] - 1.0) < 1e-3
    assert r["worst_layer"] == 0
    assert r["gate_hit"] == 0
    assert r["kill_hit"] == 0
```

- [ ] **Step 2-4: Implement, run, verify**

```python
_LOAD_PER_LAYER_RE = re.compile(
    r"\[load\]\s+step=(\d+)\s+layer=(\d+)\s+K=(\d+)\s+"
    r"ema=\[([\d.,\s\-]+)\]\s+"
    r"min=([\d.\-]+)\s+max=([\d.\-]+)\s+ratio=([\d.\-]+)"
)
_LOAD_SUMMARY_RE = re.compile(
    r"\[load\]\s+step=(\d+)\s+SUMMARY\s+"
    r"worst_ratio=([\d.\-]+)\s+worst_layer=(\d+)\s+"
    r"gate_hit=(\d+)\s+kill_hit=(\d+)"
)


def parse_load_per_layer(line: str) -> dict | None:
    m = _LOAD_PER_LAYER_RE.search(line)
    if not m:
        return None
    ema_str = m.group(4)
    ema = [float(x.strip()) for x in ema_str.split(",") if x.strip()]
    return {
        "step": int(m.group(1)),
        "layer": int(m.group(2)),
        "K": int(m.group(3)),
        "ema": ema,
        "min_load": float(m.group(5)),
        "max_load": float(m.group(6)),
        "ratio": float(m.group(7)),
    }


def parse_load_summary(line: str) -> dict | None:
    m = _LOAD_SUMMARY_RE.search(line)
    if not m:
        return None
    return {
        "step": int(m.group(1)),
        "worst_ratio": float(m.group(2)),
        "worst_layer": int(m.group(3)),
        "gate_hit": int(m.group(4)),
        "kill_hit": int(m.group(5)),
    }
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: training_digest — parse_load_per_layer + parse_load_summary (MoE balance trace parsers)"
```

---

### Task 6: training_digest.py — parse_final_summary() (the post-Training-Complete block)

**Files:**
- Modify: `tools/training_digest.py`
- Modify: `tests/test_training_digest_parser.py`

- [ ] **Step 1: Write failing test**

```python
def test_parse_final_summary_extracts_all_metrics():
    from tools.training_digest import parse_final_summary
    sample = """
--- Training Complete ---

Summary:
  Model:          dense50m (~34.62M params)
  Steps trained:  5000 / 5000
  Epochs:         0
  Tokens trained: 10240K
  Initial PPL:    57601.60
  Best PPL:       2.12
  Best loss:      0.7528
  Recent avg LM:  6.1159
  Random baseline:10.3972 (PPL=32768)
  NaN events:     0 (total)
  Stream diagnostics:
    [WT103] bytes=2359171  epochs=0
    [OWT] bytes=20184043  epochs=0
    [Python] bytes=11402085  epochs=0
    [GSM8K] bytes=3014537  epochs=0
    [MATH] bytes=3473062  epochs=0
    [TOT] bytes=40432898 (38.56 MB)

Verdict: CONVERGING -- LM loss well below random baseline (58.8% of random)

=== Scale-Up Experiment Complete ===

*** ATEXIT: clean process exit ***
"""
    r = parse_final_summary(sample)
    assert r["model"] == "dense50m"
    assert r["params_M"] == 34.62
    assert r["steps_trained"] == 5000
    assert r["steps_planned"] == 5000
    assert r["epochs"] == 0
    assert r["tokens_k"] == 10240
    assert abs(r["initial_ppl"] - 57601.60) < 0.1
    assert abs(r["best_ppl"] - 2.12) < 1e-2
    assert abs(r["best_loss"] - 0.7528) < 1e-3
    assert abs(r["recent_avg_lm"] - 6.1159) < 1e-3
    assert r["nan_events_total"] == 0
    assert r["streams"]["WT103_bytes"] == 2359171
    assert r["streams"]["OWT_bytes"] == 20184043
    assert r["streams"]["total_bytes"] == 40432898
    assert "CONVERGING" in r["verdict_line"]
    assert r["atexit_clean"] is True
```

- [ ] **Step 2-4: Implement, run, verify**

(Function reads with regex per pattern: model, params, steps, ppl, loss, streams, verdict, atexit. Returns dict per spec JSON sidecar.)

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: training_digest.parse_final_summary() — post-Training-Complete summary block parser"
```

---

### Task 7: training_digest.py — classify_loss_trajectory()

**Files:**
- Modify: `tools/training_digest.py`
- Modify: `tests/test_training_digest_parser.py`

- [ ] **Step 1: Write failing tests** for the 4 classes (monotone_decreasing, plateau_after_N, oscillating, collapsed) using synthetic step_summary lists with explicit thresholds documented in the function docstring.

- [ ] **Step 2-4: Implement** with thresholds:
  - `monotone_decreasing`: best_ppl strictly decreasing across step summaries (allowing 1 noisy step)
  - `plateau_after_N`: |Δ avg_lm| < 0.05 for ≥3 consecutive step summaries; report N as the step where plateau begins
  - `oscillating`: avg_lm direction changes ≥4 times across step summaries
  - `collapsed`: best_ppl > 50% of initial AT the last step (training failed to converge)

- [ ] **Step 5: Commit**

---

### Task 8: training_digest.py — classify_moe_regime() + classify_nan()

- [ ] **Step 1-5:** Same TDD pattern. classify_moe_regime takes per-layer load summaries, classifies as `n/a | uniform | collapsed | oscillating` based on per-layer load EMA std-dev and worst-ratio behavior. classify_nan takes raw run_index nan count, step-summary cumulative, fatal/rc — returns `false_positive | hard_nan | unclassified`.

---

### Task 9: training_digest.py — generate_digest() end-to-end + JSON output

- [ ] **Step 1-5:** generate_digest(cell_id, phase, repo_root) reads stdout.log + run_index entry, runs all parsers + classifiers, writes `data/digests/training/<cell_id>.json` per schema spec.

---

### Task 10: training_digest.py — markdown narrative renderer

- [ ] **Step 1-5:** render_markdown(digest_dict) → markdown string per spec § 3 schema. Test against fixture.

---

### Task 11: training_digest.py — main() wiring + mtime staleness check

- [ ] **Step 1-5:** main(cell_id) does: discover phase if not given, check digest mtime vs stdout mtime, regenerate if stale or `--regen`, write both `.md` and `.json`, return 0.

---

## Stage D1 Gate (user review): pause to verify per-cell parser produces correct A42 digest matching mechanism_extractor's hand-extracted claims, before building aggregator.

---

## Stage D2: Phase-Level Aggregator

**Goal:** `tools/training_digest_aggregate.py` reads all per-cell digests for a phase, emits cross-cell summary.

### Task 12: training_digest_aggregate.py — module skeleton + cell discovery

- [ ] **Step 1-5:** Skeleton + discover_cells(phase, repo_root) reads `data/checkpoints/<phase>/run_index.json` → list of cell IDs. Test with phase3_factorial fixture.

### Task 13: training_digest_aggregate.py — per-cell metrics table generation

- [ ] **Step 1-5:** For each discovered cell, load JSON sidecar, build a row: cell_id | best_ppl | recent_avg_lm | wall_h | rc | nan_class. Render as markdown table.

### Task 14: training_digest_aggregate.py — outlier detection within factorial cell-mates

- [ ] **Step 1-5:** Group cells by config (model, lr) ignoring seed; flag cells whose best_ppl deviates >2 std from cell-mate median. (E.g., C42's best_ppl 3.18 vs C43=2.61, C44=2.43 → C42 flagged.)

### Task 15: training_digest_aggregate.py — anomaly catalogue + assertion of NO ANOVA fields

- [ ] **Step 1-5:** Cross-cell anomaly catalogue (every cell's anomalies, deduped by class). Test that aggregator output JSON has NO `anova_F`, `p_value`, `bonferroni_alpha`, etc. fields (P9 statistical_reviewer's domain).

### Task 16: training_digest_aggregate.py — main() + markdown narrative

- [ ] **Step 1-5:** main(phase) generates `data/digests/training/<phase>_summary.{md,json}`. Markdown contains: phase header, factorial structure if known, cell-by-cell table, outliers, anomaly catalogue, forward-carry candidates.

---

## Stage D2 Gate (user review): pause to verify phase aggregate captures every quantitative claim mechanism_extractor made manually at D-313.

---

## Stage D3: Wire-In

### Task 17: Orchestrator hook — run digester at cell complete

- [ ] **Step 1-5:** Identify orchestrator's per-cell-complete branch (likely `tools/run_phase3_factorial.py` or `scale_experiment` atexit shell wrapper). Add invocation:
```bash
python3 tools/training_digest.py "$CELL_ID" --phase "$PHASE" >> "$LOG" 2>&1 || true
```
Plus phase-level aggregator at last-cell-complete.

### Task 18: mechanism_extractor procedural — read digests first

- [ ] **Step 1-5:** Edit `data/agents/mechanism_extractor/procedural.md`. Add "Read digests first" step. Test asserts the procedural references the digest paths.

---

## Stage D4: Validation

### Task 19: Backfill all 12 phase-3 cells + verify content vs phase10_mechanism_report.md

- [ ] **Step 1: Run digester on all 12 cells**
```bash
for cell in A42 A43 A44 B42 B43 B44 C42 C43 C44 D42 D43 D44; do
  python3 tools/training_digest.py "$cell" --phase phase3_factorial --regen
done
python3 tools/training_digest_aggregate.py phase3_factorial --regen
```

- [ ] **Step 2: Cross-check claims**

For each quantitative claim in `programs/program_2_example/phase10_mechanism_report.md` § 1-§ 8 about training-time signals, verify the digest captures it. Document any gaps as future-spec items.

- [ ] **Step 3: Document any gaps**

If digests miss any claim, decide: (a) parser gap → fix in v1.1, (b) intentional out-of-scope → document. Either way, write a brief "validation notes" doc at `docs/superpowers/notes/2026-05-05-training-digest-d4-validation.md`.

- [ ] **Step 4: Commit**

```bash
git add data/digests/training/ docs/superpowers/notes/
git commit -m "validate: backfill phase-3 cells; cross-check digest vs phase10_mechanism_report.md"
```

---

## Stage D4 Gate (user review): final validation. Decide whether digest content is ready for mechanism_extractor's next P10 dispatch.

---

## Self-Review Notes

Before marking the plan complete, verify:

1. **Spec coverage:** every section of the spec maps to at least one task.
   - Per-cell parser → Tasks 2-11
   - Aggregator → Tasks 12-16
   - Wire-in → Tasks 17-18
   - Validation → Task 19

2. **No placeholders:** all task steps have concrete code or commands.

3. **TDD discipline:** every implementation task has Test → Run-fail → Implement → Run-pass → Commit cadence.

4. **Schema lockdown:** parser tests reference `src/training/scale_experiment.c:363` line for column schema. If C source changes, parser tests will fail loudly.

---

## Estimated effort

- Stage D1: ~3h (10 tasks: scaffold + 8 parser components + main wiring)
- Stage D2: ~2h (5 tasks: aggregator)
- Stage D3: ~1h (2 tasks: wire-in)
- Stage D4: ~1h (validation + write-up)

**Total: ~7 hours** via subagent-driven-development, plus user-review gates between D1/D2, D2/D3, D3/D4.
