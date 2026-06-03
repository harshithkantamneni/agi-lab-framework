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
    assert r is not None, "expected dict, got None"
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


def test_parse_step_row_negative_loss_components():
    """The C source uses %6.3f for loss_pred which permits negative values
    (e.g., when prediction is better than zero baseline)."""
    from tools.training_digest import parse_step_row
    line = "  100  |   5.1234 (-0.123 + 0.0500 +  5.196) |    150.00 |  2.500 | 0.10000 | 0.0500 |  0 | 1234"
    r = parse_step_row(line)
    assert r is not None
    assert r["step"] == 100
    assert abs(r["loss_pred"] - (-0.123)) < 1e-3


def test_parse_step_row_returns_none_for_non_match():
    from tools.training_digest import parse_step_row
    assert parse_step_row("[load] step=500 layer=0 K=1 ema=[1.0]") is None
    assert parse_step_row("--- Training Complete ---") is None
    assert parse_step_row("") is None
    assert parse_step_row("step   | loss      (pred   + bal    + lm    ) | ppl       | gnorm  | ent     | vn_d   | ep | ms") is None  # header line
    assert parse_step_row("-------+---------------------------------------+----------+--------+---------+--------+----+------") is None  # separator


def test_parse_step_row_fixture_yields_at_least_one_row():
    """Ensure the regex matches at least one real step row from the live fixture."""
    from tools.training_digest import parse_step_row
    fixture = REPO / "tests/fixtures/training_digest/phase3_A42_sample.log"
    rows = []
    for line in fixture.read_text().splitlines():
        r = parse_step_row(line)
        if r is not None:
            rows.append(r)
    assert len(rows) >= 1, "expected at least one step row in the fixture"
    assert all(r["step"] >= 0 for r in rows)
    assert all(r["ms"] > 0 for r in rows)


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


def test_parse_step_summary_with_nonzero_nan_and_epoch():
    from tools.training_digest import parse_step_summary
    line = "  >>> Step 5000 summary: avg_lm=4.0123, best_ppl=12.34, epochs=2, tokens=10000K, NaN=3 <<<"
    r = parse_step_summary(line)
    assert r is not None
    assert r["step"] == 5000
    assert r["epochs"] == 2
    assert r["nan"] == 3


def test_parse_step_summary_returns_none_for_non_match():
    from tools.training_digest import parse_step_summary
    assert parse_step_summary("step | loss") is None
    assert parse_step_summary("") is None
    assert parse_step_summary("[load] step=500 SUMMARY worst_ratio=1.00 worst_layer=0 gate_hit=0 kill_hit=0") is None
    # The step row from Task 3 should NOT match this parser
    assert parse_step_summary("  820  |   7.3117 ( 0.000 + 0.0000 +  7.312) |   1497.68 |  7.762 | 0.00000 | 0.0000 |  0 | 8683") is None


def test_parse_step_summary_fixture_yields_summaries():
    """Sanity check against the live fixture."""
    from tools.training_digest import parse_step_summary
    fixture = REPO / "tests/fixtures/training_digest/phase3_A42_sample.log"
    summaries = []
    for line in fixture.read_text().splitlines():
        r = parse_step_summary(line)
        if r is not None:
            summaries.append(r)
    assert len(summaries) >= 1, "expected at least one >>> Step N summary <<< in fixture"
    # Steps should be multiples of 500 (training emits every 500 steps)
    assert all(s["step"] % 500 == 0 for s in summaries), \
        f"expected all step values to be multiples of 500, got {[s['step'] for s in summaries]}"


def test_parse_load_per_layer_canonical_K1():
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


def test_parse_load_per_layer_K8_uniform():
    from tools.training_digest import parse_load_per_layer
    line = "[load] step=820 layer=2 K=8 ema=[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125] min=0.12500 max=0.12500 ratio=1.00"
    r = parse_load_per_layer(line)
    assert r is not None
    assert r["K"] == 8
    assert len(r["ema"]) == 8
    assert all(abs(v - 0.125) < 1e-4 for v in r["ema"])


def test_parse_load_per_layer_K8_imbalanced():
    """Realistic imbalanced MoE state (one expert hot, others cold)."""
    from tools.training_digest import parse_load_per_layer
    line = "[load] step=4500 layer=4 K=8 ema=[0.50000,0.10000,0.10000,0.10000,0.05000,0.05000,0.05000,0.05000] min=0.05000 max=0.50000 ratio=10.00"
    r = parse_load_per_layer(line)
    assert r is not None
    assert r["K"] == 8
    assert abs(r["ema"][0] - 0.5) < 1e-4
    assert abs(r["min_load"] - 0.05) < 1e-4
    assert abs(r["max_load"] - 0.5) < 1e-4
    assert abs(r["ratio"] - 10.0) < 1e-2


def test_parse_load_per_layer_returns_none_for_non_match():
    from tools.training_digest import parse_load_per_layer
    assert parse_load_per_layer("[load] step=820 SUMMARY worst_ratio=1.00 worst_layer=0 gate_hit=0 kill_hit=0") is None
    assert parse_load_per_layer("") is None
    assert parse_load_per_layer("  >>> Step 4500 summary: avg_lm=5.4874, best_ppl=2.12, epochs=0, tokens=9218K, NaN=0 <<<") is None


def test_parse_load_summary_canonical():
    from tools.training_digest import parse_load_summary
    line = "[load] step=820 SUMMARY worst_ratio=1.00 worst_layer=0 gate_hit=0 kill_hit=0"
    r = parse_load_summary(line)
    assert r is not None
    assert r["step"] == 820
    assert abs(r["worst_ratio"] - 1.0) < 1e-3
    assert r["worst_layer"] == 0
    assert r["gate_hit"] == 0
    assert r["kill_hit"] == 0


def test_parse_load_summary_with_imbalance():
    from tools.training_digest import parse_load_summary
    line = "[load] step=4500 SUMMARY worst_ratio=10.00 worst_layer=4 gate_hit=12 kill_hit=3"
    r = parse_load_summary(line)
    assert r is not None
    assert abs(r["worst_ratio"] - 10.0) < 1e-3
    assert r["worst_layer"] == 4
    assert r["gate_hit"] == 12
    assert r["kill_hit"] == 3


def test_parse_load_summary_returns_none_for_non_match():
    from tools.training_digest import parse_load_summary
    # A per-layer line is NOT a SUMMARY line
    assert parse_load_summary("[load] step=820 layer=2 K=1 ema=[1.00000] min=1.00000 max=1.00000 ratio=1.00") is None
    assert parse_load_summary("") is None


def test_parse_load_fixture_yields_per_layer_and_summary_pairs():
    """In the fixture (dense model = K=1 across all layers), each step
    that emits per-layer lines should also emit a SUMMARY line."""
    from tools.training_digest import parse_load_per_layer, parse_load_summary
    fixture = REPO / "tests/fixtures/training_digest/phase3_A42_sample.log"
    per_layer_count = 0
    summary_count = 0
    for line in fixture.read_text().splitlines():
        if parse_load_per_layer(line) is not None:
            per_layer_count += 1
        if parse_load_summary(line) is not None:
            summary_count += 1
    assert per_layer_count >= 1, "expected at least one [load] per-layer line"
    assert summary_count >= 1, "expected at least one [load] SUMMARY line"


def test_parse_final_summary_extracts_all_fields():
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
    assert r is not None
    assert r["model"] == "dense50m"
    assert abs(r["params_M"] - 34.62) < 1e-2
    assert r["steps_trained"] == 5000
    assert r["steps_planned"] == 5000
    assert r["epochs"] == 0
    assert r["tokens_k"] == 10240
    assert abs(r["initial_ppl"] - 57601.60) < 0.1
    assert abs(r["best_ppl"] - 2.12) < 1e-2
    assert abs(r["best_loss"] - 0.7528) < 1e-3
    assert abs(r["recent_avg_lm"] - 6.1159) < 1e-3
    assert abs(r["random_baseline_loss"] - 10.3972) < 1e-3
    assert r["nan_events_total"] == 0

    # Stream diagnostics
    assert r["streams"]["WT103_bytes"] == 2359171
    assert r["streams"]["OWT_bytes"] == 20184043
    assert r["streams"]["Python_bytes"] == 11402085
    assert r["streams"]["GSM8K_bytes"] == 3014537
    assert r["streams"]["MATH_bytes"] == 3473062
    assert r["streams"]["total_bytes"] == 40432898

    # Verdict + atexit
    assert "CONVERGING" in r["verdict_line"]
    assert abs(r["pct_of_random"] - 58.8) < 0.5
    assert r["atexit_clean"] is True


def test_parse_final_summary_diverging_verdict():
    """Different verdict-class wording should still parse."""
    from tools.training_digest import parse_final_summary
    sample = """
--- Training Complete ---

Summary:
  Model:          moe50m_8exp (~62.99M params)
  Steps trained:  5000 / 5000
  Epochs:         0
  Tokens trained: 10240K
  Initial PPL:    57601.60
  Best PPL:       9.63
  Best loss:      2.2654
  Recent avg LM:  9.4500
  Random baseline:10.3972 (PPL=32768)
  NaN events:     5 (total)
  Stream diagnostics:
    [WT103] bytes=2400000  epochs=0
    [TOT] bytes=2400000 (2.29 MB)

Verdict: DIVERGING -- LM loss above random baseline (90.9% of random)

=== Scale-Up Experiment Complete ===

*** ATEXIT: clean process exit ***
"""
    r = parse_final_summary(sample)
    assert r is not None
    assert r["model"] == "moe50m_8exp"
    assert "DIVERGING" in r["verdict_line"]
    assert abs(r["pct_of_random"] - 90.9) < 0.5
    assert r["nan_events_total"] == 5


def test_parse_final_summary_truncated_log_no_atexit():
    """If stdout was truncated mid-write (e.g., process killed before atexit),
    the parser should mark atexit_clean=False but still extract whatever fields are present."""
    from tools.training_digest import parse_final_summary
    sample = """
--- Training Complete ---

Summary:
  Model:          dense50m (~34.62M params)
  Steps trained:  3500 / 5000
  Epochs:         0
  Tokens trained: 7170K
  Initial PPL:    57601.60
  Best PPL:       4.20
  Best loss:      1.5000
  Recent avg LM:  5.5280
  Random baseline:10.3972 (PPL=32768)
  NaN events:     0 (total)
"""
    r = parse_final_summary(sample)
    assert r is not None
    assert r["steps_trained"] == 3500
    assert r["steps_planned"] == 5000
    assert r["atexit_clean"] is False
    assert r.get("verdict_line") in (None, "")


def test_parse_final_summary_returns_none_for_non_match():
    """Without the '--- Training Complete ---' marker, the parser returns None."""
    from tools.training_digest import parse_final_summary
    assert parse_final_summary("just some random text") is None
    assert parse_final_summary("") is None


def test_parse_final_summary_fixture_yields_a42():
    """Sanity check against the live A42 fixture."""
    from tools.training_digest import parse_final_summary
    fixture = REPO / "tests/fixtures/training_digest/phase3_A42_sample.log"
    text = fixture.read_text()
    r = parse_final_summary(text)
    assert r is not None
    assert r["model"] == "dense50m"
    assert r["steps_trained"] == 5000
    assert r["atexit_clean"] is True
    assert "CONVERGING" in r["verdict_line"]


def test_classify_loss_trajectory_monotone_decreasing():
    from tools.training_digest import classify_loss_trajectory
    summaries = [
        {"step": 500,  "avg_lm": 7.0, "best_ppl": 200.0, "tokens_k": 1024, "nan": 0, "epochs": 0},
        {"step": 1000, "avg_lm": 6.5, "best_ppl": 100.0, "tokens_k": 2048, "nan": 0, "epochs": 0},
        {"step": 1500, "avg_lm": 6.0, "best_ppl": 50.0,  "tokens_k": 3072, "nan": 0, "epochs": 0},
        {"step": 2000, "avg_lm": 5.5, "best_ppl": 25.0,  "tokens_k": 4096, "nan": 0, "epochs": 0},
        {"step": 2500, "avg_lm": 5.0, "best_ppl": 12.0,  "tokens_k": 5120, "nan": 0, "epochs": 0},
    ]
    r = classify_loss_trajectory(summaries)
    assert r["shape_class"] == "monotone_decreasing"
    assert "shape_class_evidence" in r
    assert "best_ppl" in r["shape_class_evidence"].lower() or "monotone" in r["shape_class_evidence"].lower()


def test_classify_loss_trajectory_plateau():
    """avg_lm flat after step 1500."""
    from tools.training_digest import classify_loss_trajectory
    summaries = [
        {"step": 500,  "avg_lm": 7.0, "best_ppl": 200.0, "tokens_k": 1024, "nan": 0, "epochs": 0},
        {"step": 1000, "avg_lm": 6.5, "best_ppl": 100.0, "tokens_k": 2048, "nan": 0, "epochs": 0},
        {"step": 1500, "avg_lm": 6.0, "best_ppl": 50.0,  "tokens_k": 3072, "nan": 0, "epochs": 0},
        {"step": 2000, "avg_lm": 6.01, "best_ppl": 50.0, "tokens_k": 4096, "nan": 0, "epochs": 0},
        {"step": 2500, "avg_lm": 6.02, "best_ppl": 50.0, "tokens_k": 5120, "nan": 0, "epochs": 0},
        {"step": 3000, "avg_lm": 6.00, "best_ppl": 50.0, "tokens_k": 6144, "nan": 0, "epochs": 0},
        {"step": 3500, "avg_lm": 6.01, "best_ppl": 50.0, "tokens_k": 7168, "nan": 0, "epochs": 0},
    ]
    r = classify_loss_trajectory(summaries)
    assert r["shape_class"].startswith("plateau"), f"got {r['shape_class']}"
    assert r.get("plateau_start_step") == 1500 or r.get("plateau_start_step") == 2000


def test_classify_loss_trajectory_oscillating():
    """avg_lm zigzags multiple times."""
    from tools.training_digest import classify_loss_trajectory
    summaries = [
        {"step": 500,  "avg_lm": 7.0, "best_ppl": 200.0, "tokens_k": 1024, "nan": 0, "epochs": 0},
        {"step": 1000, "avg_lm": 6.0, "best_ppl": 100.0, "tokens_k": 2048, "nan": 0, "epochs": 0},  # down
        {"step": 1500, "avg_lm": 6.5, "best_ppl": 80.0,  "tokens_k": 3072, "nan": 0, "epochs": 0},  # up
        {"step": 2000, "avg_lm": 5.5, "best_ppl": 50.0,  "tokens_k": 4096, "nan": 0, "epochs": 0},  # down
        {"step": 2500, "avg_lm": 6.2, "best_ppl": 40.0,  "tokens_k": 5120, "nan": 0, "epochs": 0},  # up
        {"step": 3000, "avg_lm": 5.0, "best_ppl": 30.0,  "tokens_k": 6144, "nan": 0, "epochs": 0},  # down
    ]
    r = classify_loss_trajectory(summaries)
    assert r["shape_class"] == "oscillating", f"got {r['shape_class']}"


def test_classify_loss_trajectory_collapsed():
    """best_ppl never improves; final >50% of first."""
    from tools.training_digest import classify_loss_trajectory
    summaries = [
        {"step": 500,  "avg_lm": 7.0, "best_ppl": 100.0, "tokens_k": 1024, "nan": 5, "epochs": 0},
        {"step": 1000, "avg_lm": 7.5, "best_ppl": 100.0, "tokens_k": 2048, "nan": 8, "epochs": 0},
        {"step": 1500, "avg_lm": 8.0, "best_ppl": 100.0, "tokens_k": 3072, "nan": 12, "epochs": 0},
        {"step": 2000, "avg_lm": 8.5, "best_ppl": 100.0, "tokens_k": 4096, "nan": 15, "epochs": 0},
    ]
    r = classify_loss_trajectory(summaries)
    assert r["shape_class"] == "collapsed", f"got {r['shape_class']}"


def test_classify_loss_trajectory_insufficient_data():
    from tools.training_digest import classify_loss_trajectory
    assert classify_loss_trajectory([])["shape_class"] == "insufficient_data"
    assert classify_loss_trajectory([{"step": 500, "avg_lm": 7.0, "best_ppl": 200.0, "tokens_k": 1024, "nan": 0, "epochs": 0}])["shape_class"] == "insufficient_data"


def test_classify_loss_trajectory_returns_metadata():
    """Returns dict with shape_class + evidence + key inflection points."""
    from tools.training_digest import classify_loss_trajectory
    summaries = [
        {"step": 500,  "avg_lm": 7.2, "best_ppl": 200.0, "tokens_k": 1024, "nan": 0, "epochs": 0},
        {"step": 1000, "avg_lm": 6.3, "best_ppl": 56.0,  "tokens_k": 2048, "nan": 0, "epochs": 0},
        {"step": 1500, "avg_lm": 5.0, "best_ppl": 8.12,  "tokens_k": 3072, "nan": 0, "epochs": 0},
        {"step": 2000, "avg_lm": 5.7, "best_ppl": 4.60,  "tokens_k": 4096, "nan": 0, "epochs": 0},
        {"step": 2500, "avg_lm": 5.6, "best_ppl": 4.60,  "tokens_k": 5120, "nan": 0, "epochs": 0},
        {"step": 3000, "avg_lm": 5.7, "best_ppl": 4.60,  "tokens_k": 6144, "nan": 0, "epochs": 0},
        {"step": 3500, "avg_lm": 5.5, "best_ppl": 2.51,  "tokens_k": 7168, "nan": 0, "epochs": 0},
        {"step": 4000, "avg_lm": 5.0, "best_ppl": 2.12,  "tokens_k": 8192, "nan": 0, "epochs": 0},
        {"step": 4500, "avg_lm": 5.4, "best_ppl": 2.12,  "tokens_k": 9216, "nan": 0, "epochs": 0},
    ]
    r = classify_loss_trajectory(summaries)
    assert "shape_class" in r
    assert "shape_class_evidence" in r
    # Best PPL was first achieved at step 4000
    assert r.get("best_ppl_first_seen_step") == 4000
    # Should be one of the valid classes
    assert r["shape_class"] in ("monotone_decreasing", "plateau_after_N", "oscillating", "collapsed", "insufficient_data")


# === MoE regime classifier ===

def test_classify_moe_regime_dense_returns_na():
    from tools.training_digest import classify_moe_regime
    per_layer = [
        {"step": 0,   "layer": 0, "K": 1, "ema": [1.0], "min_load": 1.0, "max_load": 1.0, "ratio": 1.0},
        {"step": 0,   "layer": 1, "K": 1, "ema": [1.0], "min_load": 1.0, "max_load": 1.0, "ratio": 1.0},
        {"step": 100, "layer": 0, "K": 1, "ema": [1.0], "min_load": 1.0, "max_load": 1.0, "ratio": 1.0},
    ]
    summary = [{"step": 0, "worst_ratio": 1.0, "worst_layer": 0, "gate_hit": 0, "kill_hit": 0}]
    r = classify_moe_regime(per_layer, summary)
    assert r["regime_class"] == "n/a"


def test_classify_moe_regime_uniform_K8():
    from tools.training_digest import classify_moe_regime
    per_layer = [
        {"step": s, "layer": l, "K": 8, "ema": [0.125] * 8, "min_load": 0.125, "max_load": 0.125, "ratio": 1.0}
        for s in (0, 100, 1000, 5000)
        for l in range(8)
    ]
    summary = [{"step": s, "worst_ratio": 1.0, "worst_layer": 0, "gate_hit": 0, "kill_hit": 0}
               for s in (0, 100, 1000, 5000)]
    r = classify_moe_regime(per_layer, summary)
    assert r["regime_class"] == "uniform"
    assert r["gate_hit_total"] == 0
    assert r["kill_hit_total"] == 0


def test_classify_moe_regime_collapsed():
    """One expert dominates with ratio > 5."""
    from tools.training_digest import classify_moe_regime
    per_layer = [
        {"step": 4500, "layer": 4, "K": 8,
         "ema": [0.6, 0.05, 0.05, 0.05, 0.05, 0.075, 0.075, 0.05],
         "min_load": 0.05, "max_load": 0.6, "ratio": 12.0},
    ]
    summary = [{"step": 4500, "worst_ratio": 12.0, "worst_layer": 4, "gate_hit": 8, "kill_hit": 2}]
    r = classify_moe_regime(per_layer, summary)
    assert r["regime_class"] == "collapsed"
    assert r["gate_hit_total"] == 8
    assert r["kill_hit_total"] == 2
    # Imbalance step range should include step 4500
    assert any(start <= 4500 <= end for start, end in r["imbalance_step_ranges"])


def test_classify_moe_regime_oscillating():
    """worst_ratio swings: ok (< 1.5) → bad (> 2.0) repeatedly."""
    from tools.training_digest import classify_moe_regime
    per_layer = [
        # Provide K=8 entries so it isn't classified as n/a
        {"step": s, "layer": 0, "K": 8, "ema": [0.125] * 8,
         "min_load": 0.125, "max_load": 0.125, "ratio": 1.0}
        for s in range(0, 6000, 1000)
    ]
    summary = [
        {"step": 0,    "worst_ratio": 1.0, "worst_layer": 0, "gate_hit": 0, "kill_hit": 0},  # ok
        {"step": 500,  "worst_ratio": 2.5, "worst_layer": 0, "gate_hit": 0, "kill_hit": 0},  # bad
        {"step": 1000, "worst_ratio": 1.2, "worst_layer": 0, "gate_hit": 0, "kill_hit": 0},  # ok
        {"step": 1500, "worst_ratio": 2.3, "worst_layer": 0, "gate_hit": 0, "kill_hit": 0},  # bad
        {"step": 2000, "worst_ratio": 1.4, "worst_layer": 0, "gate_hit": 0, "kill_hit": 0},  # ok
        {"step": 2500, "worst_ratio": 2.1, "worst_layer": 0, "gate_hit": 0, "kill_hit": 0},  # bad
    ]
    r = classify_moe_regime(per_layer, summary)
    assert r["regime_class"] == "oscillating"


def test_classify_moe_regime_mostly_uniform_default():
    """K>1 with mild imbalance (EMA std-dev > 0.02) but no class triggers → mostly_uniform.

    EMA [0.16, 0.10, 0.16, 0.10, 0.13, 0.13, 0.12, 0.12] has std-dev ~0.023
    which is above the uniform threshold (0.02), ratio 1.6 is below the collapse
    threshold (5.0), and worst_ratio 1.6 is in the dead-band (1.5-2.0) so it
    neither counts as ok nor bad for oscillation counting.
    """
    from tools.training_digest import classify_moe_regime
    per_layer = [
        {"step": s, "layer": l, "K": 8,
         "ema": [0.16, 0.10, 0.16, 0.10, 0.13, 0.13, 0.12, 0.12],
         "min_load": 0.10, "max_load": 0.16, "ratio": 1.6}
        for s in (0, 1000, 5000)
        for l in range(2)
    ]
    summary = [
        {"step": s, "worst_ratio": 1.6, "worst_layer": 0, "gate_hit": 0, "kill_hit": 0}
        for s in (0, 1000, 5000)
    ]
    r = classify_moe_regime(per_layer, summary)
    assert r["regime_class"] == "mostly_uniform"


def test_classify_moe_regime_empty_returns_na():
    from tools.training_digest import classify_moe_regime
    r = classify_moe_regime([], [])
    assert r["regime_class"] == "n/a"


# === NaN classifier ===

def test_classify_nan_none():
    from tools.training_digest import classify_nan
    r = classify_nan(raw_count=0, step_summary_cumulative=0, fatal=0, rc=0)
    assert r["classification"] == "none"


def test_classify_nan_false_positive_d291_class():
    """D-291 class: raw NaN counter fired but training trajectory unaffected."""
    from tools.training_digest import classify_nan
    r = classify_nan(raw_count=9, step_summary_cumulative=0, fatal=0, rc=0)
    assert r["classification"] == "false_positive"
    assert "D-291" in r["classification_reason"] or "false-positive" in r["classification_reason"].lower()


def test_classify_nan_hard_nan_with_fatal():
    from tools.training_digest import classify_nan
    r = classify_nan(raw_count=5, step_summary_cumulative=3, fatal=2, rc=1)
    assert r["classification"] == "hard_nan"


def test_classify_nan_hard_nan_with_rc_only():
    """rc != 0 alone is enough to classify as hard_nan."""
    from tools.training_digest import classify_nan
    r = classify_nan(raw_count=5, step_summary_cumulative=3, fatal=0, rc=137)
    assert r["classification"] == "hard_nan"


def test_classify_nan_unclassified_inconsistent():
    """raw_count=0 but step_summary>0 — parser inconsistency."""
    from tools.training_digest import classify_nan
    r = classify_nan(raw_count=0, step_summary_cumulative=3, fatal=0, rc=0)
    assert r["classification"] == "unclassified"


# ---------------------------------------------------------------------------
# generate_digest() end-to-end tests (Task 9)
# ---------------------------------------------------------------------------

def _setup_digest_repo(tmp_path):
    """Construct a sandbox repo with a single A42 cell suitable for generate_digest."""
    import json
    from pathlib import Path

    # data/runs/phase3_A42/stdout.log — copy real fixture
    runs_dir = tmp_path / "data/runs/phase3_A42"
    runs_dir.mkdir(parents=True)
    fixture_log = REPO / "tests/fixtures/training_digest/phase3_A42_sample.log"
    (runs_dir / "stdout.log").write_text(fixture_log.read_text())

    # data/checkpoints/phase3_factorial/run_index.json — wrap A42 entry
    ckpt_dir = tmp_path / "data/checkpoints/phase3_factorial"
    ckpt_dir.mkdir(parents=True)
    fixture_entry = json.loads((REPO / "tests/fixtures/training_digest/phase3_A42_run_index_entry.json").read_text())
    (ckpt_dir / "run_index.json").write_text(json.dumps({"A42": fixture_entry}, indent=2))

    return tmp_path


def test_generate_digest_writes_json_sidecar(tmp_path):
    import tools.training_digest as td
    import json
    repo = _setup_digest_repo(tmp_path)

    result = td.generate_digest(cell_id="A42", phase="phase3_factorial", repo_root=repo)
    assert isinstance(result, dict)
    assert result["schema_version"] == "1.0"
    assert result["cell_id"] == "A42"
    assert result["phase"] == "phase3_factorial"

    # File written to data/digests/training/A42.json
    json_path = repo / "data/digests/training/A42.json"
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert on_disk["cell_id"] == "A42"
    assert on_disk["schema_version"] == "1.0"


def test_generate_digest_extracts_config_from_run_index_cmd(tmp_path):
    """Config block (lr, seed, steps_planned) parsed from run_index.json's cmd field."""
    import tools.training_digest as td
    repo = _setup_digest_repo(tmp_path)
    result = td.generate_digest(cell_id="A42", phase="phase3_factorial", repo_root=repo)

    assert result["config"]["model"] == "dense50m"
    assert result["config"]["lr"] == 0.001
    assert result["config"]["seed"] == 42
    assert result["config"]["steps_planned"] == 5000


def test_generate_digest_outcome_block_matches_run_index_and_summary(tmp_path):
    import tools.training_digest as td
    repo = _setup_digest_repo(tmp_path)
    result = td.generate_digest(cell_id="A42", phase="phase3_factorial", repo_root=repo)

    o = result["outcome"]
    assert o["rc"] == 0
    assert o["atexit_clean"] is True
    assert o["fatal"] == 0
    assert "CONVERGING" in o["verdict_line"]
    assert abs(o["best_ppl"] - 2.12) < 1e-2


def test_generate_digest_loss_trajectory_uses_classifier(tmp_path):
    import tools.training_digest as td
    repo = _setup_digest_repo(tmp_path)
    result = td.generate_digest(cell_id="A42", phase="phase3_factorial", repo_root=repo)

    lt = result["loss_trajectory"]
    assert "shape_class" in lt
    assert lt["shape_class"] in ("monotone_decreasing", "plateau_after_N", "oscillating",
                                  "collapsed", "insufficient_data", "noisy_no_class")
    assert isinstance(lt["step_summaries"], list)
    assert len(lt["step_summaries"]) >= 1


def test_generate_digest_nan_incidents_block_classifies_d291():
    """Real A42: raw_count=9, fatal=0, rc=0 → false_positive (D-291)."""
    import tools.training_digest as td

    # Use a fresh tmp_path
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        repo = Path(tmp)
        _setup_digest_repo(repo)
        result = td.generate_digest(cell_id="A42", phase="phase3_factorial", repo_root=repo)

    n = result["nan_incidents"]
    assert n["raw_count_run_index"] == 9
    # Note: step_summary_cumulative may be 0 for A42
    assert n["classification"] == "false_positive"


def test_generate_digest_moe_balance_applicable_false_for_dense(tmp_path):
    """A42 is dense50m — moe_balance.applicable should be False, regime n/a."""
    import tools.training_digest as td
    repo = _setup_digest_repo(tmp_path)
    result = td.generate_digest(cell_id="A42", phase="phase3_factorial", repo_root=repo)

    mb = result["moe_balance"]
    assert mb["applicable"] is False
    assert mb["regime_class"] == "n/a"


def test_generate_digest_writes_anomalies_with_line_refs(tmp_path):
    """Inject a FATAL line into the stdout, verify it surfaces in anomalies."""
    import tools.training_digest as td
    repo = _setup_digest_repo(tmp_path)
    log = repo / "data/runs/phase3_A42/stdout.log"
    txt = log.read_text()
    # Insert a fake FATAL line into the middle
    lines = txt.splitlines()
    insert_at = len(lines) // 2
    lines.insert(insert_at, "FATAL: synthetic_test_fatal — fixture-injected for anomaly test")
    log.write_text("\n".join(lines) + "\n")

    result = td.generate_digest(cell_id="A42", phase="phase3_factorial", repo_root=repo)
    anomalies = result["anomalies"]
    assert any("synthetic_test_fatal" in (a.get("text") or "") for a in anomalies), \
        f"expected synthetic FATAL line surfaced; got {anomalies}"
    assert any(a.get("level") == "FATAL" for a in anomalies)


def test_generate_digest_missing_stdout_returns_none_or_raises(tmp_path):
    import tools.training_digest as td
    repo = _setup_digest_repo(tmp_path)
    # Remove the stdout.log
    (repo / "data/runs/phase3_A42/stdout.log").unlink()
    # Should either return None or raise FileNotFoundError — both are acceptable
    try:
        r = td.generate_digest(cell_id="A42", phase="phase3_factorial", repo_root=repo)
        assert r is None
    except FileNotFoundError:
        pass


def test_generate_digest_log_path_and_metadata(tmp_path):
    """log_path, log_size_bytes, stdout_mtime, digest_generated_at fields."""
    import tools.training_digest as td
    repo = _setup_digest_repo(tmp_path)
    result = td.generate_digest(cell_id="A42", phase="phase3_factorial", repo_root=repo)

    assert result["log_path"].endswith("data/runs/phase3_A42/stdout.log")
    assert result["log_size_bytes"] > 0
    assert "stdout_mtime" in result
    assert "digest_generated_at" in result


# ---------------------------------------------------------------------------
# render_markdown() tests (Task 10)
# ---------------------------------------------------------------------------

def test_render_markdown_canonical_dense_a42(tmp_path):
    """Real A42 digest produces the canonical narrative shape."""
    import tools.training_digest as td
    repo = _setup_digest_repo(tmp_path)
    digest = td.generate_digest("A42", "phase3_factorial", repo_root=repo)
    md = td.render_markdown(digest)

    # Title line includes cell + phase + config
    assert "A42" in md
    assert "Phase 3" in md or "phase3" in md.lower()
    assert "dense50m" in md
    assert "lr=1e-3" in md or "lr=0.001" in md

    # Outcome line
    assert "Outcome" in md
    assert "rc=0" in md
    assert "atexit_clean" in md
    assert "CONVERGING" in md

    # Best PPL line
    assert "2.12" in md

    # Wall line
    assert "12.59" in md or "12.586" in md  # depends on rounding

    # Loss trajectory section
    assert "Loss trajectory" in md
    assert "Shape class" in md or "shape_class" in md.lower()

    # MoE balance section — dense → n/a
    assert "MoE balance" in md
    assert "n/a" in md.lower() or "N/A" in md or "dense" in md.lower()

    # NaN incidents — false_positive
    assert "NaN incidents" in md
    assert "false_positive" in md or "false-positive" in md.lower()
    assert "9" in md  # raw_count

    # Stream diagnostics
    assert "Stream diagnostics" in md
    assert "WT103" in md
    assert "MB" in md  # human-readable byte format

    # Anomalies — none
    assert "Anomalies" in md
    assert "None" in md or "no anomalies" in md.lower()

    # Footer cite
    assert "Generated by tools/training_digest.py" in md or "tools/training_digest.py" in md


def test_render_markdown_with_anomalies(tmp_path):
    """Inject a FATAL line, verify it shows up in the Anomalies section."""
    import tools.training_digest as td
    repo = _setup_digest_repo(tmp_path)
    log = repo / "data/runs/phase3_A42/stdout.log"
    txt = log.read_text()
    lines = txt.splitlines()
    insert_at = len(lines) // 2
    lines.insert(insert_at, "FATAL: synthetic_test_anomaly_for_md")
    log.write_text("\n".join(lines) + "\n")

    digest = td.generate_digest("A42", "phase3_factorial", repo_root=repo)
    md = td.render_markdown(digest)

    assert "Anomalies" in md
    assert "FATAL" in md
    assert "synthetic_test_anomaly_for_md" in md
    # Line ref present
    assert "line" in md.lower() or "@" in md  # one of the conventions


def test_render_markdown_format_size_for_humans(tmp_path):
    """Stream byte counts rendered as human-readable MB / KB."""
    import tools.training_digest as td
    repo = _setup_digest_repo(tmp_path)
    digest = td.generate_digest("A42", "phase3_factorial", repo_root=repo)
    md = td.render_markdown(digest)

    # 2,359,171 bytes ≈ 2.25 MB
    # Should appear as ~2.25 MB or 2.4 MB in the markdown (human-readable)
    # Don't assert exact rounding — just that MB appears with reasonable values
    import re
    mb_matches = re.findall(r"([\d.]+)\s*MB", md)
    assert len(mb_matches) >= 1
    # The total bytes (40.4M) should be present as ~38.6 MB
    nums = [float(x) for x in mb_matches]
    assert any(2.0 <= n <= 50.0 for n in nums)


def test_render_markdown_handles_none_digest():
    """If digest is None (parser couldn't run), render_markdown returns a stub error string."""
    import tools.training_digest as td
    md = td.render_markdown(None)
    assert isinstance(md, str)
    assert len(md) > 0  # non-empty
    # Some indication that the digest was unavailable
    assert "unavailable" in md.lower() or "error" in md.lower() or "no digest" in md.lower()


def test_render_markdown_size_under_3kb(tmp_path):
    """Real A42 digest's markdown should be compact — under 3 KB target."""
    import tools.training_digest as td
    repo = _setup_digest_repo(tmp_path)
    digest = td.generate_digest("A42", "phase3_factorial", repo_root=repo)
    md = td.render_markdown(digest)
    assert len(md) < 3000, f"markdown is {len(md)} bytes, expected <3000"
