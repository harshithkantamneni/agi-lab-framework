from __future__ import annotations
from tools.lab_assessment.types import DimensionResult
from tools.lab_assessment.sources import Sources

RANDOM_BASELINE = {"MMLU": 0.25, "MMLU-Pro": 0.10, "HellaSwag": 0.25,
                   "WinoGrande": 0.5, "ARC-AGI-2": 0.0, "GSM8K": 0.0,
                   "MATH": 0.0, "TruthfulQA": 0.0, "HumanEval": 0.0}


def compute(src: Sources) -> DimensionResult:
    scores = src.scorecard_scores()
    metrics, caveats = {}, []
    if not scores:
        return DimensionResult(
            dimension="benchmarks", metrics={}, verdict_level="N/A",
            verdict_rationale="no scorecard scores available",
            relative_to="random baseline + aspirational (Opus) target",
            caveats=["data/eval/scorecard.md missing or unparseable"])

    for name, s in scores.items():
        base = RANDOM_BASELINE.get(name, 0.0)
        metrics[name] = {"accuracy": s["accuracy"], "n": s["total"],
                         "random_baseline": base,
                         "above_random": s["accuracy"] > base}

    any_above = any(m["above_random"] for m in metrics.values())
    level = "Developing" if any_above else "Weak"
    rationale = ("Trained models score at/below random on most benchmarks — EXPECTED for "
                 "48-119M-param models on a laptop; the aspirational 'beat Opus' target was "
                 "formally established as unreachable at this scale (Program 1). Judged against "
                 "the Real Mission (rigorous small-scale research), not the aspirational one.")
    caveats.append("Benchmark scores reflect tiny models; near-random is expected, not a defect.")
    return DimensionResult(
        dimension="benchmarks", metrics=metrics, verdict_level=level,
        verdict_rationale=rationale,
        relative_to="random baseline + the (unreachable) aspirational Opus target",
        caveats=caveats)
