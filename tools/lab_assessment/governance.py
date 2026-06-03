from __future__ import annotations
from collections import defaultdict
from tools.lab_assessment.types import DimensionResult
from tools.lab_assessment.sources import Sources


def calibration_error(rows: list[dict]) -> float:
    """Mean absolute error between binned stated confidence and realized accuracy.
    Reads each row's `outcome` (truthy = correct) — the real
    calibration_telemetry.jsonl schema. Rows with confidence or outcome == None
    (not yet scored) are skipped. Bins by rounding confidence to one decimal;
    |mean(confidence_in_bin) - accuracy_in_bin| weighted by bin size."""
    bins = defaultdict(list)
    for r in rows:
        c = r.get("confidence")
        ok = r.get("outcome")
        if c is None or ok is None:
            continue
        bins[round(float(c), 1)].append(bool(ok))
    if not bins:
        return 0.0
    total, err = 0, 0.0
    for conf, outcomes in bins.items():
        n = len(outcomes)
        acc = sum(outcomes) / n
        err += abs(conf - acc) * n
        total += n
    return err / total if total else 0.0


def compute(src: Sources) -> DimensionResult:
    metrics, caveats = {}, []

    cal = src.calibration_telemetry()
    resolved = [r for r in cal
                if r.get("outcome") is not None and r.get("confidence") is not None]
    metrics["calibration_logged_samples"] = len(cal)
    metrics["calibration_resolved_samples"] = len(resolved)
    if resolved:
        metrics["calibration_mean_abs_error"] = round(calibration_error(resolved), 4)
    else:
        metrics["calibration_mean_abs_error"] = "unavailable"
        if cal:
            caveats.append(f"{len(cal)} confidence(s) logged but 0 resolved outcomes "
                           "(outcome=null) — calibration not yet scorable")
        else:
            caveats.append("calibration_telemetry.jsonl missing — calibration unavailable")

    decisions = src.decision_headers()
    metrics["org_adaptation_flags"] = sum("ORG_ADAPTATION" in d["headline"] for d in decisions)
    metrics["evaluator_overall"] = src.evaluator_overall() or "unavailable"

    cae = metrics["calibration_mean_abs_error"]
    if cae == "unavailable":
        level = "N/A"
        rationale = (f"calibration not yet scorable — {metrics['calibration_resolved_samples']} "
                     f"resolved of {metrics['calibration_logged_samples']} logged confidences")
    elif cae <= 0.1:
        level, rationale = "Strong", f"calibration MAE {cae} — stated confidences track outcomes well"
    elif cae <= 0.2:
        level, rationale = "Solid", f"calibration MAE {cae} — reasonably calibrated"
    else:
        level, rationale = "Developing", f"calibration MAE {cae} — confidences drift from outcomes"

    return DimensionResult(
        dimension="governance", metrics=metrics, verdict_level=level,
        verdict_rationale=rationale,
        relative_to="self-stated calibration targets + anti-forgery/governance discipline",
        caveats=caveats)
