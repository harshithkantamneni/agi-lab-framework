"""Calibration rollup — score PI's accuracy at each confidence band.

Reads data/infra/calibration_telemetry.jsonl (records with outcomes scored
via tools/calibration_logger.score_calibration). Buckets records by
declared confidence (0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0) and
computes actual hit rate per bucket.

Produces a markdown report at data/infra/calibration_rollup.md with:
- per-bucket count
- declared confidence range
- actual hit rate
- calibration error (declared mid-point vs actual)
- flag if |error| > 15% (consistent miscalibration)

Run weekly from the runner. Meaningful only after ≥5 records per bucket.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _bucket(confidence: float) -> tuple[float, float, str]:
    """Map a confidence to its bucket (low, high, label)."""
    if confidence < 0.2:
        return (0.0, 0.2, "0-20%")
    if confidence < 0.4:
        return (0.2, 0.4, "20-40%")
    if confidence < 0.6:
        return (0.4, 0.6, "40-60%")
    if confidence < 0.8:
        return (0.6, 0.8, "60-80%")
    return (0.8, 1.0, "80-100%")


def main():
    log_path = REPO / "data/infra/calibration_telemetry.jsonl"
    if not log_path.exists():
        print("no calibration telemetry yet")
        return

    # Group by bucket; track count + outcome distribution
    buckets = defaultdict(
        lambda: {"count": 0, "scored": 0, "hits": 0, "label": "", "low": 0.0, "high": 0.0}
    )
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        conf = rec.get("confidence")
        if conf is None:
            continue
        low, high, label = _bucket(conf)
        b = buckets[label]
        b["label"] = label
        b["low"] = low
        b["high"] = high
        b["count"] += 1
        outcome = rec.get("outcome")
        if outcome is not None:
            b["scored"] += 1
            if outcome:
                b["hits"] += 1

    # Build report
    out_path = REPO / "data/infra/calibration_rollup.md"
    lines = [
        "# Calibration rollup",
        "",
        "Per-bucket accuracy of PI's confidence claims. Meaningful at ≥5 records/bucket.",
        "",
        "| confidence band | n_total | n_scored | actual_hit_rate | midpoint | error | flag |",
        "|---|---|---|---|---|---|---|",
    ]
    for label in ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]:
        if label not in buckets:
            continue
        b = buckets[label]
        midpoint = (b["low"] + b["high"]) / 2
        if b["scored"] >= 1:
            hit_rate = b["hits"] / b["scored"]
            error = hit_rate - midpoint
            flag = ""
            if b["scored"] >= 5 and abs(error) > 0.15:
                if error > 0:
                    flag = "UNDERCONFIDENT (claim lower than actual)"
                else:
                    flag = "OVERCONFIDENT (claim higher than actual)"
            lines.append(
                f"| {label} | {b['count']} | {b['scored']} | {hit_rate*100:.1f}% | "
                f"{midpoint*100:.0f}% | {error*100:+.1f}pp | {flag} |"
            )
        else:
            lines.append(
                f"| {label} | {b['count']} | 0 | (no outcomes) | "
                f"{midpoint*100:.0f}% | — | (awaiting outcomes) |"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"calibration rollup: {sum(b['count'] for b in buckets.values())} records, "
          f"{sum(b['scored'] for b in buckets.values())} scored")


if __name__ == "__main__":
    main()
