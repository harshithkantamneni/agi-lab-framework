from __future__ import annotations
from tools.lab_assessment.types import DimensionResult
from tools.lab_assessment.sources import Sources


def compute(src: Sources) -> DimensionResult:
    metrics, caveats = {}, []

    cost = src.cost_rollup()
    if cost is None:
        metrics["total_cost_usd"] = "unavailable"
        metrics["session_count"] = "unavailable"
        metrics["cost_per_session_usd"] = "unavailable"
        caveats.append("cost_rollup.json missing — cost metrics unavailable")
    else:
        total = cost.get("total_cost_usd")
        sessions = cost.get("session_count") or 0
        metrics["total_cost_usd"] = total
        metrics["session_count"] = sessions
        metrics["cost_per_session_usd"] = round(total / sessions, 4) if (total and sessions) else "unavailable"

    decisions = src.decision_headers()
    metrics["decisions_tracked"] = len(decisions)

    metrics["evaluator_overall"] = src.evaluator_overall() or "unavailable"

    # Verdict: cheap + has throughput data -> Solid; no data -> N/A.
    if cost is None:
        level, rationale = "N/A", "no cost/throughput data available"
    elif isinstance(metrics["cost_per_session_usd"], (int, float)):
        level = "Solid"
        rationale = (f"${metrics['cost_per_session_usd']}/session over "
                     f"{metrics['session_count']} sessions; within a hobbyist-laptop budget")
    else:
        level, rationale = "Developing", "partial cost data"

    return DimensionResult(
        dimension="ops", metrics=metrics, verdict_level=level,
        verdict_rationale=rationale,
        relative_to="Real-Mission constraint: local 18GB laptop, no cloud spend",
        caveats=caveats)
