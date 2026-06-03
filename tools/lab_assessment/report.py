from __future__ import annotations
import json
from tools.lab_assessment.types import DimensionResult


def render_json(results: list[DimensionResult], generated_at: str, git_sha: str) -> str:
    return json.dumps({"generated_at": generated_at, "git_sha": git_sha,
                       "dimensions": [r.to_dict() for r in results]}, indent=2)


def _fmt_value(v) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt_value(x)}" for k, x in v.items())
    return str(v)


def render_markdown(results: list[DimensionResult], generated_at: str, git_sha: str) -> str:
    lines = [f"# Lab Assessment", "",
             f"*Generated {generated_at} · git {git_sha} · "
             f"`python -m tools.lab_assessment`*", "",
             "## Executive Summary", "",
             "| Dimension | Verdict | One-line |", "|---|---|---|"]
    for r in results:
        lines.append(f"| {r.dimension} | **{r.verdict_level}** | {r.verdict_rationale} |")
    lines.append("")
    for r in results:
        lines += [f"## {r.dimension} — {r.verdict_level}", "",
                  f"*Relative to:* {r.relative_to}", "",
                  f"{r.verdict_rationale}", "", "| Metric | Value |", "|---|---|"]
        for name, val in r.metrics.items():
            lines.append(f"| {name} | {_fmt_value(val)} |")
        if r.caveats:
            lines += ["", "**Caveats:**"] + [f"- {c}" for c in r.caveats]
        lines.append("")
    lines += ["## Methodology", "",
              "Computed by `tools/lab_assessment/` (read-only) from the lab's own data. "
              "Hard metrics are source-grounded; verdicts are labeled qualitative judgments, "
              "not numeric scores. See spec `docs/superpowers/specs/2026-06-02-lab-assessment-design.md`."]
    return "\n".join(lines) + "\n"
