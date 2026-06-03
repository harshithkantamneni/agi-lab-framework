import json
from tools.lab_assessment.types import DimensionResult
from tools.lab_assessment.report import render_markdown, render_json


def _results():
    return [
        DimensionResult("research", {"programs_closed": 2}, "Solid", "two programs", "goals", []),
        DimensionResult("benchmarks", {"MMLU": {"accuracy": 0.29}}, "Weak", "tiny models", "baseline",
                        ["near-random expected"]),
    ]


def test_render_markdown_has_summary_and_sections():
    md = render_markdown(_results(), generated_at="2026-06-02T00:00:00Z", git_sha="abc123")
    assert "# Lab Assessment" in md
    assert "## Executive Summary" in md
    assert "research" in md and "benchmarks" in md
    assert "Solid" in md and "Weak" in md
    assert "abc123" in md
    assert "near-random expected" in md   # caveat surfaced


def test_render_json_round_trips():
    js = render_json(_results(), generated_at="2026-06-02T00:00:00Z", git_sha="abc123")
    data = json.loads(js)
    assert data["git_sha"] == "abc123"
    assert {d["dimension"] for d in data["dimensions"]} == {"research", "benchmarks"}
    assert data["dimensions"][0]["verdict"]["level"] == "Solid"
