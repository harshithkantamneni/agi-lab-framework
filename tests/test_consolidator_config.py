"""Validates that the consolidator agent is configured correctly in agents.json
and that its procedural exists.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AGENTS_JSON = REPO / "data/agents/agents.json"
CONSOLIDATOR_PROCEDURAL = REPO / "data/agents/consolidator/procedural.md"


def _load_agents():
    with open(AGENTS_JSON) as f:
        return json.load(f)


def test_consolidator_is_in_agents_json():
    agents = _load_agents()
    assert "consolidator" in agents, "consolidator agent missing from agents.json"


def test_consolidator_runs_at_haiku_tier():
    agents = _load_agents()
    assert agents["consolidator"]["model"] == "claude-haiku-4-5", \
        f"consolidator must be haiku-4-5 (off critical path); got {agents['consolidator'].get('model')}"


def test_consolidator_has_required_fields():
    agents = _load_agents()
    c = agents["consolidator"]
    for f in ("description", "model", "prompt"):
        assert f in c, f"consolidator missing {f}"
    assert len(c["prompt"]) > 50, "consolidator prompt looks too short"


def test_consolidator_procedural_file_exists():
    assert CONSOLIDATOR_PROCEDURAL.exists(), \
        f"missing: {CONSOLIDATOR_PROCEDURAL}"
    text = CONSOLIDATOR_PROCEDURAL.read_text()
    # Procedural should mention the trigger conditions, archival, breadcrumbs
    for keyword in ("trigger", "archive", "breadcrumb", "log.md", "BLOCKED"):
        assert keyword.lower() in text.lower(), \
            f"procedural missing keyword: {keyword}"
