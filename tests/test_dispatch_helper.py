"""Tests for tools/dispatch_helper.py — model selection at dispatch time."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.dispatch_helper import dispatch


def test_uses_role_default_when_no_signals():
    """findings_curator default in agents.json is sonnet-4-6 post-Stage-2."""
    result = dispatch("findings_curator", "compile the latest bibliography updates")
    assert result["model"] == "claude-sonnet-4-6"  # role default


def test_upgrades_to_opus_on_review_keywords():
    result = dispatch("findings_curator", "review and propose changes to the procedural")
    assert result["model"] == "claude-opus-4-7"


def test_upgrades_haiku_default_to_sonnet_on_implement_keywords():
    """Hypothetical haiku-default role + implement keyword → sonnet upgrade."""
    # findings_curator is sonnet by default; "implement" keeps it sonnet
    result = dispatch("findings_curator", "implement the fix from yesterday's diagnostic")
    assert result["model"] == "claude-sonnet-4-6"


def test_keeps_haiku_on_extract_keywords_for_non_judgment_role():
    """tooling_engineer is sonnet by default; 'extract' downgrades to haiku."""
    result = dispatch("tooling_engineer", "extract last 5 D-N entries into a json file")
    assert result["model"] == "claude-haiku-4-5"


def test_does_not_downgrade_opus_role_on_extract_keywords():
    """Director is opus-tier; 'extract' should not downgrade to haiku, just reduce effort."""
    result = dispatch("director", "extract latest D-N from log.md")
    assert result["model"] == "claude-opus-4-7"
    assert result["effort"] == "low"


def test_unknown_role_falls_back_to_sonnet():
    result = dispatch("nonexistent_role", "do a thing")
    assert result["model"] == "claude-sonnet-4-6"


def test_returns_reasoning_string():
    result = dispatch("findings_curator", "archive last week")
    assert isinstance(result.get("reasoning"), str)
    assert len(result["reasoning"]) > 0
