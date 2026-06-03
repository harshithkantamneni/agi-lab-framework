"""Validates that every agent procedural references the self-escalation contract."""
from __future__ import annotations
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO / "data/agents"
SHARED_CONTRACT = AGENTS_DIR / "_shared/self_escalation_contract.md"

CONTRACT_MARKER = "Self-Escalation Contract"


def _agent_procedurals():
    return sorted(p for p in AGENTS_DIR.glob("*/procedural.md"))


def test_shared_contract_file_exists():
    assert SHARED_CONTRACT.exists(), \
        f"shared contract file missing: {SHARED_CONTRACT}"
    text = SHARED_CONTRACT.read_text()
    assert "Tier A" in text
    assert "Tier B" in text
    assert "Tier C" in text
    assert "BLOCKED" in text


def test_every_procedural_references_contract():
    procedurals = _agent_procedurals()
    assert len(procedurals) > 20, \
        f"unexpectedly few procedurals: {len(procedurals)}"
    missing = []
    for proc in procedurals:
        if CONTRACT_MARKER not in proc.read_text():
            missing.append(str(proc.relative_to(REPO)))
    assert not missing, \
        f"procedurals missing self-escalation contract reference: {missing}"


def test_every_procedural_points_at_context_brief():
    """Every agent must know where to find program/phase context on demand."""
    procedurals = _agent_procedurals()
    missing = []
    for proc in procedurals:
        text = proc.read_text()
        if "data/memories/context_brief.md" not in text:
            missing.append(str(proc.relative_to(REPO)))
    assert not missing, \
        f"procedurals missing context_brief pointer: {missing}"
