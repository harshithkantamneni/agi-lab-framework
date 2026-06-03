"""Validates that agents.json tier assignments match the audit at
docs/superpowers/plans/2026-05-04-agent-tier-audit.md.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AGENTS_JSON = REPO / "data/agents/agents.json"

VALID_MODELS = {"claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"}

# Roles that MUST remain opus-4-7. Cross-cutting judgment-tier work where a
# subtle miss costs more than the model premium.
JUDGMENT_TIER_REQUIRED = {
    "pi",
    "director",
    "code_reviewer",
    "evaluator",
    "paper_writer",
    "lab_architect",
    "experimental_methodologist",
    "red_team",
    "grant_reviewer",
    "scientific_reviewer",
    "statistical_reviewer",
    "pre_reg_auditor",
    "measurement_theorist",
    "math_theorist",
    "mechanism_extractor",
    "hypothesis_generator",
    "chief_scientist",
    "kernel_specialist",
    "unanimous_compromise_mediator",
}


def _load_agents():
    with open(AGENTS_JSON) as f:
        return json.load(f)


def test_agents_json_uses_only_supported_models():
    agents = _load_agents()
    for name, defn in agents.items():
        m = defn.get("model")
        assert m in VALID_MODELS, f"{name} has invalid model: {m!r}"


def test_judgment_tier_roles_remain_opus():
    agents = _load_agents()
    for role in JUDGMENT_TIER_REQUIRED:
        if role not in agents:
            continue  # role may not exist; skip
        m = agents[role].get("model")
        assert m == "claude-opus-4-7", \
            f"{role} must be claude-opus-4-7 (judgment tier required), got {m!r}"


def test_every_role_has_a_model():
    agents = _load_agents()
    for name, defn in agents.items():
        assert "model" in defn, f"{name} missing model field"
        assert isinstance(defn["model"], str), f"{name} model is not a string"


def test_at_least_one_non_opus_role_present():
    """Sanity: tiering actually happened. If all roles are still opus-4-7,
    something went wrong with Stage 2 application."""
    agents = _load_agents()
    non_opus = [n for n, d in agents.items() if d.get("model") != "claude-opus-4-7"]
    assert len(non_opus) > 0, "no roles tiered down — Stage 2 application missing"
