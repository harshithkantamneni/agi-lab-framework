"""Validates verifier_pairs.json schema and that listed roles exist in agents.json."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAIRS_PATH = REPO / "data/agents/_shared/verifier_pairs.json"
AGENTS_PATH = REPO / "data/agents/agents.json"


def _load_pairs():
    with open(PAIRS_PATH) as f:
        return json.load(f)


def _load_agents():
    with open(AGENTS_PATH) as f:
        return json.load(f)


def test_verifier_pairs_file_exists_and_parses():
    assert PAIRS_PATH.exists(), f"missing: {PAIRS_PATH}"
    pairs = _load_pairs()
    assert isinstance(pairs, dict)
    assert len(pairs) > 0


def test_each_pair_has_required_fields():
    pairs = _load_pairs()
    for producer, spec in pairs.items():
        assert isinstance(spec, dict), f"{producer}: spec not dict"
        # verifier_role may be None (e.g., evaluator has no further verifier)
        assert "verifier_role" in spec, f"{producer}: missing verifier_role"
        assert "trigger" in spec, f"{producer}: missing trigger"
        if spec["verifier_role"] is not None:
            assert "verifier_model" in spec, f"{producer}: missing verifier_model"
            assert "max_iterations" in spec, f"{producer}: missing max_iterations"
            assert isinstance(spec["max_iterations"], int)
            assert 1 <= spec["max_iterations"] <= 5


def test_referenced_roles_exist_in_agents_json():
    pairs = _load_pairs()
    agents = _load_agents()
    for producer, spec in pairs.items():
        assert producer in agents, f"verifier_pairs producer {producer!r} not in agents.json"
        v = spec.get("verifier_role")
        if v is not None:
            assert v in agents, f"verifier_role {v!r} for {producer} not in agents.json"


def test_verifier_models_are_valid():
    pairs = _load_pairs()
    valid = {"claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"}
    for producer, spec in pairs.items():
        if spec.get("verifier_model"):
            assert spec["verifier_model"] in valid, \
                f"{producer}: invalid verifier_model {spec['verifier_model']!r}"


def test_high_stakes_producers_have_verifiers():
    """At minimum, pi/paper_writer/code_reviewer must have verifiers configured."""
    pairs = _load_pairs()
    for required in ("pi", "paper_writer", "code_reviewer"):
        assert required in pairs, f"{required} must have a verifier pair entry"
        assert pairs[required]["verifier_role"] is not None, \
            f"{required} must have a non-null verifier_role"
