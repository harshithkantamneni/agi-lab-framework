"""Validates data/agents/_shared/agent_contracts.json structure + Director contract."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CONTRACTS_PATH = REPO / "data/agents/_shared/agent_contracts.json"


def test_contracts_file_exists():
    assert CONTRACTS_PATH.exists(), f"missing: {CONTRACTS_PATH}"


def test_contracts_parses_as_json():
    data = json.loads(CONTRACTS_PATH.read_text())
    assert isinstance(data, dict)
    assert "schema_version" in data
    assert "contracts" in data


def test_director_contract_present():
    data = json.loads(CONTRACTS_PATH.read_text())
    contracts = data["contracts"]
    assert "director" in contracts
    director = contracts["director"]
    # Required fields per spec
    assert "max_tokens_per_session" in director
    assert "success_criteria" in director
    assert "violation_handler" in director
    assert "max_redispatches_per_session" in director


def test_director_max_tokens_sane():
    data = json.loads(CONTRACTS_PATH.read_text())
    director = data["contracts"]["director"]
    # Should be a positive integer; typically ~500K for opus-4-7 session
    assert isinstance(director["max_tokens_per_session"], int)
    assert director["max_tokens_per_session"] > 0
    assert director["max_tokens_per_session"] <= 1_000_000  # 1M ceiling sanity
    # v1 enforcement-status annotation: reader must know these are metadata-only
    assert "_max_tokens_per_session_note" in director, \
        "max_tokens_per_session must carry _max_tokens_per_session_note clarifying observability-only status in v1"


def test_director_success_criteria_list_of_dicts():
    data = json.loads(CONTRACTS_PATH.read_text())
    director = data["contracts"]["director"]
    crit = director["success_criteria"]
    assert isinstance(crit, list)
    assert len(crit) >= 2, "expected at least 2 success criteria (schema valid + next_action populated)"
    for c in crit:
        assert "id" in c, f"criterion missing id: {c}"
        assert "check" in c, f"criterion missing check: {c}"


def test_director_success_criteria_cover_schema_and_next_action():
    """Spec requires checks for: session_exit present, schema valid, next_action populated."""
    data = json.loads(CONTRACTS_PATH.read_text())
    director = data["contracts"]["director"]
    check_types = {c["check"] for c in director["success_criteria"]}
    # At least these checks must be present:
    expected = {"file_exists", "session_exit_schema_v1.1", "next_action_populated_or_program_complete_true"}
    missing = expected - check_types
    assert not missing, f"missing required checks: {missing}"


def test_director_redispatch_ceiling():
    data = json.loads(CONTRACTS_PATH.read_text())
    director = data["contracts"]["director"]
    assert director["max_redispatches_per_session"] >= 1
    assert director["max_redispatches_per_session"] <= 5  # sanity ceiling
    # Escalation behavior must be specified
    assert "escalation_on_max_redispatches" in director


def test_contracts_has_pi_stub_for_v2():
    """PI contract should be stubbed for v2; can be empty/minimal."""
    data = json.loads(CONTRACTS_PATH.read_text())
    # Either pi is present (stub OK) or there's a v2 marker
    contracts = data["contracts"]
    # We allow pi to be absent in v1, but if present it should at least have schema
    if "pi" in contracts:
        pi = contracts["pi"]
        assert isinstance(pi, dict)
