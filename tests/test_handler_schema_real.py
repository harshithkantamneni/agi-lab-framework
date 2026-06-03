"""Validates real work_queue_handlers.md has schema for all 9 item types."""
from pathlib import Path
from tools.handler_schema import load_schema

REPO = Path(__file__).resolve().parent.parent
HANDLERS = REPO / "data/agents/_shared/work_queue_handlers.md"

EXPECTED_TYPES = {
    "phase_advance", "cell_complete", "cell_failed", "operator_nudge",
    "verifier_review", "consolidator_run", "paper_review",
    "carry_forward_resolve", "heartbeat",
    "session_recovery_review", "diagnostic_review",  # added in Stage R4 / Task 12
}


def test_all_item_types_have_schema_entry():
    schemas = load_schema(HANDLERS)
    missing = EXPECTED_TYPES - set(schemas)
    assert not missing, f"missing schema for: {missing}"


def test_each_schema_has_required_keys():
    schemas = load_schema(HANDLERS)
    for type_name, block in schemas.items():
        assert "expected_deliverable_pattern" in block, f"{type_name}: missing key"
        assert "next_action_template" in block, f"{type_name}: missing key"


def test_phase_advance_has_concrete_pattern():
    schemas = load_schema(HANDLERS)
    pa = schemas["phase_advance"]
    assert pa["expected_deliverable_pattern"] is not None
    assert "{program}" in pa["expected_deliverable_pattern"]


def test_cell_failed_opts_out_of_recovery():
    """cell_failed should NOT auto-recover — user might still need to act on the failure."""
    schemas = load_schema(HANDLERS)
    cf = schemas["cell_failed"]
    assert cf["expected_deliverable_pattern"] is None
