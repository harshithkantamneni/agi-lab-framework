"""Validates tools/handler_schema.py — YAML-block parser for work_queue_handlers.md."""
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests/fixtures/work_queue_handlers_sample.md"


def test_load_schema_parses_full_block():
    from tools.handler_schema import load_schema
    schemas = load_schema(FIXTURE)
    assert "phase_advance" in schemas
    pa = schemas["phase_advance"]
    assert pa["expected_deliverable_pattern"] == "programs/{program}/phase{to_phase_num}_*.md"
    assert pa["next_action_template"]["type"] == "phase_advance"
    assert pa["next_action_template"]["payload"]["from_phase"] == "{to_phase}"


def test_load_schema_parses_null_block():
    from tools.handler_schema import load_schema
    schemas = load_schema(FIXTURE)
    cf = schemas["cell_failed"]
    assert cf["expected_deliverable_pattern"] is None
    assert cf["next_action_template"] is None


def test_load_schema_skips_legacy_no_block():
    from tools.handler_schema import load_schema
    schemas = load_schema(FIXTURE)
    # legacy_no_schema has no YAML block; load_schema should return None or absent for it
    assert schemas.get("legacy_no_schema") is None or schemas.get("legacy_no_schema") == {}


def test_render_template_substitutes_payload_vars():
    from tools.handler_schema import render_template
    template = "programs/{program}/phase{to_phase_num}_*.md"
    payload = {"program": "program_2_dense_vs_moe_sub100m", "to_phase": "P10"}
    rendered = render_template(template, payload)
    assert rendered == "programs/program_2_dense_vs_moe_sub100m/phase10_*.md"


def test_render_template_handles_missing_var():
    from tools.handler_schema import render_template
    # Missing var — leave placeholder as-is
    rendered = render_template("programs/{program}/foo", {})
    assert rendered == "programs/{program}/foo"


def test_render_template_dict_recurses():
    from tools.handler_schema import render_template
    template = {
        "type": "phase_advance",
        "payload": {"from_phase": "{to_phase}", "to_phase": "{to_phase}_close"}
    }
    payload = {"to_phase": "P10"}
    rendered = render_template(template, payload)
    assert rendered["payload"]["from_phase"] == "P10"
    assert rendered["payload"]["to_phase"] == "P10_close"
