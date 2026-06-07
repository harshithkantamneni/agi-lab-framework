"""Tests for tools/artifact_queue_projector.py — artifact-presence queue projection."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FIXTURE = REPO / "tests/fixtures/artifact_projector/sample_schema.yaml"


def _setup_sample_program(tmp_path):
    """Copy the fixture schema into a fresh tmp_path repo for the 'sample_program' program."""
    prog_dir = tmp_path / "programs/sample_program"
    prog_dir.mkdir(parents=True)
    (prog_dir / "artifact_schema.yaml").write_text(FIXTURE.read_text())
    return tmp_path


def test_load_artifact_schema_canonical(tmp_path):
    from tools.artifact_queue_projector import load_artifact_schema
    _setup_sample_program(tmp_path)
    schema = load_artifact_schema("sample_program", repo_root=tmp_path)
    assert schema is not None
    assert schema["program"] == "sample_program"
    assert schema["current_phase"] == "P1"
    assert len(schema["phases"]) == 2


def test_load_artifact_schema_missing_returns_none(tmp_path):
    from tools.artifact_queue_projector import load_artifact_schema
    assert load_artifact_schema("nonexistent", repo_root=tmp_path) is None


def test_load_artifact_schema_malformed_yaml_returns_none(tmp_path):
    from tools.artifact_queue_projector import load_artifact_schema
    prog_dir = tmp_path / "programs/broken"
    prog_dir.mkdir(parents=True)
    (prog_dir / "artifact_schema.yaml").write_text("not: valid: yaml: [")
    assert load_artifact_schema("broken", repo_root=tmp_path) is None


def test_project_artifacts_empty_when_all_exist(tmp_path):
    from tools.artifact_queue_projector import project_artifacts
    _setup_sample_program(tmp_path)
    # Create all 3 active-phase artifacts
    (tmp_path / "programs/sample_program/foundation.md").write_text("done")
    (tmp_path / "programs/sample_program/middle.md").write_text("done")
    (tmp_path / "programs/sample_program/capstone.md").write_text("done")

    items = project_artifacts("sample_program", repo_root=tmp_path)
    assert items == []


def test_project_artifacts_emits_foundation_when_missing(tmp_path):
    from tools.artifact_queue_projector import project_artifacts
    _setup_sample_program(tmp_path)
    # No artifacts exist → only the leaf (no prereqs) should fire

    items = project_artifacts("sample_program", repo_root=tmp_path)
    assert len(items) == 1
    item = items[0]
    assert item["type"] == "apparatus_build"
    assert item["priority"] == "normal"
    assert item["program"] == "sample_program"
    assert item["created_by"] == "artifact_queue_projector"
    assert item["payload"]["_artifact_id"] == "P1.foundation"
    # {program} should be substituted
    assert "sample_program" in item["payload"]["scope"]


def test_project_artifacts_emits_middle_after_foundation_exists(tmp_path):
    from tools.artifact_queue_projector import project_artifacts
    _setup_sample_program(tmp_path)
    (tmp_path / "programs/sample_program/foundation.md").write_text("done")
    # Now middle's prereq is satisfied; foundation already exists; capstone has unsatisfied prereqs

    items = project_artifacts("sample_program", repo_root=tmp_path)
    assert len(items) == 1
    assert items[0]["payload"]["_artifact_id"] == "P1.middle"


def test_project_artifacts_emits_capstone_after_both_prereqs(tmp_path):
    from tools.artifact_queue_projector import project_artifacts
    _setup_sample_program(tmp_path)
    (tmp_path / "programs/sample_program/foundation.md").write_text("done")
    (tmp_path / "programs/sample_program/middle.md").write_text("done")

    items = project_artifacts("sample_program", repo_root=tmp_path)
    assert len(items) == 1
    assert items[0]["payload"]["_artifact_id"] == "P1.capstone"
    assert items[0]["type"] == "phase_advance"


def test_project_artifacts_skips_inactive_phase(tmp_path):
    from tools.artifact_queue_projector import project_artifacts
    _setup_sample_program(tmp_path)
    # P2.inactive_artifact is missing but P2 is not current_phase → skip
    # All P1 artifacts also missing — but only P1 should emit

    items = project_artifacts("sample_program", repo_root=tmp_path)
    assert not any(it["payload"]["_artifact_id"].startswith("P2") for it in items), \
        "P2 artifacts should never emit when current_phase=P1"
    # Also assert SOME P1 item emitted (sanity — avoids vacuous pass on empty list)
    assert any(it["payload"]["_artifact_id"].startswith("P1") for it in items)


def test_project_artifacts_idempotent(tmp_path):
    """Same filesystem state → same items (deterministic IDs via _artifact_id key)."""
    from tools.artifact_queue_projector import project_artifacts
    _setup_sample_program(tmp_path)

    items_a = project_artifacts("sample_program", repo_root=tmp_path)
    items_b = project_artifacts("sample_program", repo_root=tmp_path)
    assert items_a == items_b
    # The _artifact_id in payload makes compute_id deterministic
    ids_a = [it["payload"]["_artifact_id"] for it in items_a]
    ids_b = [it["payload"]["_artifact_id"] for it in items_b]
    assert ids_a == ids_b


def test_project_artifacts_renders_program_substitution(tmp_path):
    """{program} in payload values is interpolated."""
    from tools.artifact_queue_projector import project_artifacts
    _setup_sample_program(tmp_path)

    items = project_artifacts("sample_program", repo_root=tmp_path)
    # The foundation handler's payload has "Build the foundation for {program}"
    foundation = next(it for it in items if it["payload"]["_artifact_id"] == "P1.foundation")
    assert foundation["payload"]["scope"] == "Build the foundation for sample_program"


def test_project_artifacts_missing_schema_returns_empty(tmp_path):
    from tools.artifact_queue_projector import project_artifacts
    items = project_artifacts("no_such_program", repo_root=tmp_path)
    assert items == []


def test_project_real_program_2_emits_apparatus_build(tmp_path, monkeypatch):
    """Sanity check against the LIVE programs/program_2/artifact_schema.yaml."""
    import shutil
    from tools.artifact_queue_projector import project_artifacts

    # Copy the real schema into tmp_path without other artifacts
    src = REPO / "programs/program_2_example/artifact_schema.yaml"
    prog_dir = tmp_path / "programs/program_2_example"
    prog_dir.mkdir(parents=True)
    shutil.copy(src, prog_dir / "artifact_schema.yaml")

    items = project_artifacts("program_2_example", repo_root=tmp_path)
    # Should emit the apparatus_methodology_compliant artifact (no prereqs)
    assert len(items) == 1
    assert items[0]["payload"]["_artifact_id"] == "P11.apparatus_methodology_compliant"
    assert items[0]["type"] == "apparatus_build"


def test_main_inspection_returns_zero(tmp_path, monkeypatch, capsys):
    import tools.artifact_queue_projector as ap
    monkeypatch.setattr(ap, "REPO", tmp_path)
    rc = ap.main(argv=["nonexistent_program"])
    assert rc == 0


def test_artifact_item_dedup_key_present_in_payload(tmp_path):
    """_dedup_key field must be present in every emitted item's payload."""
    from tools.artifact_queue_projector import project_artifacts
    _setup_sample_program(tmp_path)

    items = project_artifacts("sample_program", repo_root=tmp_path)
    assert len(items) >= 1, "expected at least one item to check _dedup_key"
    for item in items:
        assert "_dedup_key" in item["payload"], (
            f"item for artifact {item['payload'].get('_artifact_id')!r} "
            f"is missing _dedup_key"
        )
        # _dedup_key must equal _artifact_id
        assert item["payload"]["_dedup_key"] == item["payload"]["_artifact_id"]


def test_artifact_item_id_stable_across_payload_text_edits(tmp_path):
    """compute_id must be the same before and after editing payload_template text.

    Simulates an operator updating the 'context' or 'next_action' description
    in artifact_schema.yaml without changing the artifact ID.
    """
    import yaml
    from tools.artifact_queue_projector import project_artifacts
    from tools.work_queue import compute_id

    _setup_sample_program(tmp_path)

    # Emit items with original schema
    items_before = project_artifacts("sample_program", repo_root=tmp_path)
    assert len(items_before) >= 1
    item_before = items_before[0]
    id_before = compute_id(
        item_before["type"], item_before["program"], item_before["payload"]
    )

    # Mutate a text field in the schema's payload_template (simulates operator edit)
    schema_path = tmp_path / "programs/sample_program/artifact_schema.yaml"
    schema = yaml.safe_load(schema_path.read_text())
    # Update the first artifact's handler payload_template text
    for phase in schema.get("phases", []):
        for artifact in phase.get("artifacts", []):
            handler = artifact.get("handler", {})
            if handler.get("payload_template"):
                pt = handler["payload_template"]
                if isinstance(pt, dict):
                    # Add or update a descriptive text field
                    pt["context"] = "UPDATED operator description text"
                break
        break
    schema_path.write_text(yaml.dump(schema))

    # Re-emit with modified schema
    items_after = project_artifacts("sample_program", repo_root=tmp_path)
    assert len(items_after) >= 1
    item_after = next(
        (it for it in items_after
         if it["payload"]["_artifact_id"] == item_before["payload"]["_artifact_id"]),
        None,
    )
    assert item_after is not None, "same artifact should still be emitted after text edit"
    id_after = compute_id(
        item_after["type"], item_after["program"], item_after["payload"]
    )

    assert id_before == id_after, (
        f"compute_id changed after payload text edit: {id_before!r} → {id_after!r}. "
        "This would orphan the pending item in the queue."
    )
