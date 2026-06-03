"""Phase 1 end-to-end validation — B' + C' + D' chained scenarios.

Simulates the actual failure modes that motivated Phase 1:
- D-313 (Director silent-death with next_action=null) — B' catches via schema rejection
- Persistent looping violations — D' catches via redispatch ceiling escalation
- Missing critical-path artifacts — C' catches via artifact-presence queue projection

These tests use a sandboxed tmp_path repo with the Phase 1 contracts + schemas
copied in. They drive the same code paths the live runner uses.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _setup_phase1_repo(tmp_path):
    """Build a sandbox repo with all Phase 1 infrastructure: schema doc + contracts + agents/_shared."""
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/work_queue").mkdir(parents=True)
    (tmp_path / "data/work_queue/completed").mkdir()
    (tmp_path / "data/work_queue/failed").mkdir()
    (tmp_path / "data/infra").mkdir(parents=True)
    (tmp_path / "data/agents/_shared").mkdir(parents=True)

    # Minimal log + current
    (tmp_path / "data/memories/log.md").write_text("# Log\n\n*preamble*\n\n---\n\n")
    (tmp_path / "data/memories/current.md").write_text(
        "# Current state\n\nactive_program: e2e_test_program\n\n## Phase: P1 IN PROGRESS\n"
    )

    # Copy contracts file
    shutil.copy(
        REPO / "data/agents/_shared/agent_contracts.json",
        tmp_path / "data/agents/_shared/agent_contracts.json",
    )
    # Copy handler schema
    shutil.copy(
        REPO / "data/agents/_shared/work_queue_handlers.md",
        tmp_path / "data/agents/_shared/work_queue_handlers.md",
    )
    return tmp_path


def test_scenario_a_d313_pattern_schema_violation_blocks_mutations(tmp_path, monkeypatch):
    """Scenario A: Director writes status=success + next_action=null → schema rejection,
    no mutations applied, redispatch flag written, counter=1."""
    import tools.post_director as pd
    import tools.work_queue as wq
    repo = _setup_phase1_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    monkeypatch.setattr(wq, "_REPO", repo)

    # Set up an active claim (the work Director was processing)
    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal",
        "program": "e2e_test_program", "created_by": "test",
        "payload": {"to_phase": "P10"}
    })
    wq.claim(item_id, "director_session_d_t313")

    # Snapshot pre-state
    log_before = (repo / "data/memories/log.md").read_text()
    cur_before = (repo / "data/memories/current.md").read_text()
    summary_before = wq.summary()

    # Director writes D-313-pattern session_exit
    invalid_se = {
        "status": "success",
        "session_id": "D-T313",
        "claimed_item_id": item_id,
        "log_entry_text": "D-T313: produced deliverable_X.md. Next session should dispatch verifier.\n",
        "current_md_patches": [{"old": "P1 IN PROGRESS", "new": "P1 DELIVERABLE_X PRODUCED"}],
        "next_action": None,
        # program_complete NOT set — the D-313 mistake
    }
    (repo / "data/session_exit.md").write_text(
        f"```json\n{json.dumps(invalid_se)}\n```\n\nreason: GRACEFUL_CHECKPOINT\n"
    )

    # Run post_director — the equivalent of one runner iteration
    pd.main()

    # B' assertions: no mutations applied
    assert (repo / "data/memories/log.md").read_text() == log_before, \
        "log.md should NOT be modified on violation"
    assert (repo / "data/memories/current.md").read_text() == cur_before, \
        "current.md should NOT be modified on violation"
    summary_after = wq.summary()
    assert summary_after["claimed"] == summary_before["claimed"], \
        "claim should NOT be completed on violation"

    # B' assertions: redispatch flag written + counter at 1
    flag = repo / "data/session_exit_redispatch_pending.flag"
    assert flag.exists(), "redispatch flag should be written for schema violation"

    counter = json.loads((repo / "data/infra/director_redispatch_count.json").read_text())
    assert counter["current_session_id"] == "D-T313"
    assert counter["count"] == 1

    # B' assertions: telemetry recorded the violation
    telem = (repo / "data/infra/post_director_telemetry.jsonl").read_text().strip().split("\n")
    last_record = json.loads(telem[-1])
    assert last_record["branch_taken"] == "schema_violation"
    assert "next_action" in last_record["violation_reason"]


def test_scenario_b_3x_violation_triggers_escalation(tmp_path, monkeypatch):
    """Scenario B: Same session_id violates 3 times → escalation kicks in."""
    import tools.post_director as pd
    import tools.work_queue as wq
    repo = _setup_phase1_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    monkeypatch.setattr(wq, "_REPO", repo)

    invalid_se_text = (
        "```json\n"
        + json.dumps({
            "status": "success", "session_id": "D-LOOP",
            "claimed_item_id": "wq-loop", "next_action": None,
        })
        + "\n```\n\nreason: GRACEFUL_CHECKPOINT\n"
    )

    # Three consecutive violations
    for iteration in range(3):
        (repo / "data/session_exit.md").write_text(invalid_se_text)
        pd.main()

    # After 3rd violation, escalation should have fired
    operator_review = repo / "data/operator_review_pending.md"
    assert operator_review.exists(), "operator review file should exist after 3rd violation"

    review_text = operator_review.read_text()
    assert "D-LOOP" in review_text, "operator review must reference the looping session"
    assert "3" in review_text, "operator review must show the count"

    # Redispatch flag should NOT exist (escalation halts redispatch)
    flag = repo / "data/session_exit_redispatch_pending.flag"
    assert not flag.exists(), "redispatch flag should be cleared on escalation"

    # Final telemetry should show escalation
    telem_records = []
    for line in (repo / "data/infra/post_director_telemetry.jsonl").read_text().strip().split("\n"):
        if line.strip():
            telem_records.append(json.loads(line))
    last = telem_records[-1]
    assert "escalat" in last["branch_taken"].lower() or "max_redispatch" in last["branch_taken"].lower(), \
        f"final telemetry branch_taken should indicate escalation, got: {last['branch_taken']!r}"


def test_scenario_c_artifact_projection_progressively_emits_chain(tmp_path, monkeypatch):
    """Scenario C: artifact_schema.yaml chain (A → B → C); projector emits items
    only for satisfied prereqs; creating artifact A unlocks B."""
    import tools.artifact_queue_projector as ap
    repo = _setup_phase1_repo(tmp_path)
    monkeypatch.setattr(ap, "REPO", repo)

    # Create a schema with chain: A (leaf) → B (depends on A) → C (depends on A+B)
    schema_yaml = """
program: e2e_test_program
current_phase: P1
schema_version: "1.0"
phases:
  - id: P1
    description: "E2E test phase"
    artifacts:
      - id: P1.foundation
        path: "programs/{program}/foundation.md"
        description: "First, no prereqs"
        handler:
          type: apparatus_build
          priority: normal
          payload_template:
            next_action: "build foundation for {program}"
      - id: P1.middle
        path: "programs/{program}/middle.md"
        description: "Depends on foundation"
        prereqs: [P1.foundation]
        handler:
          type: apparatus_build
          priority: normal
          payload_template:
            next_action: "build middle for {program}"
      - id: P1.capstone
        path: "programs/{program}/capstone.md"
        description: "Depends on both"
        prereqs: [P1.foundation, P1.middle]
        handler:
          type: phase_advance
          priority: normal
          payload_template:
            from_phase: P1_middle_done
            to_phase: P1_capstone
"""
    prog_dir = repo / "programs/e2e_test_program"
    prog_dir.mkdir(parents=True)
    (prog_dir / "artifact_schema.yaml").write_text(schema_yaml)

    # State 1: no artifacts exist → only foundation should emit
    items1 = ap.project_artifacts("e2e_test_program", repo_root=repo)
    assert len(items1) == 1
    assert items1[0]["payload"]["_artifact_id"] == "P1.foundation"

    # State 2: create foundation → only middle should emit
    (prog_dir / "foundation.md").write_text("foundation content")
    items2 = ap.project_artifacts("e2e_test_program", repo_root=repo)
    assert len(items2) == 1
    assert items2[0]["payload"]["_artifact_id"] == "P1.middle"

    # State 3: create middle too → only capstone should emit
    (prog_dir / "middle.md").write_text("middle content")
    items3 = ap.project_artifacts("e2e_test_program", repo_root=repo)
    assert len(items3) == 1
    assert items3[0]["payload"]["_artifact_id"] == "P1.capstone"

    # State 4: create capstone too → empty (program done)
    (prog_dir / "capstone.md").write_text("capstone content")
    items4 = ap.project_artifacts("e2e_test_program", repo_root=repo)
    assert items4 == []


def test_scenario_d_valid_session_exit_passes_all_layers(tmp_path, monkeypatch):
    """Sanity: a properly-formed session_exit passes both schema AND contract,
    mutations are applied, no flag written."""
    import tools.post_director as pd
    import tools.work_queue as wq
    repo = _setup_phase1_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    monkeypatch.setattr(wq, "_REPO", repo)

    # Set up an active claim
    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal",
        "program": "e2e_test_program", "created_by": "test",
        "payload": {"to_phase": "P10"}
    })
    wq.claim(item_id, "director_session_d_valid")

    # Valid session_exit with next_action populated
    valid_se = {
        "status": "success",
        "session_id": "D-VALID",
        "claimed_item_id": item_id,
        "log_entry_text": "D-VALID: clean session\n",
        "current_md_patches": [{"old": "P1 IN PROGRESS", "new": "P1 STEP COMPLETE"}],
        "next_action": {
            "type": "phase_advance", "priority": "normal",
            "program": "e2e_test_program", "created_by": "director_session_d_valid",
            "payload": {"to_phase": "P2"}
        },
    }
    (repo / "data/session_exit.md").write_text(
        f"```json\n{json.dumps(valid_se)}\n```\n\nreason: GRACEFUL_CHECKPOINT\n"
    )

    pd.main()

    # No flag (no violation)
    flag = repo / "data/session_exit_redispatch_pending.flag"
    assert not flag.exists()

    # No operator_review (no escalation)
    op_review = repo / "data/operator_review_pending.md"
    assert not op_review.exists()

    # Mutations applied: log updated, current updated, claim completed, next_action enqueued
    log = (repo / "data/memories/log.md").read_text()
    assert "D-VALID" in log

    current = (repo / "data/memories/current.md").read_text()
    assert "P1 STEP COMPLETE" in current

    summary = wq.summary()
    assert summary["claimed"] == 0
    assert summary["completed_today"] >= 1
    assert summary["pending"] >= 1  # the next_action enqueued

    # Telemetry shows structured_success
    telem = (repo / "data/infra/post_director_telemetry.jsonl").read_text().strip().split("\n")
    last = json.loads(telem[-1])
    assert last["branch_taken"] == "structured_success"


def test_scenario_e_counter_resets_on_new_session_id(tmp_path, monkeypatch):
    """Different session_id triggers counter reset to 1, not accumulation."""
    import tools.post_director as pd
    repo = _setup_phase1_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)

    invalid_template = lambda sid: (
        "```json\n"
        + json.dumps({
            "status": "success", "session_id": sid,
            "claimed_item_id": f"wq-{sid}", "next_action": None,
        })
        + "\n```\n\nreason: x\n"
    )

    # Session A violates twice
    (repo / "data/session_exit.md").write_text(invalid_template("D-A"))
    pd.main()
    (repo / "data/session_exit.md").write_text(invalid_template("D-A"))
    pd.main()
    counter = json.loads((repo / "data/infra/director_redispatch_count.json").read_text())
    assert counter["current_session_id"] == "D-A"
    assert counter["count"] == 2

    # Session B violates once — counter resets
    (repo / "data/session_exit.md").write_text(invalid_template("D-B"))
    pd.main()
    counter = json.loads((repo / "data/infra/director_redispatch_count.json").read_text())
    assert counter["current_session_id"] == "D-B"
    assert counter["count"] == 1  # NOT 3
