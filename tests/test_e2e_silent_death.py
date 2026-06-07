"""End-to-end test: simulate D-313-like silent-death scenario, verify auto-recovery.

Reproduces the scenario where Director claims a phase_advance work item,
dispatches a subagent, receives the deliverable, but exits without writing
data/session_exit.md (the actual D-313 failure mode). With RO-CO v1 in
place, tools/post_director.py should detect the orphan claim, find the
deliverable on disk via the handler schema's expected_deliverable_pattern,
and:
  - Append a synthetic recovery log entry to log.md.
  - Complete the orphan claim (outcome.status=success_recovered).
  - Enqueue the follow-on phase_advance per next_action_template.
  - Enqueue a session_recovery_review for operator inspection.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _setup_repo(tmp_path: Path) -> Path:
    """Build a minimal repo skeleton matching the runner's expectations."""
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/work_queue").mkdir(parents=True)
    (tmp_path / "data/work_queue/completed").mkdir()
    (tmp_path / "data/work_queue/failed").mkdir()
    (tmp_path / "data/infra").mkdir(parents=True)
    (tmp_path / "data/agents/_shared").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text(
        "# Log\n\n*preamble*\n\n---\n\n"
    )
    (tmp_path / "data/memories/current.md").write_text(
        "# Current\n\nStatus: Phase 3 P10 MECHANISM OPEN\n"
    )
    # Copy the real handler schema so post_director can find expected_deliverable_pattern
    src_handlers = REPO / "data/agents/_shared/work_queue_handlers.md"
    (tmp_path / "data/agents/_shared/work_queue_handlers.md").write_text(
        src_handlers.read_text()
    )
    return tmp_path


def test_d313_silent_death_simulation_recovers(tmp_path, monkeypatch):
    """Full simulation of D-313: claim + deliverable + no session_exit -> auto-recovery."""
    import tools.post_director as pd
    import tools.work_queue as wq

    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    # 1. Director "claims" the phase_advance item (simulating session start)
    item_id = wq.enqueue({
        "type": "phase_advance",
        "priority": "normal",
        "program": "program_2_example",
        "created_by": "test_d313_simulation",
        "payload": {
            "program": "program_2_example",
            "from_phase": "P9",
            "to_phase": "P10",
            "context": "P9 ANALYZE closed; advance to P10 mechanism extraction",
        },
    })
    wq.claim(item_id, "director_session_d_test_313")

    # 2. Subagent "produces" the deliverable matching the phase_advance schema
    #    (programs/{program}/phase{to_phase_num}_*.md)
    deliv = (
        tmp_path / "programs/program_2_example"
        / "phase10_mechanism_report.md"
    )
    deliv.parent.mkdir(parents=True)
    # Write a substantial body so size > 100-byte threshold and feels real
    deliv.write_text(
        "# Phase 3 P10 Mechanism Report\n\n"
        "## §1 Purpose\n\n"
        "Test deliverable simulating D-313 phase10_mechanism_report.md.\n\n"
        + ("Body content. " * 100)
    )

    # Critical: deliverable mtime must be > claim mtime, so touch after a tiny pause
    time.sleep(0.05)
    deliv.touch()

    # 3. Director "silent-dies" — no session_exit.md written
    assert not (tmp_path / "data/session_exit.md").exists()

    # 4. Runner invokes post_director.py
    rc = pd.main()
    assert rc == 0

    # 5. Assert recovery effects
    s = wq.summary()
    assert s["claimed"] == 0, f"orphan claim should have been resolved, summary={s}"
    assert s["completed_today"] >= 1, f"orphan should have been completed, summary={s}"

    # Pending should contain at least: 1 follow-on phase_advance + 1 session_recovery_review
    pending_path = tmp_path / "data/work_queue/pending.jsonl"
    pending_records = []
    if pending_path.exists():
        for line in pending_path.read_text().splitlines():
            if line.strip():
                pending_records.append(json.loads(line))
    pending_types = [r["type"] for r in pending_records]
    assert "session_recovery_review" in pending_types, \
        f"expected session_recovery_review enqueued, got types={pending_types}"
    assert "phase_advance" in pending_types, \
        f"expected follow-on phase_advance from next_action_template, got types={pending_types}"

    # The session_recovery_review payload should reference the orphan and the deliverable
    recovery = next(r for r in pending_records if r["type"] == "session_recovery_review")
    assert recovery["payload"]["orphan_item_id"] == item_id
    assert "phase10_mechanism_report.md" in recovery["payload"]["deliverable"]

    # 6. Synthetic log entry must be present and clearly marked as RECOVERY
    log = (tmp_path / "data/memories/log.md").read_text()
    assert "RECOVERY" in log, f"log missing recovery marker:\n{log[:500]}"
    assert item_id in log, f"log missing orphan claim ID, got:\n{log[:500]}"
    assert "auto-recovered" in log.lower() or "auto_recovered" in log.lower()

    # 7. Telemetry record was written
    telem = tmp_path / "data/infra/post_director_telemetry.jsonl"
    assert telem.exists()
    lines = [l for l in telem.read_text().splitlines() if l.strip()]
    assert len(lines) >= 1
    rec = json.loads(lines[-1])
    assert rec["branch_taken"] == "silent_death", \
        f"expected silent_death branch, got {rec['branch_taken']!r}"


def test_d313_simulation_idempotent(tmp_path, monkeypatch):
    """Running post_director.main() twice over the same scenario produces same end state."""
    import tools.post_director as pd
    import tools.work_queue as wq

    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal",
        "program": "test_program", "created_by": "test",
        "payload": {"program": "test_program", "to_phase": "P10"},
    })
    wq.claim(item_id, "director_session_test")

    deliv = tmp_path / "programs/test_program/phase10_mech.md"
    deliv.parent.mkdir(parents=True)
    deliv.write_text("# x\n" + "y" * 500)
    time.sleep(0.05)
    deliv.touch()

    # First run
    pd.main()
    log_first = (tmp_path / "data/memories/log.md").read_text()
    s_first = wq.summary()

    # Second run: claim is gone, no session_exit -> branch_taken=silent_death_no_claim
    # No additional log mutations, no double-enqueue
    pd.main()
    log_second = (tmp_path / "data/memories/log.md").read_text()
    s_second = wq.summary()

    assert log_first == log_second
    assert s_first["completed_today"] == s_second["completed_today"]
    assert s_first["pending"] == s_second["pending"]


def test_d313_simulation_no_deliverable_fails_claim(tmp_path, monkeypatch):
    """If Director silent-died WITHOUT producing the deliverable, the claim should fail."""
    import tools.post_director as pd
    import tools.work_queue as wq

    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal",
        "program": "test_program", "created_by": "test",
        "payload": {"program": "test_program", "to_phase": "P10"},
    })
    wq.claim(item_id, "director_session_test")

    # No deliverable created — silent-died WITHOUT producing artifact
    pd.main()

    s = wq.summary()
    assert s["claimed"] == 0
    assert s["failed_today"] >= 1
    # diagnostic_review should be enqueued (not session_recovery_review)
    pending_path = tmp_path / "data/work_queue/pending.jsonl"
    pending_types = []
    if pending_path.exists():
        for line in pending_path.read_text().splitlines():
            if line.strip():
                pending_types.append(json.loads(line)["type"])
    assert "diagnostic_review" in pending_types
    assert "session_recovery_review" not in pending_types
