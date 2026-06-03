"""Validates tools/post_director.py — runner-side close-out finalizer."""
from pathlib import Path
import json
import sys

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _setup_repo(tmp_path):
    """Build a minimal repo skeleton matching post_director.py's expectations."""
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
        "# Current\n\nStatus: P9 ANALYZE\n"
    )
    return tmp_path


def test_module_imports(tmp_path, monkeypatch):
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    assert hasattr(pd, "main")
    assert pd.main() == 0


def test_setup_repo_helper_creates_skeleton(tmp_path):
    repo = _setup_repo(tmp_path)
    assert (repo / "data/memories/log.md").exists()
    assert (repo / "data/memories/current.md").exists()
    assert (repo / "data/work_queue").is_dir()
    assert (repo / "data/work_queue/completed").is_dir()
    assert (repo / "data/work_queue/failed").is_dir()


def test_fixtures_exist():
    """All four fixture files exist for use by upcoming tests."""
    fixtures = REPO / "tests/fixtures"
    for name in ("session_exit_success.md", "session_exit_failure.md",
                 "session_exit_no_op.md", "session_exit_legacy.md"):
        assert (fixtures / name).exists(), f"missing fixture: {name}"


def test_success_fixture_parses_as_json():
    """Fixture's JSON block must be valid JSON."""
    import re
    text = (REPO / "tests/fixtures/session_exit_success.md").read_text()
    m = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    assert m, "session_exit_success.md missing JSON code block"
    data = json.loads(m.group(1))
    assert data["status"] == "success"
    assert data["session_id"] == "D-T301"


def test_legacy_fixture_has_no_json_block():
    """Legacy fixture must NOT contain a JSON code block."""
    import re
    text = (REPO / "tests/fixtures/session_exit_legacy.md").read_text()
    m = re.search(r"```json", text)
    assert m is None, "legacy fixture should not contain a JSON block"


def test_read_session_exit_extracts_json_block(tmp_path, monkeypatch):
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    fixture = (REPO / "tests/fixtures/session_exit_success.md").read_text()
    (tmp_path / "data/session_exit.md").write_text(fixture)

    se = pd.read_session_exit()
    assert se is not None
    assert se.get("status") == "success"
    assert se.get("session_id") == "D-T301"
    assert se.get("claimed_item_id") == "wq-test01"
    assert se.get("_legacy") is not True


def test_read_session_exit_returns_none_when_missing(tmp_path, monkeypatch):
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    # No data/session_exit.md written
    se = pd.read_session_exit()
    assert se is None


def test_read_session_exit_returns_legacy_marker_when_no_json_block(tmp_path, monkeypatch):
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    fixture = (REPO / "tests/fixtures/session_exit_legacy.md").read_text()
    (tmp_path / "data/session_exit.md").write_text(fixture)

    se = pd.read_session_exit()
    assert se is not None
    assert se.get("_legacy") is True


def test_read_session_exit_handles_invalid_json(tmp_path, monkeypatch):
    """If the JSON block is present but malformed, return _legacy + _invalid_json markers."""
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    bad = "```json\n{this is not valid json,\n```\n\nreason: GRACEFUL_CHECKPOINT\n"
    (tmp_path / "data/session_exit.md").write_text(bad)

    se = pd.read_session_exit()
    assert se is not None
    assert se.get("_legacy") is True
    assert se.get("_invalid_json") is True


def test_finalize_success_appends_log_and_completes_claim(tmp_path, monkeypatch):
    """status=success: log appended, current patched, claim completed, follow-on enqueued."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    # Set up an active claim
    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal", "program": "test",
        "created_by": "test_setup", "payload": {"to_phase": "P10"}
    })
    wq.claim(item_id, "director_session_d_t999")

    se = {
        "status": "success",
        "session_id": "D-T999",
        "claimed_item_id": item_id,
        "log_entry_text": "D-T999 (test): success entry\n\nSTATUS=success\n",
        "current_md_patches": [{"old": "Status: P9 ANALYZE", "new": "Status: P10 DONE"}],
        "next_action": {
            "type": "phase_advance", "priority": "normal", "program": "test",
            "created_by": "director_session_d_t999",
            "payload": {"to_phase": "P10_close"}
        },
    }
    pd.finalize_from_structured(se)

    log = (tmp_path / "data/memories/log.md").read_text()
    assert "D-T999" in log
    cur = (tmp_path / "data/memories/current.md").read_text()
    assert "P10 DONE" in cur
    assert "P9 ANALYZE" not in cur
    s = wq.summary()
    assert s["claimed"] == 0
    assert s["completed_today"] == 1
    assert s["pending"] == 1  # the next_action enqueue


def test_finalize_partial_completes_claim_but_no_followon(tmp_path, monkeypatch):
    """status=partial: same as success but next_action is suppressed/None."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal", "program": "test",
        "created_by": "test_setup", "payload": {}
    })
    wq.claim(item_id, "director_session_d_t900")

    se = {
        "status": "partial",
        "session_id": "D-T900",
        "claimed_item_id": item_id,
        "log_entry_text": "D-T900 (test): partial entry\n",
        "current_md_patches": [],
        "next_action": None,
        "program_complete": True,  # v1.1: partial without next_action requires program_complete
        "notes": "Partial work — needs follow-up next session.",
    }
    pd.finalize_from_structured(se)

    s = wq.summary()
    assert s["claimed"] == 0
    assert s["completed_today"] == 1
    assert s["pending"] == 0  # no follow-on


def test_apply_current_patch_skips_when_already_applied(tmp_path, monkeypatch):
    """If old not found AND new already present, patch is a no-op (idempotent)."""
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    cur = tmp_path / "data/memories/current.md"
    cur.write_text("# Current\n\nStatus: P10 DONE\n")  # already at target

    pd._apply_current_patch({"old": "Status: P9 ANALYZE", "new": "Status: P10 DONE"})
    # File should be unchanged
    assert "P10 DONE" in cur.read_text()
    assert "P9 ANALYZE" not in cur.read_text()


def test_append_log_idempotent_skips_when_session_id_present(tmp_path, monkeypatch):
    """Re-running finalize over the same session_exit doesn't double-append the log."""
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))

    pd._append_log_idempotent("D-T800 (test): first append\n", "D-T800")
    log_first = (tmp_path / "data/memories/log.md").read_text()
    assert "D-T800" in log_first

    # Run again with same session_id — should NOT double-append
    pd._append_log_idempotent("D-T800 (test): first append\n", "D-T800")
    log_second = (tmp_path / "data/memories/log.md").read_text()
    assert log_first == log_second
    assert log_second.count("D-T800") == 1


def test_finalize_failure_fails_claim_and_enqueues_diagnostic(tmp_path, monkeypatch):
    """status=failure: log appended, claim FAILED, diagnostic_review enqueued."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    item_id = wq.enqueue({
        "type": "verifier_review", "priority": "normal", "program": "test",
        "created_by": "test_setup", "payload": {}
    })
    wq.claim(item_id, "director_session_d_t405")

    se = {
        "status": "failure",
        "session_id": "D-T405",
        "claimed_item_id": item_id,
        "reason": "VERIFIER_REJECTION_LOOP",
        "log_entry_text": "D-T405 (test): VERIFIER_REJECTION_LOOP after 2 iterations\n",
        "current_md_patches": [],
        "next_action": None,
        "notes": "verifier rejected twice",
    }
    pd.finalize_from_structured(se)

    log = (tmp_path / "data/memories/log.md").read_text()
    assert "D-T405" in log
    s = wq.summary()
    assert s["claimed"] == 0
    assert s["failed_today"] == 1
    assert s["completed_today"] == 0
    # diagnostic_review item enqueued
    assert s["pending"] == 1
    assert s["by_type"].get("diagnostic_review") == 1


def test_finalize_no_op_no_queue_mutations(tmp_path, monkeypatch):
    """status=no_op: log unchanged (empty log_entry_text), no queue mutations."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    log_before = (tmp_path / "data/memories/log.md").read_text()
    cur_before = (tmp_path / "data/memories/current.md").read_text()

    se = {
        "status": "no_op",
        "session_id": "D-T320",
        "claimed_item_id": None,
        "log_entry_text": "",  # empty -> skip append
        "current_md_patches": [],
        "next_action": None,
        "notes": "heartbeat tick",
    }
    pd.finalize_from_structured(se)

    assert (tmp_path / "data/memories/log.md").read_text() == log_before
    assert (tmp_path / "data/memories/current.md").read_text() == cur_before
    s = wq.summary()
    assert s["pending"] == 0
    assert s["claimed"] == 0
    assert s["completed_today"] == 0
    assert s["failed_today"] == 0


def test_finalize_no_op_with_log_entry_appends_only(tmp_path, monkeypatch):
    """status=no_op with non-empty log_entry_text: log gets appended, but no queue ops."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    se = {
        "status": "no_op",
        "session_id": "D-T321",
        "claimed_item_id": None,
        "log_entry_text": "D-T321 (test): heartbeat with note\n",
        "current_md_patches": [],
        "next_action": None,
    }
    pd.finalize_from_structured(se)

    log = (tmp_path / "data/memories/log.md").read_text()
    assert "D-T321" in log
    s = wq.summary()
    assert s["pending"] == 0


def test_finalize_idempotent_on_rerun(tmp_path, monkeypatch):
    """Running finalize twice over the same session_exit produces same end state."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal", "program": "test",
        "created_by": "test_setup", "payload": {"to_phase": "P10"}
    })
    wq.claim(item_id, "director_session_d_t777")

    se = {
        "status": "success",
        "session_id": "D-T777",
        "claimed_item_id": item_id,
        "log_entry_text": "D-T777 (test): success entry\n",
        "current_md_patches": [{"old": "Status: P9 ANALYZE", "new": "Status: P10 DONE"}],
        "next_action": None,  # no follow-on so we can compare summaries cleanly
        "program_complete": True,  # v1.1: success without next_action requires program_complete
    }
    pd.finalize_from_structured(se)
    log_first = (tmp_path / "data/memories/log.md").read_text()
    cur_first = (tmp_path / "data/memories/current.md").read_text()
    s_first = wq.summary()

    # Run again — should be a no-op end-to-end
    pd.finalize_from_structured(se)
    log_second = (tmp_path / "data/memories/log.md").read_text()
    cur_second = (tmp_path / "data/memories/current.md").read_text()
    s_second = wq.summary()

    assert log_first == log_second
    assert cur_first == cur_second
    assert s_first["completed_today"] == s_second["completed_today"] == 1
    assert log_second.count("D-T777") == 1


def test_main_writes_telemetry_for_no_op(tmp_path, monkeypatch):
    """main() writes a telemetry record on every invocation, including no_op."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)
    fixture = (REPO / "tests/fixtures/session_exit_no_op.md").read_text()
    (tmp_path / "data/session_exit.md").write_text(fixture)

    rc = pd.main()
    assert rc == 0
    telem = tmp_path / "data/infra/post_director_telemetry.jsonl"
    assert telem.exists()
    rec = json.loads(telem.read_text().strip())
    assert rec["branch_taken"] == "structured_no_op"
    assert rec["session_id"] == "D-T320"


def test_main_writes_telemetry_for_success(tmp_path, monkeypatch):
    """main() routes success fixture to finalize_from_structured and logs telemetry."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    # Set up a claim matching the fixture's claimed_item_id
    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal", "program": "test",
        "created_by": "test_setup", "payload": {"to_phase": "P10"}
    })
    # Force the ID to match the fixture (fixture uses "wq-test01" — we override here)
    # Easiest: just claim whatever ID we got, then patch the fixture in tmp_path
    wq.claim(item_id, "director_session_d_t301")
    fixture = (REPO / "tests/fixtures/session_exit_success.md").read_text()
    fixture = fixture.replace("wq-test01", item_id)
    (tmp_path / "data/session_exit.md").write_text(fixture)

    rc = pd.main()
    assert rc == 0
    telem_lines = (tmp_path / "data/infra/post_director_telemetry.jsonl").read_text().splitlines()
    rec = json.loads(telem_lines[-1])
    assert rec["branch_taken"] == "structured_success"


def test_main_writes_telemetry_for_failure(tmp_path, monkeypatch):
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    item_id = wq.enqueue({
        "type": "verifier_review", "priority": "normal", "program": "test",
        "created_by": "test_setup", "payload": {}
    })
    wq.claim(item_id, "director_session_d_t405")
    fixture = (REPO / "tests/fixtures/session_exit_failure.md").read_text()
    fixture = fixture.replace("wq-testfail", item_id)
    (tmp_path / "data/session_exit.md").write_text(fixture)

    pd.main()
    rec = json.loads((tmp_path / "data/infra/post_director_telemetry.jsonl").read_text().strip())
    assert rec["branch_taken"] == "structured_failure"


def test_main_writes_telemetry_for_legacy(tmp_path, monkeypatch):
    """Legacy session_exit.md (no JSON block) routes to legacy_finalize and logs telemetry."""
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    fixture = (REPO / "tests/fixtures/session_exit_legacy.md").read_text()
    (tmp_path / "data/session_exit.md").write_text(fixture)

    pd.main()
    rec = json.loads((tmp_path / "data/infra/post_director_telemetry.jsonl").read_text().strip())
    assert rec["branch_taken"] == "legacy_finalize"


def test_main_writes_telemetry_for_silent_death_no_claim(tmp_path, monkeypatch):
    """No session_exit.md AND no active claim → silent_death_no_claim branch."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)
    # No data/session_exit.md and no claimed.jsonl
    pd.main()
    rec = json.loads((tmp_path / "data/infra/post_director_telemetry.jsonl").read_text().strip())
    assert rec["branch_taken"] in ("silent_death_no_claim", "silent_death")


def test_main_writes_telemetry_for_invalid_json(tmp_path, monkeypatch):
    """Invalid JSON in block → legacy fallback branch with _invalid_json flag noted."""
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    bad = "```json\n{this is not valid json,\n```\n\nreason: GRACEFUL_CHECKPOINT\n"
    (tmp_path / "data/session_exit.md").write_text(bad)

    pd.main()
    rec = json.loads((tmp_path / "data/infra/post_director_telemetry.jsonl").read_text().strip())
    assert rec["branch_taken"] == "legacy_finalize"
    assert rec.get("invalid_json") is True


def test_silent_death_with_deliverable_recovers(tmp_path, monkeypatch):
    """Director claimed phase_advance, produced deliverable, died without session_exit.md."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    # Copy real handlers schema to tmp repo
    src_handlers = REPO / "data/agents/_shared/work_queue_handlers.md"
    (tmp_path / "data/agents/_shared/work_queue_handlers.md").write_text(src_handlers.read_text())

    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal",
        "program": "test_program", "created_by": "test_setup",
        "payload": {"program": "test_program", "to_phase": "P10",
                    "from_phase": "P9", "context": "test"}
    })
    wq.claim(item_id, "director_session_d_t999")

    # Create the expected deliverable matching pattern programs/{program}/phase{to_phase_num}_*.md
    deliv = tmp_path / "programs/test_program/phase10_mechanism_report.md"
    deliv.parent.mkdir(parents=True)
    deliv.write_text("# Mechanism report\n\n" + "x" * 1000)  # > 100 bytes

    # Make sure deliverable mtime is > claim mtime
    import time
    time.sleep(0.05)
    deliv.touch()

    # No session_exit.md — silent death
    pd.recover_from_silent_death()

    s = wq.summary()
    assert s["claimed"] == 0, f"expected claim recovered, summary={s}"
    assert s["completed_today"] >= 1
    log = (tmp_path / "data/memories/log.md").read_text()
    assert "RECOVERY" in log or "silent" in log.lower()
    # Should also enqueue session_recovery_review and a follow-on phase_advance
    pending_types = []
    pending_path = tmp_path / "data/work_queue/pending.jsonl"
    if pending_path.exists():
        import json as _j
        for line in pending_path.read_text().splitlines():
            if line.strip():
                pending_types.append(_j.loads(line)["type"])
    assert "session_recovery_review" in pending_types
    assert "phase_advance" in pending_types  # follow-on from next_action_template


def test_silent_death_without_deliverable_fails(tmp_path, monkeypatch):
    """Claim active, no matching deliverable on disk → fail claim + diagnostic_review enqueue."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    src_handlers = REPO / "data/agents/_shared/work_queue_handlers.md"
    (tmp_path / "data/agents/_shared/work_queue_handlers.md").write_text(src_handlers.read_text())

    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal",
        "program": "test_program", "created_by": "test_setup",
        "payload": {"program": "test_program", "to_phase": "P10"}
    })
    wq.claim(item_id, "director_session_d_t000")

    # NO deliverable created — recover should fail this claim
    pd.recover_from_silent_death()

    s = wq.summary()
    assert s["claimed"] == 0, f"expected claim resolved, summary={s}"
    assert s["failed_today"] >= 1
    pending_types = []
    pending_path = tmp_path / "data/work_queue/pending.jsonl"
    if pending_path.exists():
        import json as _j
        for line in pending_path.read_text().splitlines():
            if line.strip():
                pending_types.append(_j.loads(line)["type"])
    assert "diagnostic_review" in pending_types


def test_silent_death_handler_opt_out_fails_claim(tmp_path, monkeypatch):
    """Item type with null expected_deliverable_pattern should fail-fast (no glob attempt)."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    src_handlers = REPO / "data/agents/_shared/work_queue_handlers.md"
    (tmp_path / "data/agents/_shared/work_queue_handlers.md").write_text(src_handlers.read_text())

    item_id = wq.enqueue({
        "type": "cell_failed",  # opted out (null pattern)
        "priority": "urgent",
        "program": "test", "created_by": "test_setup",
        "payload": {"cell_id": "A42"}
    })
    wq.claim(item_id, "director_session_d_t111")

    pd.recover_from_silent_death()

    s = wq.summary()
    assert s["failed_today"] >= 1


def test_silent_death_recovery_idempotent(tmp_path, monkeypatch):
    """Running recovery twice over same state is a no-op the second time."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    src_handlers = REPO / "data/agents/_shared/work_queue_handlers.md"
    (tmp_path / "data/agents/_shared/work_queue_handlers.md").write_text(src_handlers.read_text())

    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal",
        "program": "test_program", "created_by": "test_setup",
        "payload": {"program": "test_program", "to_phase": "P10"}
    })
    wq.claim(item_id, "director_session_d_t222")
    deliv = tmp_path / "programs/test_program/phase10_mechanism_report.md"
    deliv.parent.mkdir(parents=True)
    deliv.write_text("# x\n" + "y" * 200)
    import time
    time.sleep(0.05)
    deliv.touch()

    pd.recover_from_silent_death()
    s_first = wq.summary()
    log_first = (tmp_path / "data/memories/log.md").read_text()

    # Run again — claimed.jsonl is now empty, so this should be no-op
    pd.recover_from_silent_death()
    s_second = wq.summary()
    log_second = (tmp_path / "data/memories/log.md").read_text()

    assert s_first == s_second
    assert log_first == log_second


def test_post_director_invokes_cost_rollup_on_success(tmp_path, monkeypatch):
    """After main() runs the success branch, cost_rollup.py should be invoked."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)
    # Stub out cost_rollup.py so the exists() check passes
    (tmp_path / "tools").mkdir(exist_ok=True)
    (tmp_path / "tools/cost_rollup.py").write_text("# stub\n")
    fixture = (REPO / "tests/fixtures/session_exit_no_op.md").read_text()
    (tmp_path / "data/session_exit.md").write_text(fixture)

    # Record subprocess.run calls
    calls = []
    def fake_run(cmd, *args, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        # Return a successful CompletedProcess-like object
        class _CP:
            returncode = 0
            stdout = b""
            stderr = b""
        return _CP()

    monkeypatch.setattr("subprocess.run", fake_run)
    pd.main()

    # At least one subprocess call should be tools/cost_rollup.py
    cost_calls = [c for c in calls if any("cost_rollup" in str(x) for x in c["cmd"])]
    assert len(cost_calls) >= 1, f"expected cost_rollup invocation, got calls: {calls}"


def test_post_director_cost_rollup_is_non_fatal(tmp_path, monkeypatch):
    """Even if cost_rollup subprocess raises, post_director.main returns 0."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)
    # Stub out cost_rollup.py so the exists() check passes
    (tmp_path / "tools").mkdir(exist_ok=True)
    (tmp_path / "tools/cost_rollup.py").write_text("# stub\n")
    fixture = (REPO / "tests/fixtures/session_exit_no_op.md").read_text()
    (tmp_path / "data/session_exit.md").write_text(fixture)

    def raising_run(*args, **kwargs):
        raise OSError("subprocess simulator failure")

    monkeypatch.setattr("subprocess.run", raising_run)
    rc = pd.main()
    # main() must NOT propagate the subprocess failure
    assert rc == 0


def test_post_director_cost_rollup_fires_after_silent_death(tmp_path, monkeypatch):
    """cost_rollup should fire regardless of branch — silent_death too."""
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    # Stub out cost_rollup.py so the exists() check passes
    (tmp_path / "tools").mkdir(exist_ok=True)
    (tmp_path / "tools/cost_rollup.py").write_text("# stub\n")
    # No session_exit.md → silent_death branch

    calls = []
    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        class _CP:
            returncode = 0
            stdout = b""
            stderr = b""
        return _CP()
    monkeypatch.setattr("subprocess.run", fake_run)
    pd.main()
    cost_calls = [c for c in calls if any("cost_rollup" in str(x) for x in c)]
    assert len(cost_calls) >= 1


# === L1.2: schema validation ===

def test_validate_session_exit_schema_valid_success_with_next_action():
    from tools.post_director import _validate_session_exit_schema
    se = {"status": "success", "next_action": {"type": "phase_advance"}}
    ok, reason = _validate_session_exit_schema(se)
    assert ok is True
    assert reason is None


def test_validate_session_exit_schema_valid_program_complete():
    from tools.post_director import _validate_session_exit_schema
    se = {"status": "success", "next_action": None, "program_complete": True}
    ok, reason = _validate_session_exit_schema(se)
    assert ok is True


def test_validate_session_exit_schema_valid_no_op_null_next_action():
    """no_op with null next_action is valid (heartbeat case)."""
    from tools.post_director import _validate_session_exit_schema
    se = {"status": "no_op", "next_action": None}
    ok, reason = _validate_session_exit_schema(se)
    assert ok is True


def test_validate_session_exit_schema_valid_failure_null_next_action():
    """failure with null next_action is valid (post_director enqueues diagnostic_review)."""
    from tools.post_director import _validate_session_exit_schema
    se = {"status": "failure", "next_action": None}
    ok, reason = _validate_session_exit_schema(se)
    assert ok is True


def test_validate_session_exit_schema_violation_success_without_next_action():
    """status=success + next_action=None + program_complete missing -> violation."""
    from tools.post_director import _validate_session_exit_schema
    se = {"status": "success", "next_action": None}
    ok, reason = _validate_session_exit_schema(se)
    assert ok is False
    assert "next_action" in reason or "program_complete" in reason


def test_validate_session_exit_schema_violation_partial_without_next_action():
    """status=partial behaves same way as success for v1.1."""
    from tools.post_director import _validate_session_exit_schema
    se = {"status": "partial", "next_action": None, "program_complete": False}
    ok, reason = _validate_session_exit_schema(se)
    assert ok is False


def test_validate_session_exit_schema_invalid_status():
    from tools.post_director import _validate_session_exit_schema
    se = {"status": "weird_status"}
    ok, reason = _validate_session_exit_schema(se)
    assert ok is False
    assert "invalid_status" in reason or "weird_status" in reason


def test_finalize_returns_violation_marker_when_invalid(tmp_path, monkeypatch):
    """Invalid session_exit short-circuits -- no mutations applied."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    # Snapshot log + current pre-finalize
    log_before = (tmp_path / "data/memories/log.md").read_text()
    cur_before = (tmp_path / "data/memories/current.md").read_text()

    se = {
        "status": "success",
        "session_id": "D-VIOLATION",
        "claimed_item_id": "wq-x",
        "log_entry_text": "should not be appended\n",
        "current_md_patches": [{"old": "P9 ANALYZE", "new": "VIOLATED"}],
        "next_action": None,
        # program_complete absent
    }
    result = pd.finalize_from_structured(se)
    # finalize returns a violation marker (not None for valid, but a dict here)
    assert isinstance(result, dict)
    assert result.get("violation") is True
    assert result.get("violation_type") == "schema"

    # No mutations applied
    assert (tmp_path / "data/memories/log.md").read_text() == log_before
    assert (tmp_path / "data/memories/current.md").read_text() == cur_before


def test_main_writes_schema_violation_telemetry_and_flag(tmp_path, monkeypatch):
    """main() with invalid session_exit writes branch_taken=schema_violation + touches redispatch flag."""
    import tools.post_director as pd
    import json as _j
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))

    # Write invalid session_exit.md
    invalid_se = {
        "status": "success",
        "session_id": "D-V001",
        "claimed_item_id": "wq-vi",
        "log_entry_text": "ignored\n",
        "current_md_patches": [],
        "next_action": None,
    }
    (tmp_path / "data/session_exit.md").write_text(
        f"```json\n{_j.dumps(invalid_se)}\n```\n\nreason: GRACEFUL_CHECKPOINT\n"
    )

    pd.main()

    # Telemetry: last record is schema_violation
    telem_lines = (tmp_path / "data/infra/post_director_telemetry.jsonl").read_text().strip().split("\n")
    last = _j.loads(telem_lines[-1])
    assert last["branch_taken"] == "schema_violation"
    assert "violation_reason" in last, f"expected violation_reason in telemetry, got: {last}"

    # Redispatch flag file present
    flag = tmp_path / "data/session_exit_redispatch_pending.flag"
    assert flag.exists(), "expected redispatch flag file"

    # Verify flag CONTENT — runner depends on grep '^reason=' / 'session_id=' / 'violation_type='
    flag_content = flag.read_text()
    assert "violation_type=" in flag_content, f"flag missing violation_type line: {flag_content!r}"
    assert "reason=" in flag_content, f"flag missing reason line: {flag_content!r}"
    assert "session_id=" in flag_content, f"flag missing session_id line: {flag_content!r}"

    # Verify the specific values match what was written
    assert "session_id=D-V001" in flag_content
    assert "violation_type=schema" in flag_content


def test_main_valid_success_no_flag_written(tmp_path, monkeypatch):
    """Valid session_exit does NOT write the redispatch flag."""
    import tools.post_director as pd
    import tools.work_queue as wq
    import json as _j
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    # Set up a claim
    item_id = wq.enqueue({"type": "phase_advance", "priority": "normal",
                          "program": "test", "created_by": "t", "payload": {}})
    wq.claim(item_id, "director_session_d_v002")

    valid_se = {
        "status": "success",
        "session_id": "D-V002",
        "claimed_item_id": item_id,
        "log_entry_text": "D-V002: ok\n",
        "current_md_patches": [],
        "next_action": {"type": "phase_advance", "priority": "normal",
                        "program": "test", "created_by": "director_session_d_v002",
                        "payload": {"to_phase": "P10"}},
    }
    (tmp_path / "data/session_exit.md").write_text(
        f"```json\n{_j.dumps(valid_se)}\n```\n\nreason: GRACEFUL_CHECKPOINT\n"
    )

    pd.main()

    flag = tmp_path / "data/session_exit_redispatch_pending.flag"
    assert not flag.exists(), "valid session_exit should NOT write redispatch flag"


# === L3.2: Agent contract enforcement ===

def test_check_contract_returns_valid_when_session_exit_complete(tmp_path, monkeypatch):
    """Valid session_exit + file present → contract passes."""
    import tools.post_director as pd
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    # Copy real contracts file
    import shutil
    (repo / "data/agents/_shared").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "data/agents/_shared/agent_contracts.json",
                repo / "data/agents/_shared/agent_contracts.json")
    # Write a valid session_exit.md so file_exists check passes
    (repo / "data/session_exit.md").write_text("ok\n")

    se = {
        "status": "success",
        "session_id": "D-CTR-1",
        "claimed_item_id": "wq-x",
        "next_action": {"type": "phase_advance", "priority": "normal"},
    }
    ok, reason = pd._check_contract(se, role="director")
    assert ok is True
    assert reason is None


def test_check_contract_violation_when_session_exit_missing(tmp_path, monkeypatch):
    """Contract check `file_exists` for data/session_exit.md fails when file absent."""
    import tools.post_director as pd
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    import shutil
    (repo / "data/agents/_shared").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "data/agents/_shared/agent_contracts.json",
                repo / "data/agents/_shared/agent_contracts.json")
    # Do NOT write data/session_exit.md

    se = {
        "status": "success",
        "session_id": "D-CTR-2",
        "claimed_item_id": "wq-x",
        "next_action": {"type": "phase_advance"},
    }
    ok, reason = pd._check_contract(se, role="director")
    assert ok is False
    assert "session_exit" in reason or "file_exists" in reason


def test_check_contract_violation_when_next_action_missing(tmp_path, monkeypatch):
    """Contract `next_action_populated_or_program_complete_true` violated."""
    import tools.post_director as pd
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    import shutil
    (repo / "data/agents/_shared").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "data/agents/_shared/agent_contracts.json",
                repo / "data/agents/_shared/agent_contracts.json")
    (repo / "data/session_exit.md").write_text("ok\n")

    se = {
        "status": "success",
        "session_id": "D-CTR-3",
        "claimed_item_id": "wq-x",
        "next_action": None,
        # program_complete not set
    }
    ok, reason = pd._check_contract(se, role="director")
    assert ok is False


def test_check_contract_program_complete_satisfies(tmp_path, monkeypatch):
    """program_complete=True satisfies the next_action requirement."""
    import tools.post_director as pd
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    import shutil
    (repo / "data/agents/_shared").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "data/agents/_shared/agent_contracts.json",
                repo / "data/agents/_shared/agent_contracts.json")
    (repo / "data/session_exit.md").write_text("ok\n")

    se = {
        "status": "success",
        "session_id": "D-CTR-4",
        "claimed_item_id": "wq-x",
        "next_action": None,
        "program_complete": True,
    }
    ok, reason = pd._check_contract(se, role="director")
    assert ok is True


def test_check_contract_missing_contracts_file_passes(tmp_path, monkeypatch):
    """If contracts file is absent, no enforcement (backward-compat)."""
    import tools.post_director as pd
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    # Do NOT copy contracts file → directory doesn't have it

    se = {"status": "success", "next_action": None}
    ok, reason = pd._check_contract(se, role="director")
    assert ok is True
    assert reason is None


def test_finalize_returns_contract_violation_when_next_action_missing(tmp_path, monkeypatch):
    """finalize_from_structured short-circuits on contract violation too (after schema check)."""
    import tools.post_director as pd
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    import shutil
    (repo / "data/agents/_shared").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "data/agents/_shared/agent_contracts.json",
                repo / "data/agents/_shared/agent_contracts.json")
    (repo / "data/session_exit.md").write_text("ok\n")

    # Schema would FAIL first (next_action=None without program_complete);
    # but we want to verify contract violation_type is distinct.
    # Use a session that PASSES schema but FAILS contract:
    # Schema accepts: status=no_op with next_action=None.
    # But contract also has next_action_populated_or_program_complete_true.
    # Need to test the case where schema passes but contract still cares.
    # Actually contract's next_action_populated_or_program_complete_true check
    # is ONLY applied for success/partial — same rule as schema.
    # So a session passing schema (no_op) also passes contract.
    # The case where schema PASSES but contract FAILS is:
    #   - status=success
    #   - next_action populated
    #   - BUT data/session_exit.md doesn't exist (contract's file_exists check)

    se = {
        "status": "success",
        "session_id": "D-CTR-FIN-1",
        "claimed_item_id": "wq-fin1",
        "next_action": {"type": "phase_advance", "priority": "normal",
                        "program": "test", "created_by": "t", "payload": {}},
    }
    # Remove session_exit.md so contract file_exists check fails
    (repo / "data/session_exit.md").unlink()

    result = pd.finalize_from_structured(se)
    assert isinstance(result, dict)
    assert result.get("violation") is True
    assert result.get("violation_type") == "contract"


def test_main_writes_contract_violation_telemetry(tmp_path, monkeypatch):
    """main() with contract violation writes branch_taken=contract_violation."""
    import tools.post_director as pd
    import json as _j
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    import shutil
    (repo / "data/agents/_shared").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "data/agents/_shared/agent_contracts.json",
                repo / "data/agents/_shared/agent_contracts.json")

    # Schema-valid + contract-INVALID requires: status=success, next_action populated,
    # but contract's file_exists check on data/session_exit.md.
    # But the runner writes session_exit.md BEFORE post_director runs, so in main()'s
    # actual execution, session_exit.md DOES exist (it's how we read se in the first place).
    #
    # The only way contract can fail when schema passes: the contracts JSON adds a check
    # that schema doesn't (e.g., max_tokens, which we don't enforce yet in v1).
    #
    # For v1, schema check is essentially a strict subset of contract checks. So
    # ANY contract violation also fails schema. We test the OVERLAPPING case below
    # for the main() branch routing.

    invalid_se = {
        "status": "success",
        "session_id": "D-CTR-MAIN-1",
        "claimed_item_id": "wq-cm1",
        "next_action": None,
        # No program_complete — fails both schema AND contract
    }
    (repo / "data/session_exit.md").write_text(
        f"```json\n{_j.dumps(invalid_se)}\n```\n\nreason: GRACEFUL_CHECKPOINT\n"
    )

    pd.main()
    telem_lines = (repo / "data/infra/post_director_telemetry.jsonl").read_text().strip().split("\n")
    last = _j.loads(telem_lines[-1])
    # branch is either schema_violation OR contract_violation — both are valid
    assert last["branch_taken"] in ("schema_violation", "contract_violation"), \
        f"got {last['branch_taken']!r}"


# === L3.3: Redispatch ceiling + operator escalation ===

def _setup_contracts(repo):
    """Helper: copy contracts file into a tmp repo."""
    import shutil
    (repo / "data/agents/_shared").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "data/agents/_shared/agent_contracts.json",
                repo / "data/agents/_shared/agent_contracts.json")


def test_first_violation_writes_flag_and_count_1(tmp_path, monkeypatch):
    """First violation for a session_id: count=1, flag written."""
    import tools.post_director as pd
    import json as _j
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    _setup_contracts(repo)

    invalid_se = {"status": "success", "session_id": "D-CEIL-1",
                  "claimed_item_id": "wq-c1", "next_action": None}
    (repo / "data/session_exit.md").write_text(
        f"```json\n{_j.dumps(invalid_se)}\n```\n\nreason: GRACEFUL_CHECKPOINT\n"
    )
    pd.main()

    flag = repo / "data/session_exit_redispatch_pending.flag"
    count_path = repo / "data/infra/director_redispatch_count.json"
    assert flag.exists()
    assert count_path.exists()
    count_data = _j.loads(count_path.read_text())
    assert count_data["count"] == 1
    assert count_data["current_session_id"] == "D-CEIL-1"


def test_second_violation_same_session_increments_to_2(tmp_path, monkeypatch):
    """Second violation for SAME session_id: count=2, flag still written."""
    import tools.post_director as pd
    import json as _j
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    _setup_contracts(repo)

    invalid_se = {"status": "success", "session_id": "D-CEIL-2",
                  "claimed_item_id": "wq-c2", "next_action": None}
    se_text = f"```json\n{_j.dumps(invalid_se)}\n```\n\nreason: GRACEFUL_CHECKPOINT\n"

    (repo / "data/session_exit.md").write_text(se_text)
    pd.main()
    (repo / "data/session_exit.md").write_text(se_text)  # rewrite (runner deleted between)
    pd.main()

    count = _j.loads((repo / "data/infra/director_redispatch_count.json").read_text())
    assert count["count"] == 2


def test_violation_new_session_resets_count(tmp_path, monkeypatch):
    """Different session_id resets count to 1."""
    import tools.post_director as pd
    import json as _j
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    _setup_contracts(repo)

    se1_text = f"```json\n{_j.dumps({'status':'success','session_id':'D-CEIL-A','claimed_item_id':'wq-a','next_action':None})}\n```\n\nreason: x\n"
    se2_text = f"```json\n{_j.dumps({'status':'success','session_id':'D-CEIL-B','claimed_item_id':'wq-b','next_action':None})}\n```\n\nreason: x\n"

    (repo / "data/session_exit.md").write_text(se1_text)
    pd.main()
    (repo / "data/session_exit.md").write_text(se2_text)
    pd.main()

    count = _j.loads((repo / "data/infra/director_redispatch_count.json").read_text())
    assert count["current_session_id"] == "D-CEIL-B"
    assert count["count"] == 1


def test_third_violation_triggers_escalation(tmp_path, monkeypatch):
    """3rd violation for same session_id: escalate, no flag written, operator review file created."""
    import tools.post_director as pd
    import json as _j
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    _setup_contracts(repo)

    invalid_se = {"status": "success", "session_id": "D-CEIL-3",
                  "claimed_item_id": "wq-c3", "next_action": None}
    se_text = f"```json\n{_j.dumps(invalid_se)}\n```\n\nreason: GRACEFUL_CHECKPOINT\n"

    for i in range(3):
        (repo / "data/session_exit.md").write_text(se_text)
        pd.main()

    # After 3rd: operator review file exists, flag is NOT written (or removed)
    operator_review = repo / "data/operator_review_pending.md"
    assert operator_review.exists()
    review_text = operator_review.read_text()
    assert "D-CEIL-3" in review_text  # session_id in the review
    assert "3" in review_text  # count

    # Flag should NOT be present (3rd violation means STOP redispatching)
    flag = repo / "data/session_exit_redispatch_pending.flag"
    assert not flag.exists() or flag.read_text() == "", \
        "redispatch flag should NOT be set after escalation"


def test_escalation_telemetry_distinguishes_branch(tmp_path, monkeypatch):
    """When escalation fires, branch_taken indicates max_redispatches reached."""
    import tools.post_director as pd
    import json as _j
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    _setup_contracts(repo)

    invalid_se = {"status": "success", "session_id": "D-CEIL-T",
                  "claimed_item_id": "wq-ct", "next_action": None}
    se_text = f"```json\n{_j.dumps(invalid_se)}\n```\n\nreason: x\n"

    for i in range(3):
        (repo / "data/session_exit.md").write_text(se_text)
        pd.main()

    telem_lines = (repo / "data/infra/post_director_telemetry.jsonl").read_text().strip().split("\n")
    last = _j.loads(telem_lines[-1])
    # Should be an escalation branch
    assert "escalat" in last["branch_taken"].lower() or "max_redispatch" in last["branch_taken"].lower(), \
        f"expected escalation branch_taken, got {last['branch_taken']!r}"


def test_count_file_corrupt_is_handled_gracefully(tmp_path, monkeypatch):
    """Corrupt count file → treat as count=0 (don't crash)."""
    import tools.post_director as pd
    import json as _j
    repo = _setup_repo(tmp_path)
    monkeypatch.setattr(pd, "REPO", repo)
    _setup_contracts(repo)

    (repo / "data/infra").mkdir(parents=True, exist_ok=True)
    (repo / "data/infra/director_redispatch_count.json").write_text("THIS IS NOT JSON")

    invalid_se = {"status": "success", "session_id": "D-CORRUPT",
                  "claimed_item_id": "wq-x", "next_action": None}
    (repo / "data/session_exit.md").write_text(
        f"```json\n{_j.dumps(invalid_se)}\n```\n\nreason: x\n"
    )
    # Should not raise
    pd.main()
    # Count should now be 1 (reset from corrupt)
    count = _j.loads((repo / "data/infra/director_redispatch_count.json").read_text())
    assert count["count"] == 1
