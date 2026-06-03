from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.brief_assembler import assemble_brief, classify_session


def test_assemble_brief_returns_markdown_with_required_sections(tmp_path):
    # Set up a minimal fake lab state in tmp_path
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text(
        "D-100: minimal entry / STATUS=success / SUMMARY=test\n"
    )
    (tmp_path / "data/memories/current.md").write_text("# Current\nactive_program: test_prog\n")
    (tmp_path / "data/checkpoints/phase3_factorial").mkdir(parents=True)
    (tmp_path / "data/checkpoints/phase3_factorial/run_index.json").write_text(
        '{"_meta": {}, "A42": {"state": "completed"}}'
    )

    brief = assemble_brief(repo_root=tmp_path)

    assert brief.startswith("---\ngenerated_at:")
    assert "session_type:" in brief
    assert "# Active program" in brief
    assert "# Last 5 decisions" in brief
    assert "# Active carry-forwards" in brief
    assert "# Decision-critical files" in brief


def test_classify_phase_transition_when_orchestrator_dead_and_run_index_newer(tmp_path):
    # log.md older than run_index.json AND no orchestrator process
    log_path = tmp_path / "data/memories/log.md"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("D-100: old\n")
    run_idx = tmp_path / "data/checkpoints/phase3_factorial/run_index.json"
    run_idx.parent.mkdir(parents=True)
    run_idx.write_text("{}")
    # Make run_idx newer than log.md
    import os, time
    old = time.time() - 86400
    os.utime(log_path, (old, old))

    assert classify_session(tmp_path, _orchestrator_alive=False) == "phase-transition"


def test_classify_user_action_when_user_notes_recent(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("x\n")
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data/user_notes.md").write_text("DIRECTIVE\n")
    # user_notes is fresh by default

    assert classify_session(tmp_path, _orchestrator_alive=True) == "user-action"


def test_classify_post_failure_when_session_exit_says_so(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("x\n")
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data/session_exit.md").write_text("reason: RATE_LIMIT\n")

    assert classify_session(tmp_path, _orchestrator_alive=True) == "post-failure"


def test_classify_routine_monitor_default(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("x\n")

    assert classify_session(tmp_path, _orchestrator_alive=True) == "routine-monitor"


def test_active_program_block_extracted_from_current_md(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("x\n")
    (tmp_path / "data/memories/current.md").write_text(
        "# Current\n"
        "## §Current Program\n"
        "active_program: program_2_dense_vs_moe_sub100m\n"
        "phase: 3\n"
        "status_line: B42 step 400/5000 plateau-center\n"
        "## Some other section\n"
        "irrelevant content\n"
    )
    brief = assemble_brief(tmp_path)
    assert "program_2_dense_vs_moe_sub100m" in brief
    assert "phase: 3" in brief
    assert "B42 step 400/5000 plateau-center" in brief


def test_last_5_decisions_extracted_from_log_md(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text(
        "D-307: latest / STATUS=success / KEY_FINDING=X / SUMMARY=lorem\n"
        "\n"
        "D-306: prior / STATUS=success / KEY_FINDING=Y / SUMMARY=ipsum\n"
        "\n"
        "D-305: more / STATUS=success / KEY_FINDING=Z / SUMMARY=dolor\n"
        "\n"
        "D-304: even / STATUS=success / KEY_FINDING=W / SUMMARY=sit\n"
        "\n"
        "D-303: oldest in window / STATUS=success / KEY_FINDING=V / SUMMARY=amet\n"
        "\n"
        "D-302: should NOT appear / SUMMARY=outside-window\n"
    )
    brief = assemble_brief(tmp_path)
    for did in ("D-307", "D-306", "D-305", "D-304", "D-303"):
        assert did in brief
    assert "D-302" not in brief


def test_state_delta_lists_run_index_changes(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("D-100: x\n")
    (tmp_path / "data/checkpoints/phase3_factorial").mkdir(parents=True)
    (tmp_path / "data/checkpoints/phase3_factorial/run_index.json").write_text(
        json.dumps({
            "_meta": {},
            "A42": {"state": "completed", "elapsed_hours": 12.5},
            "B42": {"state": "in_progress", "run_index": 3},
        })
    )
    brief = assemble_brief(tmp_path)
    assert "A42" in brief
    assert "B42" in brief
    assert "completed" in brief
    assert "in_progress" in brief


def test_last_5_decisions_handles_real_parenthetical_timestamp_format(tmp_path):
    """Real log.md uses 'D-NNN (timestamp): content', not 'D-NNN: content'."""
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text(
        "D-308 (2026-05-04, ~22:05 UTC): Phase 3 P8 EXECUTE COMPLETE\n"
        "\n"
        "D-307 (2026-04-29, ~14:12 UTC): Substantive 30-min LW anchor\n"
        "\n"
        "D-306 (2026-04-29, ~10:05 UTC): D-261 ESCALATION CRITERION NON-FIRE\n"
    )
    brief = assemble_brief(tmp_path)
    assert "D-308" in brief
    assert "D-307" in brief
    assert "D-306" in brief


def test_carry_forwards_extracted_from_log_md(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text(
        "D-200: P-FOO-BAR carry-forward opened — needs follow-up\n"
        "D-199: P-BAZ-QUX queued for next phase\n"
        "D-150: P-OLD resolved at D-180\n"
    )
    brief = assemble_brief(tmp_path)
    # Check that carry-forwards section appears and contains the tokens
    assert "# Active carry-forwards" in brief
    carry_section = brief.split("# Active carry-forwards")[1].split("#")[0]
    assert "P-FOO-BAR" in carry_section
    assert "P-BAZ-QUX" in carry_section


def test_routine_monitor_session_includes_brief_action_hint(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("D-1: x\n")
    brief = assemble_brief(tmp_path)
    # Check that action hint section exists and contains routine-monitor guidance
    assert "# What this session likely needs to do" in brief
    action_section = brief.split("# What this session likely needs to do")[1].split("#")[0]
    # Should contain guidance about monitoring/ticking for routine-monitor type
    assert "monitor" in action_section.lower() or "tick" in action_section.lower()


def test_decision_critical_files_for_phase_transition(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("D-1: x\n")
    (tmp_path / "data/memories/current.md").write_text(
        "active_program: program_2_dense_vs_moe_sub100m\n"
    )
    (tmp_path / "programs/program_2_dense_vs_moe_sub100m").mkdir(parents=True)
    (tmp_path / "programs/program_2_dense_vs_moe_sub100m/phase3_close_carry_forwards.md").write_text("x")
    (tmp_path / "programs/program_2_dense_vs_moe_sub100m/spec.md").write_text("x")
    # Force phase-transition
    (tmp_path / "data/checkpoints/phase3_factorial").mkdir(parents=True)
    idx = tmp_path / "data/checkpoints/phase3_factorial/run_index.json"
    idx.write_text("{}")
    log = tmp_path / "data/memories/log.md"
    import os, time
    old = time.time() - 86400
    os.utime(log, (old, old))

    brief = assemble_brief(tmp_path, _orchestrator_alive=False)
    assert "phase3_close_carry_forwards.md" in brief
    assert "spec.md" in brief


def test_brief_size_under_30kb(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    # Big log to test size cap
    big_log = "\n".join(f"D-{i}: lorem ipsum dolor " * 20 for i in range(500))
    (tmp_path / "data/memories/log.md").write_text(big_log)
    brief = assemble_brief(tmp_path)
    assert len(brief.encode("utf-8")) < 30 * 1024


def test_wiki_not_loaded_includes_agent_semantic_memories(tmp_path):
    """Director should know its semantic.md is reachable on-demand."""
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("D-1: x\n")
    (tmp_path / "data/agents/director").mkdir(parents=True)
    (tmp_path / "data/agents/director/semantic.md").write_text("# patterns\n")
    (tmp_path / "data/agents/pi").mkdir(parents=True)
    (tmp_path / "data/agents/pi/semantic.md").write_text("# strategy\n")

    brief = assemble_brief(tmp_path)
    assert "data/agents/director/semantic.md" in brief
    assert "data/agents/pi/semantic.md" in brief


def test_brief_assembler_injects_correction_prompt_on_redispatch(tmp_path, monkeypatch):
    """When DIRECTOR_REDISPATCH_REASON env var is set, brief_assembler prepends a CORRECTION PROMPT block."""
    import importlib
    import tools.brief_assembler as ba
    importlib.reload(ba)
    # Set up minimal repo
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/current.md").write_text("# Current\n\nactive_program: test\n")
    (tmp_path / "data/memories/log.md").write_text("# Log\n\n---\n\n")

    monkeypatch.setenv("DIRECTOR_REDISPATCH_REASON", "schema_violation_previous_session_exit")
    monkeypatch.setattr(ba, "REPO", tmp_path)

    brief = ba.assemble_brief(tmp_path)
    # Brief should contain a CORRECTION PROMPT section at the top
    assert "CORRECTION PROMPT" in brief or "REDISPATCH" in brief.upper(), \
        f"brief should include correction prompt; got: {brief[:500]}"
    # And reference the schema_violation reason
    assert "schema" in brief.lower() or "next_action" in brief.lower()


def test_brief_assembler_no_correction_when_env_unset(tmp_path, monkeypatch):
    import importlib
    import tools.brief_assembler as ba
    importlib.reload(ba)
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/current.md").write_text("# Current\n\n")
    (tmp_path / "data/memories/log.md").write_text("# Log\n\n---\n\n")

    monkeypatch.delenv("DIRECTOR_REDISPATCH_REASON", raising=False)
    monkeypatch.setattr(ba, "REPO", tmp_path)

    brief = ba.assemble_brief(tmp_path)
    assert "CORRECTION PROMPT" not in brief.upper()


def test_last_decision_matches_log_md_head(tmp_path, monkeypatch):
    """The `Last decision` field in the brief must reflect log.md head,
    not a cached/stale snapshot. Regression test for P-D427."""
    repo = tmp_path / "repo"
    (repo / "data/memories").mkdir(parents=True)

    log_md = repo / "data/memories/log.md"
    log_md.write_text(
        "Log tier — recent decisions.\n\n---\n\n"
        "### D-999 (2026-05-20, ~14:00 UTC): **FRESHEST DECISION — should appear in brief.**\n"
        "Lorem ipsum.\n\n"
        "### D-998 (2026-05-20, ~13:00 UTC): **Older decision.**\n"
    )
    (repo / "data/memories/current.md").write_text(
        "## Current Program: foo\nCurrent Phase: bar\n"
    )

    from tools.brief_assembler import _extract_last_n_decisions
    out = _extract_last_n_decisions(repo, n=2)
    assert "D-999" in out, f"expected D-999 in last-decisions, got: {out!r}"
    assert "FRESHEST" in out, f"expected newest decision body, got: {out!r}"

    # Now write the full brief and check the rendered Last decision line
    import tools.brief_assembler as ba
    brief_text = ba.assemble_brief(repo)
    last_dec_line = next(
        (l for l in brief_text.splitlines() if l.startswith("Last decision:") or "D-999" in l),
        None,
    )
    assert last_dec_line is not None, f"no Last decision line in brief; brief was: {brief_text[:500]!r}"
    assert "D-999" in last_dec_line, f"Last decision line stale; got: {last_dec_line!r}"
