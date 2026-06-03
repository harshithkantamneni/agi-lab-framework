"""Tests for tools/slack_bot.py — 3 D-216 detectors added 2026-04-25.

Coverage scope (per D-217 dispatch):
  Validates the 3 detector functions added in commit 88918bd:
    1. detect_user_action_required (line 710)  — USER_*.md GO/NOGO surfacing
    2. detect_holding_loop          (line 815) — D-202..D-210 incident motivator
    3. detect_phase_boundary        (line 889) — phase transition surfacing

Discipline (HARD):
  - These tests MUST NOT post to Slack. The detectors return message dicts;
    the actual posting happens in poll_once() / post() which we don't call.
    Tests inspect return values + state-mutation effects only.
  - slack_bolt is mocked at sys.modules level pre-import to avoid:
      (a) sys.exit(1) on missing AGI_LAB_SLACK_* env vars
      (b) live Slack API auth.test call from App(token=...) constructor

Test layout per detector:
  - At least 1 positive (detector fires; assert returned event dict shape)
  - At least 1 negative (detector returns []; state still well-formed)
  - For detect_holding_loop: dedupe regression test (call twice, assert
    second returns []; the D-202..D-210 incident's entire fix surface).

Reference: data/agents/tooling_engineer/episodic/2026-04-25_program_2_phase_3_d217_slack_detector_validation.md
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys
import time
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Module-level slack_bolt mock + slack_bot import.
#
# slack_bot.py:
#   - sys.exit(1) at import-time if AGI_LAB_SLACK_* env vars are unset.
#   - calls App(token=BOT_TOKEN) which hits Slack's auth.test API.
# Both are dodged by stubbing sys.modules['slack_bolt*'] and setting
# placeholder env vars BEFORE importing slack_bot.
# ---------------------------------------------------------------------------


def _import_slack_bot_with_mocked_slack():
    """Import tools/slack_bot.py with slack_bolt + env vars stubbed.

    Returns the imported module. Subsequent calls return a fresh import so
    each test gets a clean module-level state (REPO is re-readable, but
    detectors don't keep module-level mutable state — only the passed-in
    `state: dict`).
    """
    os.environ.setdefault("AGI_LAB_SLACK_BOT_TOKEN", "xoxb-test-stub")
    os.environ.setdefault("AGI_LAB_SLACK_APP_TOKEN", "xapp-test-stub")
    os.environ.setdefault("AGI_LAB_SLACK_CHANNEL", "C-test-stub")

    fake_app = MagicMock()
    fake_app.return_value = MagicMock()
    sys.modules["slack_bolt"] = MagicMock(App=fake_app)
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock(
        SocketModeHandler=MagicMock()
    )

    tools_dir = pathlib.Path(__file__).resolve().parent.parent / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    if "slack_bot" in sys.modules:
        return importlib.reload(sys.modules["slack_bot"])
    return importlib.import_module("slack_bot")


@pytest.fixture
def slack_bot_module():
    """Per-test-clean import of slack_bot with mocked Slack SDK."""
    return _import_slack_bot_with_mocked_slack()


@pytest.fixture
def fake_repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
              slack_bot_module: Any) -> pathlib.Path:
    """Re-point slack_bot.REPO at a tmp_path scratch dir.

    The 3 detectors all glob/read files relative to REPO. Re-pointing it
    isolates each test from the real <repo root>
    tree, which has live program-2 USER_*.md files + active session logs
    that would otherwise trigger every test.
    """
    monkeypatch.setattr(slack_bot_module, "REPO", tmp_path)
    return tmp_path


# Confirm-import sanity: ensures the mock approach didn't dodge silently.


def test_module_imports_with_three_detectors(slack_bot_module):
    """Sanity: slack_bot loaded; the 3 D-216 detectors are present."""
    assert hasattr(slack_bot_module, "detect_user_action_required")
    assert hasattr(slack_bot_module, "detect_holding_loop")
    assert hasattr(slack_bot_module, "detect_phase_boundary")
    # All three are in the DETECTORS list (so poll_once() invokes them).
    detector_names = [d.__name__ for d in slack_bot_module.DETECTORS]
    assert "detect_user_action_required" in detector_names
    assert "detect_holding_loop" in detector_names
    assert "detect_phase_boundary" in detector_names


# ===========================================================================
# detect_user_action_required
# ===========================================================================


def _make_user_md(repo: pathlib.Path, program: str, filename: str,
                  decision: str = "PENDING",
                  body: str = "User must decide whether Phase 3 GO or NOGO.") -> pathlib.Path:
    """Helper: write a programs/<program>/USER_<...>.md file with given DECISION."""
    program_dir = repo / "programs" / program
    program_dir.mkdir(parents=True, exist_ok=True)
    path = program_dir / filename
    content = (
        f"# {filename[:-3]}\n\n"
        f"DECISION: {decision}\n\n"
        f"{body}\n"
    )
    path.write_text(content)
    return path


def test_user_action_required_fires_on_first_sighting(
    fake_repo: pathlib.Path, slack_bot_module: Any
) -> None:
    """POSITIVE: a fresh USER_*.md file → exactly 1 URGENT event."""
    _make_user_md(fake_repo, "program_2_dense_vs_moe_sub100m",
                  "USER_GO_NOGO_DECISION.md",
                  decision="PENDING",
                  body="Decide GO/NOGO on the 146h Phase-3 factorial run.")
    state: dict = {}

    events = slack_bot_module.detect_user_action_required(state)

    assert len(events) == 1, f"Expected 1 event for fresh USER_*.md; got: {events}"
    ev = events[0]
    # Schema (lines 758-766 of slack_bot.py):
    assert ev["type"] == "USER_ACTION_REQUIRED"
    assert ev["urgent"] is True
    assert ev["category"] == "URGENT"
    # post-D-235 cleanup: headline is "User action required" (not all caps)
    assert "User action required" in ev["text"]
    assert "USER_GO_NOGO_DECISION.md" in ev["text"]
    assert "PENDING" in ev["text"]
    # Body preview must surface (the 'Decide GO/NOGO...' line).
    assert "GO/NOGO" in ev["text"] or "Phase-3" in ev["text"]

    # State mutation: file is now tracked.
    rel = "programs/program_2_dense_vs_moe_sub100m/USER_GO_NOGO_DECISION.md"
    assert rel in state["user_action_files"]
    assert state["user_action_files"][rel]["decision_value"] == "PENDING"


def test_user_action_required_does_not_fire_when_no_user_files(
    fake_repo: pathlib.Path, slack_bot_module: Any
) -> None:
    """NEGATIVE: no USER_*.md present → empty event list, clean state."""
    # programs dir exists but contains NO USER_*.md (only a normal markdown).
    program_dir = fake_repo / "programs" / "program_2_dense_vs_moe_sub100m"
    program_dir.mkdir(parents=True)
    (program_dir / "regular_doc.md").write_text("# Some other doc\n")
    state: dict = {}

    events = slack_bot_module.detect_user_action_required(state)

    assert events == [], f"Expected no events when no USER_*.md present; got: {events}"
    assert state["user_action_files"] == {}, (
        "user_action_files must remain empty when no matching files exist"
    )


def test_user_action_required_does_not_repost_within_six_hours(
    fake_repo: pathlib.Path, slack_bot_module: Any,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEGATIVE (re-post throttle): second call within 6h returns no event.

    Pin the 6h re-post cadence (line 724: `repost_interval = 6 * 3600`).
    Bonus: also tests that the state correctly carries across calls.
    """
    _make_user_md(fake_repo, "program_2_dense_vs_moe_sub100m",
                  "USER_GO_NOGO_DECISION.md", decision="PENDING")
    state: dict = {}

    # 1st call at t=1000.0 → posts (first-sight).
    monkeypatch.setattr(slack_bot_module.time, "time", lambda: 1000.0)
    events1 = slack_bot_module.detect_user_action_required(state)
    assert len(events1) == 1

    # 2nd call at t=1000.0 + 1h → must NOT re-post (6h cadence not met).
    monkeypatch.setattr(slack_bot_module.time, "time", lambda: 1000.0 + 3600.0)
    events2 = slack_bot_module.detect_user_action_required(state)
    assert events2 == [], (
        f"Expected no re-post within 6h; got: {events2}. The 6h cadence is the "
        "documented re-post interval (line 724 repost_interval=6*3600)."
    )


# ===========================================================================
# detect_holding_loop
# ===========================================================================


def _make_session_logs(repo: pathlib.Path, n_short: int, n_long: int = 0,
                       short_size: int = 4096, long_size: int = 32768,
                       mtime_offset_seconds: float = 0.0) -> pathlib.Path:
    """Create N short + M long session_*.log files.

    Each file's mtime is set to now-mtime_offset_seconds to control the 2h-window
    inclusion. short_size < 8192 (the threshold per line 829); long_size >= 8192.
    """
    log_dir = repo / "data/infra/session_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for i in range(n_short):
        f = log_dir / f"session_{1000 + i}_short.log"
        f.write_bytes(b"x" * short_size)
        os.utime(f, (now - mtime_offset_seconds, now - mtime_offset_seconds))
    for i in range(n_long):
        f = log_dir / f"session_{2000 + i}_long.log"
        f.write_bytes(b"x" * long_size)
        os.utime(f, (now - mtime_offset_seconds, now - mtime_offset_seconds))
    return log_dir


def test_holding_loop_fires_on_5_short_sessions(
    fake_repo: pathlib.Path, slack_bot_module: Any
) -> None:
    """POSITIVE: 5 short (<8KB) session_*.log files in trailing 2h → URGENT event.

    Pins the trigger threshold (line 831: trigger_count = 5).
    """
    _make_session_logs(fake_repo, n_short=5)
    state: dict = {}

    events = slack_bot_module.detect_holding_loop(state)

    assert len(events) == 1, f"Expected 1 event at trigger_count=5; got: {events}"
    ev = events[0]
    assert ev["type"] == "HOLDING_LOOP_DETECTED"
    assert ev["urgent"] is True
    assert ev["category"] == "URGENT"
    # Headline phrasing was rewritten post-D-217 (current: "Director looping").
    assert "Director looping" in ev["text"] or "Holding-loop" in ev["text"]
    # Token-burn line surfaces the count.
    assert "5 short" in ev["text"]
    # State mutation: signaled-at now non-zero, last_count=5.
    assert state["holding_loop_signaled_at"] > 0
    assert state["holding_loop_last_count"] == 5


def test_holding_loop_does_not_fire_below_threshold(
    fake_repo: pathlib.Path, slack_bot_module: Any
) -> None:
    """NEGATIVE: only 2 short sessions (well below trigger_count=5) → no event."""
    _make_session_logs(fake_repo, n_short=2)
    state: dict = {}

    events = slack_bot_module.detect_holding_loop(state)

    assert events == [], (
        f"Expected no event with 2 short sessions (< trigger_count=5); got: {events}"
    )
    # State stays at zero — no false signaling.
    assert state["holding_loop_signaled_at"] == 0
    assert state["holding_loop_last_count"] == 0


def test_holding_loop_does_not_fire_on_long_sessions(
    fake_repo: pathlib.Path, slack_bot_module: Any
) -> None:
    """NEGATIVE: 6 LONG (>=8KB) sessions — over the count, but not 'short' — no event.

    Defends the size threshold (line 829: short_threshold = 8192).
    """
    _make_session_logs(fake_repo, n_short=0, n_long=6)
    state: dict = {}

    events = slack_bot_module.detect_holding_loop(state)
    assert events == [], (
        f"Long sessions must not count toward holding-loop trigger; got: {events}"
    )


def test_holding_loop_dedupe_no_double_post(
    fake_repo: pathlib.Path, slack_bot_module: Any
) -> None:
    """DEDUPE REGRESSION: call detect_holding_loop twice with same fixtures.

    The first call posts; the second call must return [] because
    holding_loop_signaled_at was just set in call 1. This is the entire
    point of the D-202..D-210 incident fix — without dedupe, the bot would
    re-post every poll cycle (every 30s) for as long as the holding loop
    persists, generating exactly the holding-pattern notification spam the
    incident catalogued.

    Pin: line 851 (`if last_signaled_at == 0: should_post = True`) +
         line 853 (`elif (short_recent - last_count) >= repost_step`).
    """
    _make_session_logs(fake_repo, n_short=6)  # over threshold
    state: dict = {}

    # Call 1: fires.
    events1 = slack_bot_module.detect_holding_loop(state)
    assert len(events1) == 1, (
        f"Sanity: call 1 must fire (count >= trigger); got: {events1}"
    )
    signaled_at_after_call_1 = state["holding_loop_signaled_at"]
    assert signaled_at_after_call_1 > 0
    last_count_after_call_1 = state["holding_loop_last_count"]
    assert last_count_after_call_1 == 6

    # Call 2: same fixtures + persisted state → must NOT re-post.
    # (count has not grown by +5; last_signaled_at != 0.)
    events2 = slack_bot_module.detect_holding_loop(state)
    assert events2 == [], (
        f"DEDUPE BROKEN: call 2 with persisted state must return [] "
        f"(holding_loop_signaled_at={signaled_at_after_call_1} should suppress "
        f"re-post; growth-based re-post needs +5, not 0). Got: {events2}. "
        f"This is the D-202..D-210 incident fix — every poll cycle re-posting "
        f"is the exact bug that motivated the dedupe state keys."
    )
    # State: signaled_at unchanged (dedupe took effect, not a fresh post).
    assert state["holding_loop_signaled_at"] == signaled_at_after_call_1
    assert state["holding_loop_last_count"] == 6


# ===========================================================================
# D-251 amendment: live-training discriminator
# ===========================================================================


def _make_runs_dir_stdout(repo: pathlib.Path, run_name: str = "phase3_A42",
                          mtime_offset_seconds: float = 30.0) -> pathlib.Path:
    """Write a fake data/runs/<run_name>/stdout.log under the fake REPO.

    Default mtime_offset=30s → "fresh" (within 30-min window).
    Pass 3600.0 for stale (1 hour ago).
    """
    run_dir = repo / "data/runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "stdout.log"
    log.write_text("step=2160\n")
    now = time.time()
    os.utime(log, (now - mtime_offset_seconds, now - mtime_offset_seconds))
    return log


def _patch_pgrep(monkeypatch: pytest.MonkeyPatch, slack_bot_module: Any,
                 returncode: int) -> list[list[str]]:
    """Replace subprocess.run inside slack_bot with a stub that returns the
    given returncode. Returns a list that captures all calls (for assert).
    """
    calls: list[list[str]] = []

    class _FakeProc:
        def __init__(self, rc: int) -> None:
            self.returncode = rc

    def _fake_run(argv, *args, **kwargs):
        calls.append(list(argv))
        return _FakeProc(returncode)

    monkeypatch.setattr(slack_bot_module.subprocess, "run", _fake_run)
    return calls


def test_d251_discriminator_true_when_training_alive_and_log_fresh(
    fake_repo: pathlib.Path, slack_bot_module: Any,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-251 unit test: _training_is_advancing() returns True iff
    pgrep matches AND a fresh data/runs/*/stdout.log exists.
    """
    _patch_pgrep(monkeypatch, slack_bot_module, returncode=0)
    _make_runs_dir_stdout(fake_repo, "phase3_A42",
                          mtime_offset_seconds=30.0)

    assert slack_bot_module._training_is_advancing() is True


def test_d251_discriminator_false_when_no_training_process(
    fake_repo: pathlib.Path, slack_bot_module: Any,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-251 unit test: pgrep returns 1 (no match) → discriminator False."""
    _patch_pgrep(monkeypatch, slack_bot_module, returncode=1)
    _make_runs_dir_stdout(fake_repo, "phase3_A42",
                          mtime_offset_seconds=30.0)

    assert slack_bot_module._training_is_advancing() is False


def test_d251_discriminator_false_when_log_stale(
    fake_repo: pathlib.Path, slack_bot_module: Any,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-251 unit test: pgrep matches but stdout.log mtime > 30 min ago
    (training process is alive but stuck) → discriminator False.
    """
    _patch_pgrep(monkeypatch, slack_bot_module, returncode=0)
    _make_runs_dir_stdout(fake_repo, "phase3_A42",
                          mtime_offset_seconds=3600.0)  # 1h ago

    assert slack_bot_module._training_is_advancing() is False


def test_d251_discriminator_excludes_factorial_log_dir(
    fake_repo: pathlib.Path, slack_bot_module: Any,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-251 unit test: data/runs/phase3_factorial_log/stdout.log is the
    orchestrator's own log, NOT a per-cell training log. A fresh mtime
    there alone must not mark training as advancing.
    """
    _patch_pgrep(monkeypatch, slack_bot_module, returncode=0)
    _make_runs_dir_stdout(fake_repo, "phase3_factorial_log",
                          mtime_offset_seconds=30.0)
    # No real per-cell run dir — only the excluded one.

    assert slack_bot_module._training_is_advancing() is False


def test_d251_holding_loop_suppressed_when_training_advancing(
    fake_repo: pathlib.Path, slack_bot_module: Any,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-251 INTEGRATION: 6 short sessions + live training advancing →
    detect_holding_loop returns [] (no event), state stays clean.

    This is the entire D-251 fix surface — the D-237..D-247 monitoring
    template (small sessions by design + healthy training) must not
    trigger the false-positive alert that fired at 21:56:44 UTC.
    """
    _make_session_logs(fake_repo, n_short=6)  # over trigger
    _patch_pgrep(monkeypatch, slack_bot_module, returncode=0)
    _make_runs_dir_stdout(fake_repo, "phase3_A42",
                          mtime_offset_seconds=30.0)

    state: dict = {}
    events = slack_bot_module.detect_holding_loop(state)

    assert events == [], (
        f"D-251 fix BROKEN: live-training-advancing path must suppress "
        f"the holding-loop alert. Got events: {events}. The current "
        f"D-237..D-247 monitoring template (one pgrep + one tail + one "
        f"ls per session) is intentionally small and must NOT trigger "
        f"the alert when phase-3 training is healthy."
    )
    assert state["holding_loop_signaled_at"] == 0
    assert state["holding_loop_last_count"] == 0


def test_d251_holding_loop_fires_when_no_training_alive(
    fake_repo: pathlib.Path, slack_bot_module: Any,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-251 INTEGRATION: 6 short sessions + NO live training (pgrep returns
    1) → original D-202..D-225 path fires (event emitted, state mutated).

    Pin: the genuine-stuck path is unchanged. The D-251 amendment ONLY
    suppresses when training is actively advancing.
    """
    _make_session_logs(fake_repo, n_short=6)
    _patch_pgrep(monkeypatch, slack_bot_module, returncode=1)
    # No data/runs/ either — discriminator fails on both signals.

    state: dict = {}
    events = slack_bot_module.detect_holding_loop(state)

    assert len(events) == 1, (
        f"Genuine-stuck path must STILL fire (D-202..D-225 shape). "
        f"Got events: {events}. The D-251 amendment must not break the "
        f"original incident fix."
    )
    assert events[0]["type"] == "HOLDING_LOOP_DETECTED"
    assert events[0]["urgent"] is True
    assert state["holding_loop_signaled_at"] > 0
    assert state["holding_loop_last_count"] == 6


def test_d251_holding_loop_emits_clear_when_training_resumes(
    fake_repo: pathlib.Path, slack_bot_module: Any,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-251 STATE TRANSITION: was-signaling → training-now-advancing
    must emit a HOLDING_LOOP_END "good" clear so the Slack channel
    knows the alert was a false positive that has been suppressed.

    Sequence:
      1. Pre-seed state with signaled_at > 0 (as if a prior alert fired).
      2. Set up: 6 short sessions + live training advancing.
      3. detect_holding_loop must emit ONE clear event + zero state.
    """
    _make_session_logs(fake_repo, n_short=6)
    _patch_pgrep(monkeypatch, slack_bot_module, returncode=0)
    _make_runs_dir_stdout(fake_repo, "phase3_A42",
                          mtime_offset_seconds=30.0)

    state: dict = {
        "holding_loop_signaled_at": 1000.0,  # prior alert was active
        "holding_loop_last_count": 8,
    }

    events = slack_bot_module.detect_holding_loop(state)

    assert len(events) == 1, (
        f"State transition (was-alert → suppress) must emit a clear; "
        f"got: {events}"
    )
    ev = events[0]
    assert ev["type"] == "HOLDING_LOOP_END"
    assert ev["category"] == "GOOD"
    assert ev["urgent"] is False
    # State was reset.
    assert state["holding_loop_signaled_at"] == 0
    assert state["holding_loop_last_count"] == 0


# ===========================================================================
# detect_phase_boundary
# ===========================================================================


def _make_current_md(repo: pathlib.Path, program: str = "Program 2",
                     phase: str = "Phase 3 OPEN",
                     status: str = "PRE-FLIGHT") -> pathlib.Path:
    """Write data/memories/current.md with the 3 phase-detector-relevant headings.

    Schema (lines 909-911 of slack_bot.py): three lines, '## Current Program:',
    '## Current Phase:', '## Status:'.
    """
    mem_dir = repo / "data/memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    path = mem_dir / "current.md"
    path.write_text(
        f"# Current State\n\n"
        f"## Current Program: {program}\n"
        f"## Current Phase: {phase}\n"
        f"## Status: {status}\n"
    )
    return path


def test_phase_boundary_fires_on_phase_transition(
    fake_repo: pathlib.Path, slack_bot_module: Any
) -> None:
    """POSITIVE: state has a prior phase recorded; current.md has a NEW phase
    → exactly 1 event with a clean transition headline.

    Post-D-235 rewrite: detect_phase_boundary now uses
    last_phase_summary_program / last_phase_summary_phase (clean strings)
    rather than the prior token-soup fingerprint.
    """
    _make_current_md(fake_repo,
                     program="program_2_dense_vs_moe_sub100m",
                     phase="Phase 4 OPEN", status="EXECUTE")
    state: dict = {
        "last_phase_summary_program": "program_2_dense_vs_moe_sub100m",
        "last_phase_summary_phase": "P3 CLOSED",
    }

    events = slack_bot_module.detect_phase_boundary(state)

    assert len(events) == 1, f"Expected 1 event on phase change; got: {events}"
    ev = events[0]
    assert ev["type"] == "PHASE_BOUNDARY"
    # New format: "Phase transition: P3 CLOSED → P4 OPEN" or similar
    assert "Phase transition" in ev["text"]
    assert "P3 CLOSED" in ev["text"]
    assert "P4" in ev["text"] and "OPEN" in ev["text"]
    # OPEN / EXECUTE are not in the urgent-token list → ATTENTION
    assert ev["urgent"] is False
    assert ev["category"] == "ATTENTION"
    # State updated to new phase
    assert state["last_phase_summary_phase"] != "P3 CLOSED"


def test_phase_boundary_first_scan_records_baseline_no_post(
    fake_repo: pathlib.Path, slack_bot_module: Any
) -> None:
    """NEGATIVE (first-scan rule): state has no prior program/phase →
    record + return []. Prevents post-flood on bot startup.
    """
    _make_current_md(fake_repo,
                     program="program_2_dense_vs_moe_sub100m",
                     phase="Phase 3 OPEN", status="EXECUTE")
    state: dict = {}  # no prior program/phase recorded

    events = slack_bot_module.detect_phase_boundary(state)

    assert events == [], f"First scan must NOT post; got: {events}"
    assert state.get("last_phase_summary_program"), \
        "Baseline program must be recorded for next-call comparison"
    assert state.get("last_phase_summary_phase"), \
        "Baseline phase must be recorded for next-call comparison"


def test_phase_boundary_no_change_no_post(
    fake_repo: pathlib.Path, slack_bot_module: Any
) -> None:
    """NEGATIVE: identical state across two calls → second call returns []."""
    _make_current_md(fake_repo,
                     program="program_2_dense_vs_moe_sub100m",
                     phase="Phase 3 OPEN", status="PRE-FLIGHT")

    state: dict = {}
    e1 = slack_bot_module.detect_phase_boundary(state)
    assert e1 == [], "first scan should be silent (baseline)"
    baseline_program = state["last_phase_summary_program"]
    baseline_phase = state["last_phase_summary_phase"]

    e2 = slack_bot_module.detect_phase_boundary(state)
    assert e2 == [], f"Identical-state re-scan must return []; got: {e2}"
    assert state["last_phase_summary_program"] == baseline_program
    assert state["last_phase_summary_phase"] == baseline_phase


def test_phase_boundary_urgent_on_victory_keyword(
    fake_repo: pathlib.Path, slack_bot_module: Any
) -> None:
    """A transition into a VICTORY-tagged phase → urgent=True / URGENT category."""
    _make_current_md(fake_repo,
                     program="program_2_dense_vs_moe_sub100m",
                     phase="Phase 3 VICTORY", status="VICTORY")
    state: dict = {
        "last_phase_summary_program": "program_2_dense_vs_moe_sub100m",
        "last_phase_summary_phase": "P3 OPEN",
    }

    events = slack_bot_module.detect_phase_boundary(state)
    assert len(events) == 1
    ev = events[0]
    assert ev["urgent"] is True, f"VICTORY phase must mark urgent; got: {ev}"
    assert ev["category"] == "URGENT"


# ===========================================================================
# Dry-run boundary check (per dispatch HARD requirement)
# ===========================================================================


def test_detectors_do_not_post_to_slack_when_called_directly(
    fake_repo: pathlib.Path, slack_bot_module: Any,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defensive: confirm the detectors RETURN dicts, never call .post() inline.

    Per D-217 dispatch HARD discipline: detectors must be pure (return-only).
    If a future refactor moves posting INTO a detector, this test catches it.

    Method: monkeypatch slack_bot.post to a sentinel that flags the call.
    Trigger all 3 detectors with positive fixtures. Assert the sentinel was
    NEVER hit — only the return-value contract is exercised.
    """
    posted: list = []

    def _record_post(text, *, category="INFO", urgent=False):
        posted.append((category, text))

    monkeypatch.setattr(slack_bot_module, "post", _record_post)

    # Set up positive fixtures for all 3 detectors so they all WOULD fire if
    # they posted directly.
    _make_user_md(fake_repo, "program_2_dense_vs_moe_sub100m",
                  "USER_GO_NOGO_DECISION.md")
    _make_session_logs(fake_repo, n_short=6)
    _make_current_md(fake_repo, phase="Phase 3 CLOSED", status="VICTORY")

    state: dict = {
        "last_phase_summary_program": "program_2_dense_vs_moe_sub100m",
        "last_phase_summary_phase": "P3 PRE-FLIGHT",
    }

    e1 = slack_bot_module.detect_user_action_required(state)
    e2 = slack_bot_module.detect_holding_loop(state)
    e3 = slack_bot_module.detect_phase_boundary(state)

    # All three returned events (positive-fixture sanity).
    assert len(e1) >= 1
    assert len(e2) >= 1
    assert len(e3) >= 1
    # CRITICAL: post() was NEVER called inside the detectors. They are pure.
    assert posted == [], (
        f"Detector side-effect leak: post() was called from inside a detector "
        f"function. Per D-217 architecture, detectors return events; only "
        f"poll_once() (the orchestrator) posts. Calls observed: {posted}"
    )


# ===========================================================================
# Training-progress detector tests (D-288 — slack-bot promotion of routine
# Director monitoring during long-run training).
# ===========================================================================


def _make_ckpt(fake_repo: Path, cell: str, step: int, body: bytes = b"") -> Path:
    ckpt_dir = fake_repo / "data" / "checkpoints" / "phase3_factorial" / cell
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    p = ckpt_dir / f"step_{step:06d}.ckpt"
    p.write_bytes(body or b"x")
    return p


def _make_cell_stdout(fake_repo: Path, cell: str, last_step_line: str) -> Path:
    run_dir = fake_repo / "data" / "runs" / f"phase3_{cell}"
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "stdout.log"
    p.write_text(
        # Header chatter
        "[init] starting...\n"
        "[load] step=1 layer=0 ema=[1.0]\n"
        f"{last_step_line}\n"
    )
    return p


def test_parse_latest_step_signals_extracts_loss_ppl_ms(fake_repo, slack_bot_module):
    line = "  500  |   6.5446 ( 0.000 + 0.0000 +  6.545) |    695.49 |  6.702 | 0.00000 | 0.0000 |  0 | 8799"
    p = _make_cell_stdout(fake_repo, "A42", line)
    sig = slack_bot_module._parse_latest_step_signals(p)
    assert sig == {"step": 500, "loss": 6.5446, "ppl": 695.49, "ms": 8799.0}


def test_parse_latest_step_signals_returns_empty_if_no_step_lines(fake_repo, slack_bot_module):
    p = _make_cell_stdout(fake_repo, "A42", "[init] no step lines here")
    sig = slack_bot_module._parse_latest_step_signals(p)
    assert sig == {}


def test_parse_latest_step_signals_handles_missing_file(fake_repo, slack_bot_module):
    sig = slack_bot_module._parse_latest_step_signals(fake_repo / "nope.log")
    assert sig == {}


def test_detect_training_progress_posts_on_new_checkpoint(fake_repo, slack_bot_module):
    _make_ckpt(fake_repo, "A42", 500)
    _make_cell_stdout(fake_repo, "A42",
                      "  500  |   6.5446 ( 0.000 + 0.0000 +  6.545) |    695.49 "
                      "|  6.702 | 0.00000 | 0.0000 |  0 | 8799")
    state: dict = {"ckpt_progress": {}, "cell_completed": {}}
    events = slack_bot_module.detect_training_progress(state)
    assert len(events) == 1
    assert events[0]["type"] == "CHECKPOINT_LANDED"
    assert "A42" in events[0]["text"]
    assert "500/5000" in events[0]["text"]
    # State updated so next call doesn't re-post
    assert state["ckpt_progress"]["A42"] == 500


def test_detect_training_progress_idempotent_on_no_new_ckpt(fake_repo, slack_bot_module):
    _make_ckpt(fake_repo, "A42", 500)
    _make_cell_stdout(fake_repo, "A42",
                      "  500  |   6.5  (0+0+6.5) |    695 |  6 | 0 | 0 |  0 | 8500")
    state: dict = {"ckpt_progress": {"A42": 500}, "cell_completed": {}}
    events = slack_bot_module.detect_training_progress(state)
    assert events == []


def test_detect_training_progress_fires_cell_complete_at_5000(fake_repo, slack_bot_module):
    _make_ckpt(fake_repo, "A42", 5000)
    _make_cell_stdout(fake_repo, "A42",
                      " 5000  |   4.8  (0+0+4.8) |    131 |  6 | 0 | 0 |  0 | 8500")
    state: dict = {"ckpt_progress": {"A42": 4500}, "cell_completed": {}}
    events = slack_bot_module.detect_training_progress(state)
    assert len(events) == 1
    assert events[0]["type"] == "CELL_COMPLETE"
    assert state["cell_completed"]["A42"] is True


def test_detect_training_progress_skips_non_cell_dirs(fake_repo, slack_bot_module):
    """Subdirs not matching the <cell-letter><seed> pattern (e.g. 'archive',
    'tmp') are ignored — only A42-style names are treated as cells."""
    archive_dir = fake_repo / "data" / "checkpoints" / "phase3_factorial" / "archive_d234"
    archive_dir.mkdir(parents=True)
    (archive_dir / "step_001000.ckpt").write_bytes(b"x")
    state: dict = {"ckpt_progress": {}, "cell_completed": {}}
    events = slack_bot_module.detect_training_progress(state)
    assert events == []


def test_detect_training_progress_independent_per_cell(fake_repo, slack_bot_module):
    """A new ckpt in B42 doesn't suppress posting for D42, and vice versa."""
    _make_ckpt(fake_repo, "A42", 500)
    _make_ckpt(fake_repo, "D42", 500)
    _make_cell_stdout(fake_repo, "A42",
                      "  500  |   6  (0+0+6) | 695 | 6 | 0 | 0 |  0 | 8500")
    _make_cell_stdout(fake_repo, "D42",
                      "  500  |   7  (0+0+7) | 1000 | 7 | 0 | 0 |  0 | 9000")
    state: dict = {"ckpt_progress": {}, "cell_completed": {}}
    events = slack_bot_module.detect_training_progress(state)
    assert len(events) == 2
    cells_seen = {e["text"].split()[2] for e in events}  # extract cell name
    # text starts with "*Checkpoint: A42 ..." so position 2 after split gives cell
    # Actually let's just check both cell names appear somewhere
    text_blob = " ".join(e["text"] for e in events)
    assert "A42" in text_blob and "D42" in text_blob
