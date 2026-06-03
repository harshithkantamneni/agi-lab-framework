"""Tests for run_agi_lab.sh holding-loop detector (D-251 amendment).

Coverage scope:
  Validates the bash function `detect_holding_loop_backoff` and the new
  D-251 discriminator helper `_training_is_advancing` in run_agi_lab.sh.
  These shell functions are tested via a bash subprocess that sources only
  the function definitions (not the main loop) using awk extraction.

D-251 motivation:
  Across D-237..D-247 (12 sessions, ~4-5 KB each) the detector started
  firing FALSE POSITIVES while a healthy 12-cell factorial training run
  (orchestrator pid 40822 → run_long pid 40824 → scale_experiment pid
  40825) was actively making progress at ~6.5 step/min. The Director's
  short sessions were intentionally small (one pgrep + one tail + one ls
  per the D-247 monitoring template) — the heuristic that conflated
  "small session" with "stuck" had to be amended with a live-training
  discriminator.

Test cases per dispatch:
  (a) Historical D-202..D-225 shape: small sessions + no live training
      → fire (genuine-stuck, original behavior preserved).
  (b) Current D-237..D-247 shape: small sessions + live training
      advancing → suppress (D-251 correct-monitoring path).
  (c) Edge: training PID exists but stdout.log frozen >30 min → fire
      (genuine-stuck even with stale process).

The test fixtures mock pgrep via a fake script on PATH and write fake
session_*.log + data/runs/*/stdout.log files into a tmp_path so the live
phase-3 training run is not disturbed.

Reference: data/agents/tooling_engineer/episodic/2026-04-28_d251_holding_loop_detector_false_positive_fix.md
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import stat
import subprocess
import textwrap
import time

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN_AGI_LAB = REPO_ROOT / "run_agi_lab.sh"


def _extract_functions(script: pathlib.Path) -> str:
    """Pull out only the function definitions we want to test.

    We extract `_training_is_advancing` and `detect_holding_loop_backoff`
    by name. This avoids sourcing the main loop / cd / exports at the top
    of run_agi_lab.sh (which would block the test on probe_ready calls).
    Brace-matching state machine — POSIX shell function syntax is
    'name() { ... }'.
    """
    text = script.read_text()
    wanted = {"_training_is_advancing", "detect_holding_loop_backoff"}
    extracted: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\w+)\(\)\s*\{\s*$", line)
        if m and m.group(1) in wanted:
            depth = 1
            block = [line]
            j = i + 1
            while j < len(lines) and depth > 0:
                block.append(lines[j])
                # Naive brace counting — these functions don't have
                # heredocs or quoted braces that would confuse us.
                depth += lines[j].count("{")
                depth -= lines[j].count("}")
                j += 1
            extracted.append("\n".join(block))
            i = j
            continue
        i += 1
    return "\n\n".join(extracted)


@pytest.fixture(scope="module")
def function_text() -> str:
    """Cache the extracted function text once per module."""
    text = _extract_functions(RUN_AGI_LAB)
    assert "_training_is_advancing" in text, (
        "_training_is_advancing not extracted — check run_agi_lab.sh structure"
    )
    assert "detect_holding_loop_backoff" in text, (
        "detect_holding_loop_backoff not extracted — check run_agi_lab.sh"
    )
    return text


def _make_session_logs(log_dir: pathlib.Path, n_short: int,
                       short_size: int = 4096,
                       mtime_offset_seconds: float = 0.0) -> None:
    """Write N short session_*.log files for the bash detector to find."""
    log_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for i in range(n_short):
        f = log_dir / f"session_{1000 + i}_short.log"
        f.write_bytes(b"x" * short_size)
        os.utime(f, (now - mtime_offset_seconds, now - mtime_offset_seconds))


def _make_stdout_log(runs_dir: pathlib.Path, run_name: str,
                     mtime_offset_seconds: float = 30.0) -> pathlib.Path:
    """Write a fake data/runs/<run_name>/stdout.log with controlled mtime.

    Default mtime_offset_seconds=30 → "fresh" (within 30-min window).
    For stale-fixture tests pass a value like 3600 (1 hour ago).
    """
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "stdout.log"
    log.write_text("step=2160\n")
    now = time.time()
    os.utime(log, (now - mtime_offset_seconds, now - mtime_offset_seconds))
    return log


def _make_fake_pgrep(bin_dir: pathlib.Path, *,
                     should_match: bool) -> pathlib.Path:
    """Write a fake pgrep at <bin_dir>/pgrep that always returns 0 or 1.

    We don't care about argv parsing — pgrep is invoked with a single
    pattern by the discriminator. should_match=True → exit 0, write a
    fake PID to stdout. should_match=False → exit 1 (no match), no output.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    pgrep = bin_dir / "pgrep"
    if should_match:
        pgrep.write_text("#!/bin/bash\necho 99999\nexit 0\n")
    else:
        pgrep.write_text("#!/bin/bash\nexit 1\n")
    pgrep.chmod(pgrep.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return pgrep


def _run_detector(function_text: str, workdir: pathlib.Path,
                  bin_dir: pathlib.Path, log_dir_relative: str = "logs",
                  runs_dir_relative: str = "data/runs") -> dict:
    """Run detect_holding_loop_backoff in a controlled bash subprocess.

    Returns parsed result dict:
      {
        "backoff_secs": int,
        "advancing_rc": int,  # 0 = advancing-true, 1 = advancing-false
      }
    """
    script = textwrap.dedent(f"""
        set -uo pipefail
        # PATH first → fake pgrep wins over /usr/bin/pgrep
        export PATH="{bin_dir}:$PATH"
        export LOG_DIR="{log_dir_relative}"
        export RUNS_DIR_OVERRIDE="{runs_dir_relative}"
        export RUN_FRESH_MIN="${{RUN_FRESH_MIN:-30}}"

        {function_text}

        # Probe the discriminator on its own so we can see why suppression
        # happened or didn't.
        if _training_is_advancing; then
            advancing_rc=0
        else
            advancing_rc=1
        fi

        detect_holding_loop_backoff
        echo "BACKOFF=${{HOLDING_BACKOFF_SECS}}"
        echo "ADVANCING=${{advancing_rc}}"
    """)
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=workdir, capture_output=True, text=True, timeout=15,
    )
    out = result.stdout
    backoff = int(re.search(r"BACKOFF=(\d+)", out).group(1))
    advancing = int(re.search(r"ADVANCING=(\d+)", out).group(1))
    return {
        "backoff_secs": backoff,
        "advancing_rc": advancing,
        "stdout": out,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


# ===========================================================================
# Test cases
# ===========================================================================


def test_genuine_stuck_no_training_alive_fires_alert(
    tmp_path: pathlib.Path, function_text: str
) -> None:
    """(a) Historical D-202..D-225 shape: 6 short sessions + no live training
    process → detector fires (HOLDING_BACKOFF_SECS > 0).

    Pin: original D-202..D-210 incident behavior is preserved when the
    discriminator returns False.
    """
    log_dir = tmp_path / "logs"
    bin_dir = tmp_path / "fakebin"
    _make_session_logs(log_dir, n_short=6)
    _make_fake_pgrep(bin_dir, should_match=False)
    # No data/runs/*/stdout.log either → discriminator will also fail on (2).

    result = _run_detector(function_text, tmp_path, bin_dir,
                           log_dir_relative="logs",
                           runs_dir_relative="data/runs")

    assert result["advancing_rc"] == 1, (
        f"Discriminator must return false (no training process); "
        f"stdout={result['stdout']} stderr={result['stderr']}"
    )
    assert result["backoff_secs"] > 0, (
        f"Detector must fire on D-202..D-225 shape (6 short, no training). "
        f"Got backoff={result['backoff_secs']}; stdout={result['stdout']}"
    )


def test_live_training_advancing_suppresses_alert(
    tmp_path: pathlib.Path, function_text: str
) -> None:
    """(b) Current D-237..D-247 shape: 6 short sessions + live training
    process AND fresh stdout.log → detector suppresses (HOLDING_BACKOFF_SECS == 0).

    This is the entire D-251 fix — the current healthy phase-3 monitoring
    pattern must not trigger the holding-loop backoff.
    """
    log_dir = tmp_path / "logs"
    bin_dir = tmp_path / "fakebin"
    runs_dir = tmp_path / "data/runs"
    _make_session_logs(log_dir, n_short=6)
    _make_fake_pgrep(bin_dir, should_match=True)
    _make_stdout_log(runs_dir, "phase3_A42", mtime_offset_seconds=30.0)

    result = _run_detector(function_text, tmp_path, bin_dir,
                           log_dir_relative="logs",
                           runs_dir_relative="data/runs")

    assert result["advancing_rc"] == 0, (
        f"Discriminator must return true (training alive + log fresh); "
        f"stdout={result['stdout']} stderr={result['stderr']}"
    )
    assert result["backoff_secs"] == 0, (
        f"Detector must SUPPRESS on D-237..D-247 shape (6 short + live "
        f"training advancing). Got backoff={result['backoff_secs']}; "
        f"stdout={result['stdout']}. This is the D-251 fix — false "
        f"positive must not fire when training is healthy."
    )


def test_training_pid_alive_but_log_stale_fires_alert(
    tmp_path: pathlib.Path, function_text: str
) -> None:
    """(c) Edge: 6 short sessions + pgrep matches BUT stdout.log mtime is
    >30 min old → detector fires (training process is stuck even though
    PID exists).

    Pin: the discriminator requires BOTH (process alive) AND (log fresh).
    A frozen training run (e.g., deadlocked, or hanging on I/O) must still
    be treated as "stuck Director" so the operator gets paged.
    """
    log_dir = tmp_path / "logs"
    bin_dir = tmp_path / "fakebin"
    runs_dir = tmp_path / "data/runs"
    _make_session_logs(log_dir, n_short=6)
    _make_fake_pgrep(bin_dir, should_match=True)
    # 1 hour ago → stale (>30 min default fresh window).
    _make_stdout_log(runs_dir, "phase3_A42", mtime_offset_seconds=3600.0)

    result = _run_detector(function_text, tmp_path, bin_dir,
                           log_dir_relative="logs",
                           runs_dir_relative="data/runs")

    assert result["advancing_rc"] == 1, (
        f"Discriminator must return false (process alive but log stale); "
        f"stdout={result['stdout']} stderr={result['stderr']}"
    )
    assert result["backoff_secs"] > 0, (
        f"Detector must FIRE on stuck-training edge (pid alive but log "
        f"frozen >30 min). Got backoff={result['backoff_secs']}; "
        f"stdout={result['stdout']}."
    )


def test_factorial_log_dir_excluded_from_freshness_check(
    tmp_path: pathlib.Path, function_text: str
) -> None:
    """phase3_factorial_log/ is the orchestrator's own log dir, not a
    per-cell run dir. Even if it has a fresh stdout.log, that alone must
    NOT mark training as advancing (because the orchestrator can be
    looping while no actual training cell is running).

    Negative test: only phase3_factorial_log/stdout.log present → discriminator
    returns false → detector fires.
    """
    log_dir = tmp_path / "logs"
    bin_dir = tmp_path / "fakebin"
    runs_dir = tmp_path / "data/runs"
    _make_session_logs(log_dir, n_short=6)
    _make_fake_pgrep(bin_dir, should_match=True)
    _make_stdout_log(runs_dir, "phase3_factorial_log",
                     mtime_offset_seconds=30.0)
    # NO per-cell run dir → no eligible stdout.log within window.

    result = _run_detector(function_text, tmp_path, bin_dir,
                           log_dir_relative="logs",
                           runs_dir_relative="data/runs")

    assert result["advancing_rc"] == 1, (
        f"Discriminator must exclude phase3_factorial_log/ from freshness "
        f"check. Got advancing_rc={result['advancing_rc']}; "
        f"stdout={result['stdout']}"
    )
    assert result["backoff_secs"] > 0, (
        f"Detector must FIRE when only phase3_factorial_log/ is fresh "
        f"(no real per-cell training advancing). "
        f"Got backoff={result['backoff_secs']}."
    )


def test_below_threshold_no_alert_regardless_of_training(
    tmp_path: pathlib.Path, function_text: str
) -> None:
    """Sanity / regression: 2 short sessions (below trigger_count=5) → no
    alert, regardless of training state. This pins the precondition: the
    short-session count must reach trigger BEFORE the discriminator path
    is even consulted.
    """
    log_dir = tmp_path / "logs"
    bin_dir = tmp_path / "fakebin"
    _make_session_logs(log_dir, n_short=2)
    _make_fake_pgrep(bin_dir, should_match=False)

    result = _run_detector(function_text, tmp_path, bin_dir,
                           log_dir_relative="logs",
                           runs_dir_relative="data/runs")
    assert result["backoff_secs"] == 0, (
        f"Detector must not fire below trigger threshold (2 < 5). "
        f"Got backoff={result['backoff_secs']}."
    )


# ===========================================================================
# Cross-check: discriminator runs against the LIVE process tree
# ===========================================================================


def test_discriminator_against_live_process_tree(function_text: str) -> None:
    """Cross-check per dispatch: the discriminator must return TRUE against
    the currently running phase-3 process tree (orchestrator pid 40822 +
    children). This is a smoke test, NOT a hermetic test — it's skipped if
    no live training process is found.

    If the live training run is not present (e.g., test suite running
    after factorial completion), this test is skipped rather than failing,
    so it won't break CI in cleaner environments.
    """
    # Use the REAL pgrep this time + the REAL data/runs/ tree.
    has_live = subprocess.run(
        ["pgrep", "-f",
         "tools/run_long\\.py|tools/run_phase3_factorial\\.py|build/scale_experiment"],
        capture_output=True,
    ).returncode == 0
    if not has_live:
        pytest.skip("no live training process — cross-check is informational only")

    # Run the bash discriminator alone (no fake pgrep, no fixture overrides).
    script = textwrap.dedent(f"""
        set -uo pipefail
        unset RUNS_DIR_OVERRIDE LOG_DIR RUN_FRESH_MIN || true
        cd "{REPO_ROOT}"
        {function_text}
        if _training_is_advancing; then
            echo "ADVANCING=0"
        else
            echo "ADVANCING=1"
        fi
    """)
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True, text=True, timeout=15,
    )
    assert "ADVANCING=0" in result.stdout, (
        f"D-251 cross-check FAILED: live training is running but discriminator "
        f"returned false. stdout={result.stdout!r} stderr={result.stderr!r}. "
        f"This means the fix would not actually suppress the false positive "
        f"in the current Phase-3 monitoring sessions."
    )
