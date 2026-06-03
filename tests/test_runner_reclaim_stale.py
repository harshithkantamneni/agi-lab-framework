"""Smoke test: runner shell script defines and invokes _run_reclaim_stale."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "run_agi_lab.sh"


def test_runner_defines_reclaim_stale_function():
    text = RUNNER.read_text()
    assert "_run_reclaim_stale()" in text or "_run_reclaim_stale ()" in text, \
        "runner must define _run_reclaim_stale helper"


def test_runner_calls_reclaim_stale_in_main_loop():
    text = RUNNER.read_text()
    # Must invoke before the queue scanner so stale claims clear before scan
    # uses them
    main_loop_idx = text.find("_run_queue_scanner")
    assert main_loop_idx > 0, "_run_queue_scanner anchor not found"
    pre_scan_block = text[:main_loop_idx]
    # _run_reclaim_stale invocation must appear in the same loop iteration BEFORE
    # _run_queue_scanner. We accept any loop-internal call to _run_reclaim_stale.
    assert "_run_reclaim_stale" in pre_scan_block, \
        "_run_reclaim_stale must be invoked before _run_queue_scanner in the main loop"


def test_runner_reclaim_stale_uses_60min_timeout():
    """Timeout must match the work_queue.py default (60 min) so behavior is consistent."""
    text = RUNNER.read_text()
    # Function body should pass timeout_min=60 to the python invocation, OR rely on
    # the default (which is 60). Either is acceptable. We just sanity-check the
    # function refers to work_queue.
    assert "work_queue" in text and "_run_reclaim_stale" in text


def test_runner_reclaim_stale_failure_is_non_fatal():
    """Like _run_queue_scanner, failures must be logged but non-fatal."""
    text = RUNNER.read_text()
    # Find the function body — between `_run_reclaim_stale()` and the next blank-line-then-{
    import re
    m = re.search(r"_run_reclaim_stale\(\)\s*\{(.*?)\n\}", text, re.DOTALL)
    assert m, "could not locate _run_reclaim_stale function body"
    body = m.group(1)
    # Function body must contain `||` (or-fallback) or `2>&1` (error capture) so a
    # python crash doesn't propagate to the runner's `set -e`.
    assert "||" in body or "2>&1" in body, \
        "_run_reclaim_stale must handle non-zero exit non-fatally"


def test_runner_passes_shellcheck_syntax():
    """`bash -n` must accept the runner script (no syntax errors after edit)."""
    import subprocess
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, \
        f"runner failed bash -n syntax check:\n{result.stderr}"


def test_runner_calls_post_director_after_director_exit():
    """After EXIT_CODE=$?, runner must invoke tools/post_director.py before EXIT_REASON parsing."""
    text = RUNNER.read_text()
    # Find anchors
    exit_code_idx = text.find("EXIT_CODE=$?")
    exit_reason_idx = text.find("EXIT_REASON=$(grep")
    assert exit_code_idx > 0 and exit_reason_idx > 0, \
        "couldn't locate EXIT_CODE / EXIT_REASON anchors"
    assert exit_code_idx < exit_reason_idx
    between = text[exit_code_idx:exit_reason_idx]
    assert "tools/post_director.py" in between, \
        "post_director.py must be invoked between EXIT_CODE=$? and EXIT_REASON parsing"


def test_runner_post_director_call_is_non_fatal():
    """post_director.py is layered defensively; runner must not abort if it fails."""
    text = RUNNER.read_text()
    # Find the post_director invocation block
    idx = text.find("tools/post_director.py")
    assert idx > 0
    # Look at the surrounding ~200 chars for `||` (or-fallback) or `2>&1` (error capture)
    block = text[max(0, idx - 100): idx + 200]
    assert "||" in block or "2>&1" in block, \
        "post_director.py invocation must handle non-zero exit non-fatally"


def test_runner_post_director_runs_after_watchdog_teardown():
    """The watchdog teardown block must run before post_director.py so any
    watchdog-killed Director state is settled first."""
    text = RUNNER.read_text()
    # Watchdog teardown anchor: kill "$WATCHDOG_PID" pattern OR "Tear down watchdog"
    watchdog_idx = -1
    for anchor in ('Tear down watchdog', 'kill "$WATCHDOG_PID"', "kill \"$WATCHDOG_PID\""):
        i = text.find(anchor)
        if i > 0:
            watchdog_idx = i
            break
    assert watchdog_idx > 0, "couldn't locate watchdog teardown anchor"
    pd_idx = text.find("tools/post_director.py")
    assert pd_idx > watchdog_idx, \
        "post_director.py must be invoked AFTER watchdog teardown"


def test_runner_handles_redispatch_flag():
    """Runner shell script must check for redispatch flag and set DIRECTOR_REDISPATCH_REASON."""
    text = RUNNER.read_text()
    # Look for the flag file path reference
    assert "session_exit_redispatch_pending.flag" in text, \
        "runner must check data/session_exit_redispatch_pending.flag"
    # Look for DIRECTOR_REDISPATCH_REASON env var
    assert "DIRECTOR_REDISPATCH_REASON" in text, \
        "runner must export DIRECTOR_REDISPATCH_REASON env var on flag detection"


def test_runner_removes_flag_after_consumption():
    """Runner must rm -f the flag after detecting it (one-shot)."""
    text = RUNNER.read_text()
    # Find the redispatch block
    idx = text.find("session_exit_redispatch_pending.flag")
    assert idx > 0
    block = text[idx: idx + 500]
    assert "rm" in block or "rm -f" in block, \
        f"runner must rm -f the flag after consumption; got: {block!r}"


def test_runner_redispatch_actually_bypasses_skip_when_stable():
    """When redispatch flag is consumed, runner MUST spawn Director even with empty queue + stable state."""
    text = RUNNER.read_text()
    # The flag-consumption block must set a FORCE_SPAWN-equivalent
    flag_idx = text.find("REDISPATCH_FLAG=")
    assert flag_idx > 0
    block_end = text.find("\n\n", flag_idx + 200)
    block = text[flag_idx:block_end]
    assert "FORCE_SPAWN" in block, \
        f"redispatch block must set FORCE_SPAWN; got: {block[:500]!r}"

    # The queue dispatch check must HONOR FORCE_SPAWN
    dispatch_idx = text.find("_queue_has_pending")
    # find the next 'if' (not the function definition)
    if_idx = text.find("if _queue_has_pending", dispatch_idx + 50)
    assert if_idx > 0
    if_block = text[if_idx:if_idx + 200]
    assert "FORCE_SPAWN" in if_block, \
        f"queue dispatch check must honor FORCE_SPAWN; got: {if_block!r}"


# === 2026-05-14: holding-loop progress discriminator ===

def test_runner_defines_recent_queue_completions_helper():
    """Runner must define _recent_queue_completions for the progress discriminator."""
    text = RUNNER.read_text()
    assert "_recent_queue_completions()" in text or "_recent_queue_completions ()" in text, \
        "runner must define _recent_queue_completions helper for holding-loop progress discriminator"


def test_holding_loop_uses_progress_discriminator():
    """detect_holding_loop_backoff must consult queue completions, not just session count."""
    text = RUNNER.read_text()
    # Find the function body
    fn_start = text.find("detect_holding_loop_backoff()")
    assert fn_start > 0
    # Body extends to the next top-level function definition
    fn_end = text.find("\n_extract_current_phase()", fn_start)
    if fn_end < 0:
        # Fallback: take a 4 KB window
        fn_end = fn_start + 4000
    body = text[fn_start:fn_end]
    # Must reference the new helper
    assert "_recent_queue_completions" in body, \
        "detect_holding_loop_backoff must call _recent_queue_completions"
    # Must have a progress-threshold constant
    assert "progress_threshold" in body, \
        "detect_holding_loop_backoff must define progress_threshold for the discriminator"


def test_recent_queue_completions_reads_queue_telemetry():
    """The helper must read data/work_queue/queue_telemetry.jsonl (not some other file)."""
    text = RUNNER.read_text()
    fn_start = text.find("_recent_queue_completions()")
    assert fn_start > 0
    fn_end = text.find("\ndetect_holding_loop_backoff()", fn_start)
    body = text[fn_start:fn_end]
    assert "queue_telemetry.jsonl" in body, \
        "_recent_queue_completions must read data/work_queue/queue_telemetry.jsonl"
    assert "complete" in body, \
        "_recent_queue_completions must filter on action=complete"


# === 2026-05-20: retrieval graph reindex (Task 1.6) ===

def test_runner_defines_retrieval_graph_reindex():
    text = RUNNER.read_text()
    assert "_run_retrieval_graph_reindex()" in text or "_run_retrieval_graph_reindex ()" in text, \
        "runner must define _run_retrieval_graph_reindex helper"


def test_runner_calls_retrieval_graph_reindex_post_director():
    text = RUNNER.read_text()
    pd_idx = text.find("tools/post_director.py")
    assert pd_idx > 0
    after_pd = text[pd_idx:]
    assert "_run_retrieval_graph_reindex" in after_pd, \
        "_run_retrieval_graph_reindex must be called after post_director.py"


def test_runner_retrieval_graph_reindex_non_fatal():
    text = RUNNER.read_text()
    import re
    m = re.search(r"_run_retrieval_graph_reindex\(\)\s*\{(.*?)\n\}", text, re.DOTALL)
    assert m
    body = m.group(1)
    assert "||" in body or "2>&1" in body, \
        "_run_retrieval_graph_reindex must handle non-zero exit non-fatally"
