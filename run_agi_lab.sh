#!/bin/bash
# run_agi_lab.sh — Autonomous AGI lab runner
# Start once with: tmux new-session -d -s agi-lab './run_agi_lab.sh'
# Runs forever until VICTORY or CATASTROPHIC_STOP

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Pre-emptive context compaction per HIVE lab proven pattern: compact at 50% of
# context (vs Claude Code default ~90%). Smaller request sizes → more cycles per
# rate-limit window → less wasted work from context-full exits mid-phase.
# If Evaluator reports Director context-fidelity issues across 2+ consecutive
# phases, increase this value (70 is a safer fallback; unset to use the default).
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50

# Network-free model loading for the retrieval stack. Weights are cached locally
# and the lab is local-only by constraint, so the embedder/reranker must never
# reach out to the HF Hub (network calls add ~90s+ of latency/variance per cold
# load). Applies to the retrieval server, the reindex hooks, and the fallback. (audit F-3)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

LOG_DIR="data/infra/session_logs"
mkdir -p "$LOG_DIR"

AGENTS_FILE="data/agents/agents.json"
# DIRECTOR_PROMPT and agent registry are re-read per-session inside the loop so edits
# between cycles take effect on the next session without requiring a runner restart.

SESSION_NUM=0
CONSECUTIVE_FAILS=0
MAX_CONSECUTIVE_FAILS=5

# Probe function: returns 0 only if Claude actually responds (not rate limited)
probe_ready() {
    local RESULT
    RESULT=$(claude --print --model claude-opus-4-7 "Reply with exactly: READY" 2>&1)
    if echo "$RESULT" | grep -q "READY"; then
        return 0
    else
        return 1
    fi
}

# HIVE-pattern precise rate-limit wait: reads resetsAt timestamp from disk
# (written by stream_formatter.py parsing rate_limit_event messages) and sleeps
# exactly that long + 30s buffer. Falls back to 5-min probe loop if no timestamp.
RATE_LIMIT_RESETS_FILE="data/infra/rate_limit_resets_at"

wait_for_rate_limit_reset() {
    if [ -f "$RATE_LIMIT_RESETS_FILE" ]; then
        local RESETS_AT=$(cat "$RATE_LIMIT_RESETS_FILE" | tr -d '[:space:]')
        local NOW=$(date +%s)
        local WAIT_SECS=$((RESETS_AT - NOW + 30))  # +30s buffer for safety
        if [ $WAIT_SECS -gt 0 ] && [ $WAIT_SECS -lt 86400 ]; then
            local RESETS_HUMAN=$(date -r "$RESETS_AT" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$RESETS_AT")
            echo "Rate limit resets at $RESETS_HUMAN. Waiting ${WAIT_SECS}s (+30s buffer)." | tee -a "$SESSION_LOG"
            sleep "$WAIT_SECS"
        else
            echo "Reset timestamp implausible ($WAIT_SECS s from now); falling back to probe." | tee -a "$SESSION_LOG"
            _rate_limit_probe_fallback
        fi
    else
        echo "No reset timestamp; falling back to probe loop." | tee -a "$SESSION_LOG"
        _rate_limit_probe_fallback
    fi
    # After the wait, verify with a probe before resuming.
    while ! probe_ready; do
        echo "Post-wait probe still limited; retrying in 60s... ($(date))" | tee -a "$SESSION_LOG"
        sleep 60
    done
    echo "Rate limit cleared; resuming." | tee -a "$SESSION_LOG"
}

_rate_limit_probe_fallback() {
    sleep 300
    while ! probe_ready; do
        echo "Still limited; retrying in 5m... ($(date))" | tee -a "$SESSION_LOG"
        sleep 300
    done
}

write_session_brief() {
    # Compose data/memories/session_brief.md from live state.
    # Atomic write (tmp + mv) so partial writes can't be consumed.
    local mem_root="data/memories"
    local brief_tmp="${mem_root}/session_brief.md.tmp"
    local brief_final="${mem_root}/session_brief.md"

    # If memories not yet bootstrapped, fall back gracefully (pre-migration).
    if [ ! -d "${mem_root}" ]; then
        return 0
    fi

    mkdir -p "${mem_root}"
    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local rate_limit_state="clean"
    if [ -f "data/infra/rate_limit_resets_at" ]; then
        rate_limit_state="waiting_until_$(cat data/infra/rate_limit_resets_at)"
    fi

    local last_telemetry="(none)"
    if [ -f "data/infra/memory_telemetry.log" ]; then
        last_telemetry=$(tail -n1 "data/infra/memory_telemetry.log")
    fi

    local last_decision="(unknown)"
    if [ -f "${mem_root}/log.md" ]; then
        local raw_dec
        raw_dec=$(grep -m1 '^## D-' "${mem_root}/log.md" 2>/dev/null) || raw_dec=""
        if [ -n "${raw_dec}" ]; then
            last_decision=$(echo "${raw_dec}" | sed 's/^## //')
        else
            last_decision="(none)"
        fi
    fi

    # Substrate-awareness: cursor-based unread mechanism (2026-05-18 fix).
    # Previously age-based with 24h gate, which silently dropped entries when
    # the lab was mid-long-experiment (>24h between session spawns) — entries
    # could become invisible forever. Cursor approach: track which entries the
    # lab has acknowledged. Director MUST update cursor after reading.
    local substrate_cl="SUBSTRATE_CHANGELOG.md"
    local substrate_cursor="data/infra/substrate_changelog_cursor"
    local substrate_latest_ts=""
    local substrate_cursor_ts=""
    local substrate_unread_n=0
    local substrate_last_summary="(none)"
    if [ -f "${substrate_cl}" ]; then
        # Latest entry's timestamp (e.g. "2026-05-18 01:29")
        substrate_latest_ts=$(grep -E "^## [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}" "${substrate_cl}" 2>/dev/null \
            | tail -1 | awk '{print $2" "$3}')
        substrate_cursor_ts=$(cat "${substrate_cursor}" 2>/dev/null | head -1 | tr -d '\n')
        # Count entries strictly newer than cursor (string compare works on ISO-ish format)
        substrate_unread_n=$(awk -v cursor="${substrate_cursor_ts}" '
            /^## [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}/ {
                ts=$2" "$3
                if (ts > cursor) count++
            }
            END { print count+0 }
        ' "${substrate_cl}")
        substrate_last_summary=$(grep -E "^## 20" "${substrate_cl}" 2>/dev/null | tail -1 | cut -c1-140)
    fi

    {
        echo "---"
        echo "generated_at: ${now}"
        echo "runner_pid: $$"
        echo "---"
        echo "# Session brief"
        echo ""
        echo "Last decision: ${last_decision}"
        echo "Rate-limit: ${rate_limit_state}"
        echo "Memory last session: ${last_telemetry}"
        echo "Substrate changelog: ${substrate_unread_n} unread (latest: ${substrate_latest_ts:-none}, cursor: ${substrate_cursor_ts:-<none>})"
        echo "  → ${substrate_last_summary}"
        echo ""
        echo "First tool call MUST be:"
        echo "  memory.view session_brief.md  (you are reading it now)"
        echo "Then:"
        echo "  memory.view INDEX.md"
        echo "Then selective views per INDEX."
        echo "Then (conditional, per procedural Step 4):"
        echo "  if substrate unread > 0, Read SUBSTRATE_CHANGELOG.md, adapt, then advance the cursor:"
        echo "    echo '${substrate_latest_ts}' > data/infra/substrate_changelog_cursor"
    } > "${brief_tmp}"
    mv "${brief_tmp}" "${brief_final}"

    # Generate context_brief.md (deterministic, no LLM, ~50ms).
    # Director's procedural reads this in addition to session_brief.md.
    if [ -f tools/brief_assembler.py ]; then
        python3 tools/brief_assembler.py --out data/memories/context_brief.md \
            >>data/infra/brief_assembler.log 2>&1 || \
            echo "warning: brief_assembler failed (non-fatal); Director will fall back" \
                >>data/infra/brief_assembler.log
    fi
}

_mtime_secs() {
    # Cross-platform file mtime in epoch seconds. Returns "" if the file
    # doesn't exist. Used by _state_is_stable / _seconds_since_last_director.
    local p="$1"
    [ -e "$p" ] || { echo ""; return; }
    stat -f %m "$p" 2>/dev/null || stat -c %Y "$p" 2>/dev/null || echo ""
}

_seconds_since_last_director_session() {
    # Return epoch-secs since the newest session_*.log was modified.
    # The newest session log's mtime is when the last Director session
    # ended (run_agi_lab.sh tee's into it through the session). Returns
    # 999999 if no session log exists (cold start).
    local log_dir="$LOG_DIR"
    [ -d "$log_dir" ] || { echo 999999; return; }
    local newest
    newest=$(ls -t "$log_dir"/session_*.log 2>/dev/null | head -1)
    [ -n "$newest" ] || { echo 999999; return; }
    local mt now
    mt=$(_mtime_secs "$newest")
    [ -n "$mt" ] || { echo 999999; return; }
    now=$(date +%s)
    echo $((now - mt))
}

_state_is_stable() {
    # Returns 0 (stable) iff NO meaningful state has changed in trailing
    # STABLE_WINDOW_MIN minutes. Returns 1 (not stable) if anything changed.
    # Used by the main loop to skip Director sessions when there is nothing
    # for the Director to observe.
    #
    # Added 2026-04-28 (D-288) per operator request — long-run training
    # generates ~250 Director sessions across 6-day factorial, each costing
    # ~400K input tokens for read-only monitoring with no state change.
    # Skipping the cycle when state is genuinely unchanged saves the full
    # startup-read cost without breaking the audit trail (a Director session
    # that observed nothing produces no audit record worth preserving anyway).
    #
    # Signals checked (mtime newer than cutoff = NOT stable):
    #   - data/user_notes.md (operator nudge or directive)
    #   - data/runs/phase3_factorial_log/ (any orchestrator log activity)
    #   - data/checkpoints/phase3_factorial/run_index.json (cell state)
    #   - data/diagnostics/*.md (any new diagnostic file)
    #   - programs/<*>/USER_*.md (any new user-action file)
    #
    # Excluded (Director write-targets — refresh tautologically every session;
    # including them traps the loop into running back-to-back sessions because
    # log.md/current.md/session_exit.md just got written by the previous one):
    #   - data/memories/log.md
    #   - data/memories/current.md
    #   - data/session_exit.md
    # (Bug observed Apr 29 D-296+: post-D42→B42 transition the skip stopped
    # firing for 12+ consecutive sessions because every session refreshed
    # log.md inside the 30-min window. Fix: only check external signals.)
    local stable_window_min="${STABLE_WINDOW_MIN:-30}"
    local now cutoff
    now=$(date +%s)
    cutoff=$((now - stable_window_min * 60))

    local paths=(
        "data/user_notes.md"
        "data/runs/phase3_factorial_log"
        "data/checkpoints/phase3_factorial/run_index.json"
    )
    local p mt
    for p in "${paths[@]}"; do
        mt=$(_mtime_secs "$p")
        [ -n "$mt" ] && [ "$mt" -gt "$cutoff" ] && return 1
    done

    # Walk diagnostics dir if it exists (any file mtime within window = activity)
    if [ -d "data/diagnostics" ]; then
        if find "data/diagnostics" -type f -mmin -"$stable_window_min" 2>/dev/null \
                | grep -q .; then
            return 1
        fi
    fi

    # Walk programs/*/USER_*.md
    if find programs -maxdepth 2 -name "USER_*.md" \
            -mmin -"$stable_window_min" 2>/dev/null | grep -q .; then
        return 1
    fi

    return 0
}

_phase_just_ended() {
    # Returns 0 (true, NOT stable) iff training appears to have just ended
    # and the Director hasn't processed the completion yet. Wakes the
    # Director ONCE after a phase ends, even though _state_is_stable()
    # would otherwise skip (no recent file activity == "stable" by mtime).
    #
    # Added 2026-05-04: 12-cell factorial completed cleanly while runner
    # was rate-limited. After rate limit cleared, _state_is_stable() saw
    # no recent activity and skipped — completely missing that the phase
    # had ended. Director never woke for close-out work. This detector
    # closes that hole.
    #
    # Signals (all must hold to fire):
    #   1. Orchestrator + scale_experiment processes are dead (training
    #      really ended, not just paused between samples).
    #   2. data/checkpoints/phase3_factorial/run_index.json mtime is newer
    #      than data/memories/log.md mtime (orchestrator wrote completion
    #      state AFTER the Director's last real decision).
    #
    # Once the Director wakes and writes log.md, condition 2 flips and the
    # detector stops firing — exactly the "wake once" behavior we want.
    pgrep -f 'tools/run_phase3_factorial\.py|build/scale_experiment' \
        >/dev/null 2>&1 && return 1

    local idx_mt log_mt
    idx_mt=$(_mtime_secs "data/checkpoints/phase3_factorial/run_index.json")
    [ -n "$idx_mt" ] || return 1

    log_mt=$(_mtime_secs "data/memories/log.md")
    [ -z "$log_mt" ] && return 0

    [ "$idx_mt" -gt "$log_mt" ] && return 0
    return 1
}

_should_run_consolidator() {
    # Returns 0 (true, run consolidator) if any trigger holds:
    #   1) ≥6 hours since last consolidator run
    #   2) phase just ended (delegates to _phase_just_ended)
    #   3) ≥10 new D-N entries in log.md since last run
    local last_run_file="data/infra/consolidator.last_run"
    local now last_run hours_since
    now=$(date +%s)

    # No prior run = first time, fire it
    [ -f "$last_run_file" ] || { return 0; }

    # Trigger 1: ≥6 hours since last
    last_run=$(awk 'NR==1{print $1}' "$last_run_file" 2>/dev/null)
    [ -n "$last_run" ] || last_run=0
    hours_since=$(( (now - last_run) / 3600 ))
    [ "$hours_since" -ge 6 ] && return 0

    # Trigger 2: phase just ended
    _phase_just_ended && return 0

    # Trigger 3: ≥10 new D-N entries since last run
    if [ -f data/memories/log.md ]; then
        local last_dn_count cur_dn_count
        last_dn_count=$(awk 'NR==1{print $2}' "$last_run_file" 2>/dev/null)
        [ -n "$last_dn_count" ] || last_dn_count=0
        # Match both bare "D-N" and "### D-N" line starts (log format evolved over time).
        # grep -c emits the count to stdout AND exits 1 when zero matches; guard against
        # the "0\n0" double-emit by stripping non-digits and defaulting to 0.
        cur_dn_count=$(grep -cE '^(### )?D-[0-9]' data/memories/log.md 2>/dev/null | head -1 | tr -cd '0-9')
        [ -n "$cur_dn_count" ] || cur_dn_count=0
        if [ "$((cur_dn_count - last_dn_count))" -ge 10 ]; then
            return 0
        fi
    fi

    return 1
}

_is_deep_stable() {
    # Returns 0 (true) iff the lab has been quiet for ≥2 hours AND training
    # appears healthy (orchestrator alive OR run_index updated within last 6h).
    #
    # When true, the main loop uses a longer skip window (2h vs 30min) and
    # the heartbeat extends to 8h. This is "deep monitor mode" — Director
    # has nothing to observe, training is humming, no point waking it.
    #
    # Any external change (run_index update, user_notes nudge, diagnostic,
    # USER_*.md) immediately ends deep-stable on the next iteration's
    # _state_is_stable check, so the worst case is a 2-4h delay on a
    # non-time-sensitive event. Time-sensitive events (FATAL/ERROR in
    # orchestrator log) still update file mtimes within the window.
    STABLE_WINDOW_MIN=120 _state_is_stable || return 1

    # Training-active signal: orchestrator process alive
    if pgrep -f 'tools/run_phase3_factorial\.py|build/scale_experiment' \
            >/dev/null 2>&1; then
        return 0
    fi

    # Or recent training event (run_index updated within last 6h)
    local idx_mt now age
    idx_mt=$(_mtime_secs "data/checkpoints/phase3_factorial/run_index.json")
    [ -n "$idx_mt" ] || return 1
    now=$(date +%s)
    age=$((now - idx_mt))
    [ "$age" -lt 21600 ] && return 0

    return 1
}

_compute_dynamic_heartbeat() {
    # Returns the heartbeat window in seconds for THIS iteration.
    # - 2h (7200s) if phase just ended (catch up on close-out)
    # - 8h (28800s) if deep-stable (training healthy + quiet)
    # - 4h (14400s) default
    #
    # Echoed (not returned) so caller can capture: heartbeat=$(...)
    if _phase_just_ended; then
        echo 7200
        return
    fi
    if _is_deep_stable; then
        echo 28800
        return
    fi
    echo $((HEARTBEAT_INTERVAL_MIN * 60))
}

_queue_has_pending() {
    # Returns 0 (true, item available) iff data/work_queue/pending.jsonl
    # has at least one line. Uses tools/work_queue.py to ensure the file
    # is well-formed (just bash 'wc -l' would catch a corrupt JSONL too).
    [ -f data/work_queue/pending.jsonl ] || return 1
    [ -s data/work_queue/pending.jsonl ] || return 1
    return 0
}

_run_reclaim_stale() {
    # Reclaim stale claims (>60min old, no completion event). A claim with a
    # silent-died Director sits in claimed.jsonl forever otherwise. Cheap (~10ms).
    # Failure is non-fatal; runner continues regardless.
    if [ -f tools/work_queue.py ]; then
        python3 -c "
from tools.work_queue import reclaim_stale
n = reclaim_stale(timeout_min=60)
if n:
    import sys
    print(f'reclaim_stale: recovered {n} stale claim(s)', file=sys.stderr)
" >> data/infra/queue_scanner.log 2>&1 || \
            echo "reclaim_stale failed (non-fatal); will retry next iteration" \
                >> data/infra/queue_scanner.log
    fi
}

_run_queue_scanner() {
    # Idempotent scanner — populates queue from external state. Cheap (~50ms).
    # Failure is non-fatal; runner continues with whatever's in the queue.
    if [ -f tools/queue_scanner.py ]; then
        python3 tools/queue_scanner.py >> data/infra/queue_scanner.log 2>&1 || \
            echo "queue_scanner failed (non-fatal); will retry next iteration" \
                >> data/infra/queue_scanner.log
    fi
}

_run_lab_memory_reindex() {
    # Incremental semantic re-index of *.md files whose mtime > last ingest.
    # Only touched files get re-embedded; clean cases are near-instant. Hard 2-min
    # cap so a runaway run can't freeze the loop. Failure non-fatal.
    if [ -f tools/lab_memory.py ] && [ -x .venv/bin/python ]; then
        timeout 120 .venv/bin/python tools/lab_memory.py ingest --incremental \
            --roots data/memories programs data/agents data/engineering docs \
            >> data/infra/lab_memory.log 2>&1 || \
            echo "lab_memory reindex failed/timed-out (non-fatal); will retry next iteration" \
                >> data/infra/lab_memory.log
    fi
}

_run_retrieval_graph_reindex() {
    # Change-detection gate (audit #4): the corpus is tools/lab_memory.db; if it
    # hasn't been modified since BOTH indexes were last built, the ~96MB bm25 JSON
    # + the graph are already current — skip the expensive rebuild entirely.
    if [ -f tools/lab_memory.db ] && [ -f tools/lab_bm25.json ] && [ -f tools/lab_graph.db ] \
       && [ tools/lab_memory.db -ot tools/lab_bm25.json ] \
       && [ tools/lab_memory.db -ot tools/lab_graph.db ]; then
        return 0   # corpus unchanged since last index build
    fi
    # Rebuild the token graph (D-N, P-*, program, phase, role) from the corpus.
    # Idempotent — INSERT OR IGNORE on every node/edge. ~100s on current corpus.
    # Hard 180s cap (live rebuild ~102s; headroom for growth). Failure non-fatal.
    if [ -f tools/retrieval/graph.py ] && [ -x .venv/bin/python ]; then
        timeout 180 .venv/bin/python -m tools.retrieval.graph build \
            --db tools/lab_graph.db --root . \
            >> data/infra/retrieval_graph.log 2>&1 || \
            echo "retrieval_graph reindex failed/timed-out (non-fatal); will retry next iteration" \
                >> data/infra/retrieval_graph.log
    fi
    if [ -f tools/retrieval/bm25.py ] && [ -x .venv/bin/python ]; then
        timeout 60 .venv/bin/python -m tools.retrieval.bm25 build \
            --index tools/lab_bm25.json --lab-memory-db tools/lab_memory.db \
            >> data/infra/retrieval_bm25.log 2>&1 || \
            echo "bm25 reindex failed/timed-out (non-fatal)" \
                >> data/infra/retrieval_bm25.log
    fi
    # Tell the warm retrieval server (if running) to re-read the rebuilt indexes.
    # Non-fatal: the server also self-heals on index mtime change per query.
    if [ -f tools/retrieval/client.py ] && [ -x .venv/bin/python ]; then
        timeout 15 .venv/bin/python -m tools.retrieval.client reload \
            >> data/infra/retrieval_server.log 2>&1 || true
    fi
    return 0  # whole function is best-effort; never abort the runner under set -e
}

_start_retrieval_server() {
    # Start the persistent retrieval daemon once: it holds the embedder +
    # cross-encoder reranker + parsed BM25 index warm so per-query subprocesses
    # don't reload ~1GB of models (and don't pay the slow cold import) every
    # time. Reuse a live server if one already answers. Entirely non-fatal — on
    # any failure, lab_memory.py search transparently falls back to in-process
    # retrieval, so the lab never hangs or errors. (audit F-1)
    [ -f tools/retrieval/server.py ] && [ -x .venv/bin/python ] || return 0
    if .venv/bin/python -m tools.retrieval.client ping >/dev/null 2>&1; then
        echo "retrieval server already running; reusing $(date)" >> data/infra/retrieval_server.log
        return 0
    fi
    nohup .venv/bin/python -m tools.retrieval.server --repo . \
        >> data/infra/retrieval_server.log 2>&1 &
    echo $! > data/infra/retrieval.pid.runner
    local i
    for i in $(seq 1 25); do   # poll readiness up to ~25s (cold model load)
        if .venv/bin/python -m tools.retrieval.client ping >/dev/null 2>&1; then
            echo "retrieval server ready $(date)" >> data/infra/retrieval_server.log
            return 0
        fi
        sleep 1
    done
    echo "retrieval server slow/failed to warm $(date); queries use in-process fallback" \
        >> data/infra/retrieval_server.log
    return 0
}

_stop_retrieval_server() {
    [ -x .venv/bin/python ] || return 0
    # Graceful first: the daemon handles {cmd:shutdown} -> clean exit + unlink.
    .venv/bin/python -m tools.retrieval.client shutdown >/dev/null 2>&1 || true
    # Backstop kill — but only if the PID is actually our daemon (guard against
    # PID reuse from a stale pidfile). Clean up BOTH pidfiles unconditionally.
    local f pid
    for f in data/infra/retrieval.pid data/infra/retrieval.pid.runner; do
        if [ -f "$f" ]; then
            pid=$(cat "$f" 2>/dev/null || true)
            if [ -n "$pid" ] && ps -p "$pid" -o command= 2>/dev/null | grep -q 'tools.retrieval.server'; then
                kill "$pid" 2>/dev/null || true
            fi
            rm -f "$f"
        fi
    done
    return 0
}

_training_is_advancing() {
    # Discriminator for detect_holding_loop_backoff (added D-251, 2026-04-28).
    # Returns 0 (true) iff a long-running training process is alive AND its
    # stdout.log has been written in the last RUN_FRESH_MIN (default 30) minutes.
    # Used to suppress the holding-loop false-positive when the Director's
    # short sessions are CORRECT monitoring of a healthy training run, not the
    # D-202..D-225 stuck-waiting-for-user shape.
    #
    # Signals:
    #   1) pgrep -f matches one of: tools/run_long.py, tools/run_phase3_factorial.py,
    #      build/scale_experiment (the orchestrator + its subprocesses).
    #   2) glob data/runs/*/stdout.log (excluding phase3_factorial_log/*) and
    #      check mtime within trailing RUN_FRESH_MIN minutes.
    #
    # Returns 0 (true) when BOTH hold; 1 (false) otherwise.
    # Cheap: one pgrep + one find. No log parsing.
    local fresh_min="${RUN_FRESH_MIN:-30}"
    local runs_dir="${RUNS_DIR_OVERRIDE:-data/runs}"

    pgrep -f 'tools/run_long\.py|tools/run_phase3_factorial\.py|build/scale_experiment' \
        >/dev/null 2>&1 || return 1

    [ -d "$runs_dir" ] || return 1

    # Find any stdout.log in data/runs/<run>/ (one level deep) modified within
    # the last fresh_min minutes. Exclude phase3_factorial_log/ (the
    # orchestrator's own log dir, not a per-cell run dir).
    local fresh_count
    fresh_count=$(find "$runs_dir" -maxdepth 2 -name "stdout.log" -type f \
                       -mmin -"$fresh_min" 2>/dev/null \
                       | grep -v '/phase3_factorial_log/' \
                       | wc -l | tr -d ' ')
    fresh_count=${fresh_count:-0}
    [ "$fresh_count" -ge 1 ] || return 1
    return 0
}

_phase11_eval_is_advancing() {
    # Phase-11 analog of _training_is_advancing (added 2026-05-18 after D-360
    # surfaced the gap). During real-data scoring, the eval_harness_p11.py
    # orchestrator can run 12-24h while Director sessions correctly exit as
    # `no_op` heartbeats. Those no_op exits don't produce queue completions,
    # so the progress discriminator misfires. This helper detects the
    # legitimate "long eval in flight" case so the holding-loop alert is
    # suppressed in the same way as Phase-8 training.
    #
    # Signals (both required):
    #   1) pgrep matches tools/eval_harness_p11.py / eval-p11-launch / eval_model
    #   2) latest data/infra/eval_p11_launch_*.log mtime within last RUN_FRESH_MIN
    #      minutes (default 30) — confirms the eval is actually progressing.
    #
    # Returns 0 (true) when both hold; 1 (false) otherwise.
    local fresh_min="${RUN_FRESH_MIN:-30}"

    pgrep -f 'tools/eval_harness_p11\.py|eval-p11-launch|build/eval_model' \
        >/dev/null 2>&1 || return 1

    local fresh_count
    fresh_count=$(find data/infra -maxdepth 1 -name "eval_p11_launch_*.log" -type f \
                       -mmin -"$fresh_min" 2>/dev/null \
                       | wc -l | tr -d ' ')
    fresh_count=${fresh_count:-0}
    [ "$fresh_count" -ge 1 ] || return 1
    return 0
}

_recent_queue_completions() {
    # Count queue completions in last <window_min> minutes by reading
    # data/work_queue/queue_telemetry.jsonl. Used by detect_holding_loop_backoff
    # to distinguish "stuck cycling" (no progress) from "productively short
    # sessions" (small queue items being resolved quickly).
    #
    # Returns the count on stdout; 0 on any error.
    local window_min="${1:-120}"
    python3 -c "
import json
from datetime import datetime, timezone, timedelta
cutoff = (datetime.now(timezone.utc) - timedelta(minutes=$window_min)).timestamp()
n = 0
try:
    with open('data/work_queue/queue_telemetry.jsonl') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('action') != 'complete': continue
            ts = (rec.get('ts') or '').replace('Z', '+00:00')
            try:
                if datetime.fromisoformat(ts).timestamp() >= cutoff:
                    n += 1
            except ValueError:
                continue
    print(n)
except FileNotFoundError:
    print(0)
" 2>/dev/null || echo 0
}

detect_holding_loop_backoff() {
    # Detect when the Director is in a holding pattern (writing the same
    # ~17-KB monitor-only entry over and over without state changes) and
    # back off so we don't burn weekly token budget on no-op iterations.
    #
    # Triggered by D-202..D-210 incident (2026-04-25): Director sat in 16
    # consecutive holding sessions waiting for user input on
    # USER_GO_NOGO_DECISION.md, burning ~272K-1.5M tokens of pure waiting.
    # Director can't tell itself "stop running" — that's runner-scope.
    #
    # Signal: count short session logs (<8KB) in trailing 2h. 5+ short
    # sessions = holding pattern → exponential backoff (10 min × count,
    # capped at 4h). One real-work session resets the counter.
    #
    # D-251 amendment (2026-04-28): when a long-running training process is
    # alive AND its stdout.log is advancing (mtime within last 30 min), the
    # short Director sessions are CORRECT monitoring (template per D-247:
    # one pgrep + one tail + one ls). Suppress the alert in that case so
    # the runner doesn't escalate inter-session sleep during healthy P8
    # EXECUTE phases. Genuine-stuck path (no live training OR training
    # frozen >30 min) is unchanged. See _training_is_advancing().
    #
    # 2026-05-14 amendment: progress discriminator. Phase 1 (B′+C′+D′) makes
    # short sessions COMMON because the artifact_queue_projector enqueues
    # small fast items continuously. Counting raw short sessions alone now
    # produces false positives. New rule: if the queue completed >=3 items
    # in the same 2h window (real progress), suppress the alert regardless
    # of session-count. Only fire when sessions are short AND no queue
    # progress occurred — the actual "stuck cycling" pattern.
    #
    # Sets HOLDING_BACKOFF_SECS for the caller (0 = normal flow, >0 = sleep).
    local log_dir="$LOG_DIR"
    local short_threshold_bytes=8192     # <8KB session log = "did almost nothing"
    local window_min=120                 # 2h trailing window
    local trigger_count=5                # 5+ short sessions in window = looping
    local progress_threshold=3           # >=3 queue completions in window = productive
    local per_count_secs=600             # 10 min added per count past threshold
    local max_secs=14400                 # cap at 4h

    HOLDING_BACKOFF_SECS=0
    [ -d "$log_dir" ] || return 0

    # macOS find supports -size N c (bytes) and -mmin (modification minutes)
    local short_recent
    short_recent=$(find "$log_dir" -name "session_*.log" -type f \
                       -mmin -"$window_min" -size -"${short_threshold_bytes}c" \
                       2>/dev/null | wc -l | tr -d ' ')
    short_recent=${short_recent:-0}

    if [ "$short_recent" -ge "$trigger_count" ]; then
        # D-251: live-training discriminator. If a training run is alive and
        # advancing, the short sessions are correct monitoring → suppress.
        if _training_is_advancing; then
            return 0
        fi
        # 2026-05-18: Phase-11 eval discriminator. Same shape as D-251 but for
        # the real-data scoring orchestrator. D-360 surfaced the gap — no_op
        # heartbeat exits during long evals were tripping the holding alert
        # because no_op doesn't fire queue completions.
        if _phase11_eval_is_advancing; then
            return 0
        fi
        # 2026-05-14: progress discriminator. If the queue is making real
        # progress (>=3 completions in the same 2h window), the short
        # sessions are productive — suppress the alert.
        local recent_completions
        recent_completions=$(_recent_queue_completions "$window_min")
        if [ "${recent_completions:-0}" -ge "$progress_threshold" ]; then
            return 0
        fi
        local over=$((short_recent - trigger_count + 1))
        HOLDING_BACKOFF_SECS=$((over * per_count_secs))
        [ $HOLDING_BACKOFF_SECS -gt $max_secs ] && HOLDING_BACKOFF_SECS=$max_secs
    fi
    return 0
}

_extract_current_phase() {
    # Extract the current Phase number from data/memories/current.md (post-cutover)
    # or data/state.md (pre-cutover legacy). Returns the digit string (e.g. "3")
    # or "?" if not parseable. Used by log_memory_telemetry to tag the line so
    # evaluator and operator can attribute spend per phase.
    local current="data/memories/current.md"
    [ -f "$current" ] || current="data/state.md"
    [ -f "$current" ] || { echo "?"; return; }

    local phase
    # Look for "## Current Phase: ... Phase N ..." or "Phase N <state>"
    phase=$(grep -m1 -iE 'current phase|^## phase|phase ?[0-9]+' "$current" 2>/dev/null \
            | head -1 | grep -oiE 'phase ?[0-9]+' | head -1 | grep -oE '[0-9]+')
    echo "${phase:-?}"
}

log_memory_telemetry() {
    # Records per-session orientation + tier-size telemetry + per-phase tag.
    # Runs on every session boundary (pre- AND post-refactor). Pre-refactor,
    # data/memories/ does not exist; we still measure startup_kb from the
    # session log so we have a real baseline to compare against once the
    # refactor cuts over.
    #
    # Per-phase tag (added 2026-04-27): each line carries `phase=N` extracted
    # from current.md. Lets evaluator audit cumulative spend per phase across
    # sessions. NOT a budget enforcement — pure observability per the
    # token-budget discussion (we explicitly chose observability over
    # enforcement to avoid mid-synthesis cutoff hurting deep-reasoning
    # phases like Phase 5 design synthesis or P12 peer review).
    local telem="data/infra/memory_telemetry.log"
    local session_log="${1:-}"

    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # ---- Startup orientation read (pre + post refactor) ----
    # Parse the session log for Read tool calls before first >>> AGENT dispatch;
    # sum their file sizes. Uses tools/measure_startup_read.py if available.
    local startup_kb="n/a"
    if [ -n "${session_log}" ] && [ -f "${session_log}" ] && [ -x tools/measure_startup_read.py ]; then
        startup_kb=$(python3 tools/measure_startup_read.py "${session_log}" 2>/dev/null \
                     | grep -m1 'startup_kb=' | cut -d= -f2)
        startup_kb=${startup_kb:-"n/a"}
    fi

    # ---- Tier sizes (post-refactor only) ----
    local mem_root="data/memories"
    local hot_kb=0 log_kb=0 wiki_kb=0
    local tiers_available=0
    if [ -d "${mem_root}" ]; then
        tiers_available=1
        [ -f "${mem_root}/current.md" ] && hot_kb=$(( $(stat -f%z "${mem_root}/current.md" 2>/dev/null || echo 0) / 1024 ))
        [ -f "${mem_root}/log.md" ] && log_kb=$(( $(stat -f%z "${mem_root}/log.md" 2>/dev/null || echo 0) / 1024 ))
        wiki_kb=$(find "${mem_root}" -name "*.md" \
            ! -name "current.md" ! -name "log.md" ! -name "INDEX.md" \
            ! -name "history.md" ! -name "session_brief.md" \
            -exec stat -f%z {} \; 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')
        wiki_kb=${wiki_kb:-0}
    fi

    # ---- Per-phase tag + session-size token-proxy (added 2026-04-27) ----
    local phase
    phase=$(_extract_current_phase)

    local session_kb=0
    if [ -n "${session_log}" ] && [ -f "${session_log}" ]; then
        session_kb=$(( $(stat -f%z "${session_log}" 2>/dev/null || echo 0) / 1024 ))
    fi

    mkdir -p data/infra
    if [ "$tiers_available" -eq 1 ]; then
        echo "${now} | phase=${phase} | startup_kb=${startup_kb} | session_kb=${session_kb} | hot_kb=${hot_kb} | wiki_kb=${wiki_kb} | log_kb=${log_kb} | km_closeout=phase_close_only" \
            >> "${telem}"
    else
        echo "${now} | phase=${phase} | startup_kb=${startup_kb} | session_kb=${session_kb} | tiers=pre-refactor" \
            >> "${telem}"
    fi
}

watchdog_cli_hang() {
    # Background sentinel for Claude Code CLI post-result hang.
    # See: anthropics/claude-code#21099 — CLI fails to close stdout in piped
    # mode after emitting the final result event. We detect the hang by
    # watching for data/session_exit.md being freshly written while the
    # claude process is still alive, then SIGKILL after a grace period.
    #
    # Args: $1 = session start time (epoch seconds), $2 = session log path
    local session_start="$1"
    local session_log="$2"
    local poll_interval=3
    local grace_seconds=30   # hard ceiling: kill this long after session_exit.md no matter what
    local idle_seconds=10    # primary: kill this long after the Director's last output (log goes quiet)
    local our_cpid_pattern='claude.*--print.*data/agents/agents.json'

    while true; do
        sleep "$poll_interval"

        # Find our claude process. Multiple claude instances could exist
        # (the user's own interactive claude, for example), so match against
        # our specific invocation signature.
        local cpid
        cpid=$(pgrep -f "$our_cpid_pattern" 2>/dev/null | head -1)
        if [ -z "$cpid" ]; then
            # Claude exited normally. Watchdog done.
            return 0
        fi

        # Claude is still running. Is session_exit.md fresh?
        if [ -f data/session_exit.md ]; then
            local exit_mtime
            exit_mtime=$(stat -f%m data/session_exit.md 2>/dev/null || echo 0)
            if [ "$exit_mtime" -gt "$session_start" ]; then
                # Fresh exit file written. The Director's deliverable is on disk;
                # what remains is the #21099 hang (CLI alive but not closing stdout).
                # The final `result` event does NOT reliably flush through the pipe
                # before the hang, so we can't key off it. Instead detect "done" via
                # output silence: once the session log stops growing for idle_seconds,
                # the Director has stopped producing output and the remaining aliveness
                # IS the hang. Hard 30s ceiling as fallback. Killing during a post-exit
                # self-check is safe — session_exit.md is already written and
                # post_director re-runs the same validators, so no work is lost.
                local now exit_age log_mtime log_idle
                now=$(date +%s)
                exit_age=$(( now - exit_mtime ))
                log_mtime=$(stat -f%m "$session_log" 2>/dev/null || echo "$now")
                log_idle=$(( now - log_mtime ))
                if [ "$log_idle" -ge "$idle_seconds" ] || [ "$exit_age" -ge "$grace_seconds" ]; then
                    local trig="output idle ${log_idle}s"
                    [ "$exit_age" -ge "$grace_seconds" ] && trig="ceiling ${exit_age}s"
                    echo "=== $(date) WATCHDOG: CLI hang (#21099) — session_exit.md written ${exit_age}s ago, ${trig}, claude pid=$cpid still alive. SIGKILL. ===" \
                        | tee -a "$session_log"
                    kill -9 "$cpid" 2>/dev/null || true
                    # Slightly wait so the pipe chain tears down before we return
                    sleep 2
                    return 0
                fi
            fi
        fi
    done
}

echo "================================================"
echo "  AGI LAB — Starting Autonomous Operation"
echo "  Target: Beat Opus 4.6 on all benchmarks"
echo "  Hardware: M3 Pro, 18GB, local only"
echo "  $(date)"
echo "================================================"

STABLE_SKIP_LOG="data/infra/stable_skip.log"
HEARTBEAT_INTERVAL_MIN="${HEARTBEAT_INTERVAL_MIN:-240}"
STABLE_WINDOW_MIN="${STABLE_WINDOW_MIN:-30}"

# Warm the persistent retrieval daemon once; reap it on exit. INT/TERM re-raise
# via exit so Ctrl-C actually stops the runner.
_start_retrieval_server || true
trap '_stop_retrieval_server || true' EXIT
trap 'exit 130' INT TERM

while true; do
    # Pre-spawn gates (added 2026-05-22 after D-580..D-606 incident).
    # The post-session operator_review check at ~line 1036 fires too late —
    # each iteration still spawns a full Director session before checking.
    # And the existing wait_for_rate_limit_reset() has a 24h sanity cap that
    # falls back to a 5-min poll loop for longer windows, causing the runner
    # to spin ~2400 no-op Director sessions across a 5.74d weekly-budget reset.
    # These pre-spawn gates skip Director invocation entirely.

    # Gate 1: rate-limit reset window. If the saved reset timestamp is in the
    # future, sleep until then in 1h chunks (clean interrupt path), then clear
    # the file so a stale value doesn't gate forever.
    if [ -f "data/infra/rate_limit_resets_at" ]; then
        rl_reset_at_raw=$(cat data/infra/rate_limit_resets_at 2>/dev/null | tr -d '[:space:]')
        if [[ "$rl_reset_at_raw" =~ ^[0-9]+$ ]]; then
            rl_reset_at=$rl_reset_at_raw
            rl_now=$(date +%s)
            if [ "$rl_reset_at" -gt "$rl_now" ]; then
                rl_wait=$((rl_reset_at - rl_now + 30))
                rl_human=$(date -r "$rl_reset_at" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$rl_reset_at")
                rl_hours=$((rl_wait / 3600))
                echo "$(date '+%Y-%m-%d %H:%M:%S') [RATE-LIMIT-GATE] reset at ${rl_human} (${rl_hours}h from now). Sleeping ${rl_wait}s in 1h chunks." \
                    | tee -a data/infra/rate_limit_gate.log
                while [ "$(date +%s)" -lt "$rl_reset_at" ]; do
                    chunk=3600
                    remaining=$((rl_reset_at - $(date +%s)))
                    [ $remaining -lt $chunk ] && chunk=$remaining
                    [ $chunk -le 0 ] && break
                    sleep "$chunk"
                done
                echo "$(date '+%Y-%m-%d %H:%M:%S') [RATE-LIMIT-GATE] reset window cleared; resuming." \
                    | tee -a data/infra/rate_limit_gate.log
                # Clear the file so a stale value doesn't trigger again next iteration.
                rm -f data/infra/rate_limit_resets_at
            elif [ "$rl_reset_at" -le "$rl_now" ]; then
                # Stale reset timestamp (already in past). Remove so next probe is clean.
                rm -f data/infra/rate_limit_resets_at
            fi
        fi
    fi

    # Gate 2: operator review pending. Skip Director spawn entirely (don't burn
    # a session that will exit as no_op when it sees the file). Operator must
    # remove the file to resume.
    if [ -f data/operator_review_pending.md ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [OPERATOR-REVIEW-GATE] operator_review_pending.md present; sleeping 5m before re-check (no Director spawn)." \
            | tee -a data/infra/operator_review.log
        sleep 300
        continue
    fi

    # Consolidator trigger (Stage 6, async memory maintenance). Runs BEFORE
    # the skip-when-stable check so consolidator fires even on quiet ticks.
    if _should_run_consolidator; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [CONSOLIDATOR] Triggering async consolidator." \
            | tee -a data/infra/consolidator.log
        mkdir -p data/infra
        # Async dispatch via Claude CLI with the consolidator agent registered.
        # Failure is non-fatal — Director will still run, consolidator retries
        # on next iteration.
        if [ -f data/agents/agents.json ]; then
            CONSOLIDATOR_PROMPT=$(jq -r '.consolidator.prompt' data/agents/agents.json 2>/dev/null)
            if [ -n "$CONSOLIDATOR_PROMPT" ] && [ "$CONSOLIDATOR_PROMPT" != "null" ]; then
                {
                    claude --print --model claude-haiku-4-5 --effort high \
                        --dangerously-skip-permissions \
                        "$CONSOLIDATOR_PROMPT" \
                        >> data/infra/consolidator.log 2>&1 || \
                        echo "consolidator dispatch failed (non-fatal); will retry next iteration" \
                            >> data/infra/consolidator.log
                    # Update last_run timestamp + D-N count regardless of result
                    # (idempotent — no harm in skipping next 6h if dispatch failed)
                    local dn_count
                    dn_count=$(grep -c '^D-' data/memories/log.md 2>/dev/null || echo 0)
                    echo "$(date +%s) $dn_count" > data/infra/consolidator.last_run
                } &
            fi
        fi
    fi

    # Weekly rollup of dispatch telemetry (Stage 7). Idempotent — overwrites
    # data/infra/dispatch_rollup.md. Cheap (~50ms parse + write). Runs every
    # iteration; meaningful only after enough dispatches accumulate.
    if [ -f tools/dispatch_rollup.py ] && [ -f data/infra/dispatch_telemetry.jsonl ]; then
        python3 tools/dispatch_rollup.py >> data/infra/dispatch_rollup.log 2>&1 || true
    fi

    # PI calibration rollup (Stage P3). Same pattern: idempotent, cheap,
    # meaningful once outcomes are scored. Per-confidence-band hit rates.
    if [ -f tools/calibration_rollup.py ] && [ -f data/infra/calibration_telemetry.jsonl ]; then
        python3 tools/calibration_rollup.py >> data/infra/calibration_rollup.log 2>&1 || true
    fi

    # Work queue rollup (Q6). Aggregates queue_telemetry.jsonl into readable
    # markdown report. Per-item-type counts + claim→complete latency.
    # Idempotent, cheap (~50ms). Runs every iteration; meaningful after items
    # are processed. Non-fatal on telemetry absence.
    if [ -f tools/queue_rollup.py ]; then
        python3 tools/queue_rollup.py >> data/infra/queue_rollup.log 2>&1 || true
    fi

    # Reclaim stale claims (RO-CO Stage R6). Recovers claims orphaned by
    # silent-died Director sessions before the scanner runs — so any orphan
    # detector + post_director close-out sees a clean state. Cheap, idempotent.
    _run_reclaim_stale

    # Work queue scan (Q2). Populate queue from external state every iteration.
    # Cheap (~50ms). Idempotent. Items deduped by deterministic ID.
    _run_queue_scanner

    # Schema/contract violation redispatch handling (L1.3 + L3.3).
    # If post_director.py from the previous iteration flagged a violation,
    # spawn Director immediately with DIRECTOR_REDISPATCH_REASON env var set.
    # FORCE_SPAWN=1 bypasses BOTH the queue-empty check AND skip-when-stable
    # so the correction-prompt path is never silently lost.
    REDISPATCH_FLAG="data/session_exit_redispatch_pending.flag"
    FORCE_SPAWN=0
    if [ -f "$REDISPATCH_FLAG" ]; then
        REDISPATCH_REASON=$(grep '^reason=' "$REDISPATCH_FLAG" 2>/dev/null | cut -d= -f2- || echo "unknown")
        REDISPATCH_SESSION=$(grep '^session_id=' "$REDISPATCH_FLAG" 2>/dev/null | cut -d= -f2- || echo "unknown")
        rm -f "$REDISPATCH_FLAG"
        echo "$(date '+%Y-%m-%d %H:%M:%S') [SCHEMA-VIOLATION-REDISPATCH] reason=$REDISPATCH_REASON prev_session=$REDISPATCH_SESSION; FORCING Director spawn with correction prompt" \
            | tee -a data/infra/schema_violation_redispatch.log
        export DIRECTOR_REDISPATCH_REASON="schema_violation_previous_session_exit"
        FORCE_SPAWN=1
    else
        unset DIRECTOR_REDISPATCH_REASON
    fi

    # Work queue dispatch (Q3). If queue has a pending item OR FORCE_SPAWN is set,
    # spawn Director — bypassing skip-when-stable. FORCE_SPAWN is set when the
    # redispatch flag was consumed (schema/contract violation from prior session);
    # without it, an empty queue + stable state would silently skip the correction.
    if _queue_has_pending || [ "$FORCE_SPAWN" -eq 1 ]; then
        if [ "$FORCE_SPAWN" -eq 1 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') [FORCE-SPAWN] redispatch flag honored; bypassing skip-when-stable" \
                | tee -a data/infra/schema_violation_redispatch.log
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') [QUEUE-DISPATCH] pending queue items detected; spawning Director." \
                | tee -a data/infra/queue_scanner.log
        fi
        # Fall through to the regular session spawn path. Director's
        # procedural (Stage Q4) reads the queue and claims the top item.
    else
        # Queue empty — fall through to skip-when-stable check.

        # Skip-when-stable check (D-288, 2026-04-28). Before spinning up a
        # Director session (~400K input tokens at peak), check whether there
        # is anything new for the Director to observe. If state is unchanged
        # for the trailing STABLE_WINDOW_MIN AND the last Director session
        # ran less than HEARTBEAT_INTERVAL_MIN ago, skip the cycle: sleep for
        # one stable-window then re-check. This eliminates the read-only
        # monitoring overhead during long-run training when the answer is
        # always "still healthy."
        secs_since_last=$(_seconds_since_last_director_session)
        heartbeat_sec=$(_compute_dynamic_heartbeat)
        if [ "$secs_since_last" -lt "$heartbeat_sec" ] && _state_is_stable && ! _phase_just_ended; then
            # Tuning Pass C: deep-stable extends the skip window from 30min → 2h.
            # Saves Director sessions on quiet training plateaus where there is
            # genuinely nothing for Director to observe between cell completions.
            if _is_deep_stable; then
                skip_window_sec=7200
                tier_tag="deep-stable"
            else
                skip_window_sec=$((STABLE_WINDOW_MIN * 60))
                tier_tag="stable"
            fi
            skip_msg="$(date '+%Y-%m-%d %H:%M:%S') [STABLE-SKIP/${tier_tag}] last director ${secs_since_last}s ago < ${heartbeat_sec}s heartbeat. Sleeping $((skip_window_sec / 60))min."
            echo "$skip_msg" | tee -a "$STABLE_SKIP_LOG" >/dev/null
            # Brief tmux-visible note (single line) so the operator sees the
            # runner is alive and skipping, not stuck.
            echo "$skip_msg"
            sleep "$skip_window_sec"
            continue
        fi
    fi

    SESSION_NUM=$((SESSION_NUM + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SESSION_LOG="$LOG_DIR/session_${SESSION_NUM}_${TIMESTAMP}.log"
    SESSION_START_TIME=$(date +%s)

    echo "" | tee -a "$SESSION_LOG"
    echo "=== Session $SESSION_NUM starting at $(date) ===" | tee -a "$SESSION_LOG"

    # Re-read the Director prompt each session so between-cycle edits take effect.
    # Canonical path is data/agents/director/procedural.md (same convention as every
    # other role). The legacy data/agents/director_prompt.md location is preserved
    # as a symlink for backwards-compat; prefer the canonical path for edits.
    write_session_brief

    # Substrate marker: stamp git SHA + changelog mtime at session start so
    # cost_rollup / dispatch_rollup deltas can be joined to substrate version.
    # Operator-authored changes land in SUBSTRATE_CHANGELOG.md; this row
    # records WHICH version each session ran against.
    SUBSTRATE_MARKERS="data/infra/substrate_markers.tsv"
    if [ ! -f "$SUBSTRATE_MARKERS" ]; then
        mkdir -p data/infra
        printf "timestamp\tsession\tgit_sha\tchangelog_mtime\tchangelog_last_summary\n" > "$SUBSTRATE_MARKERS"
    fi
    SUB_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")
    SUB_CHANGELOG="SUBSTRATE_CHANGELOG.md"
    if [ -f "$SUB_CHANGELOG" ]; then
        SUB_CL_MTIME=$(stat -f "%Sm" -t "%Y-%m-%dT%H:%M:%S" "$SUB_CHANGELOG" 2>/dev/null || stat -c "%y" "$SUB_CHANGELOG" 2>/dev/null | cut -c1-19 | tr ' ' 'T' || echo "unknown")
        SUB_CL_LAST=$(grep -E "^## 20" "$SUB_CHANGELOG" 2>/dev/null | tail -1 | sed 's/|/ \| /g' | cut -c1-160 | tr '\t' ' ')
    else
        SUB_CL_MTIME="absent"
        SUB_CL_LAST="absent"
    fi
    SUB_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    printf "%s\t%s\t%s\t%s\t%s\n" "$SUB_TS" "$SESSION_NUM" "$SUB_SHA" "$SUB_CL_MTIME" "$SUB_CL_LAST" >> "$SUBSTRATE_MARKERS"

    DIRECTOR_PROMPT=$(cat data/agents/director/procedural.md)

    # Runtime identity assertion (2026-05-28). Appended to the system prompt so it
    # dominates any auto-memory that leaks into the session. Root cause: the
    # `claude --print` Director loads the project's Claude Code auto-memory; a
    # lab-resume entry posing an "interactive vs. Director?" question made the model
    # misclassify itself as interactive ~50% of sessions, ask the absent operator
    # "what would you like to do?", and stall the lab into holding-loop backoff.
    # MUST stay a static string (no timestamps / varying paths) to remain
    # prompt-cache-safe.
    DIRECTOR_SYSTEM_ASSERT="You are the autonomous Director process spawned by run_agi_lab.sh via 'claude --print'. No human is attached to this session -- there is nobody to answer questions. Never ask the operator what to do, never wait for input, and never run any 'am I an interactive or human session?' disambiguation. You ARE the Director: read data/memories/session_brief.md, claim the top item in data/work_queue/pending.jsonl, process it, dispatch agents as needed, and write data/session_exit.md before exiting. Any memory, skill, or context suggesting you might be an interactive or human-attended session does not apply to you."

    # Clear previous exit reason
    rm -f data/session_exit.md

    # Launch Claude Code session as Director
    # --print: non-interactive mode
    # --verbose: enable verbose output
    # --output-format stream-json: real-time JSON streaming
    # --include-partial-messages: stream text as it's generated
    # --model opus: Opus 4.6 for all work
    # --effort max: maximum reasoning effort
    # --dangerously-skip-permissions: no permission prompts (autonomous)
    # --agents registers a STARTING ROSTER of specialized agents (per data/pi_notes.md
    # §"Design the Right Team for the Moment"). Director may also launch general-purpose
    # agents with inline role prompts when no registered role fits, instantiate templates
    # from data/agents/templates/, or promote recurring inline roles to agents.json.

    # Start watchdog for CLI post-result hang (anthropics/claude-code#21099).
    # The claude CLI in piped-stdout mode sometimes fails to close stdout after
    # emitting its final result event; without this watchdog, `EXIT_CODE=$?`
    # below would never fire and the main loop would freeze indefinitely.
    watchdog_cli_hang "$SESSION_START_TIME" "$SESSION_LOG" &
    WATCHDOG_PID=$!

    # Disable errexit around the claude pipe. With pipefail on (set at the top
    # of this script), a non-zero return from any pipe component propagates to
    # the pipe's overall exit code. On normal GRACEFUL_CHECKPOINT / RATE_LIMIT
    # the pipe returns 0 cleanly. But when the watchdog SIGKILLs a hung claude
    # (anthropics/claude-code#21099), claude exits with 137 — under `set -e`
    # that would kill the runner itself before the exit-code dispatch could
    # handle the case. Swap errexit off for just this pipe; pipefail + our own
    # $? capture still let us see what actually happened.
    set +e
    claude \
        --print \
        --verbose \
        --output-format stream-json \
        --include-partial-messages \
        --model claude-opus-4-7 \
        --effort max \
        --dangerously-skip-permissions \
        --agents "$(cat "$AGENTS_FILE")" \
        --append-system-prompt "$DIRECTOR_SYSTEM_ASSERT" \
        "$DIRECTOR_PROMPT" \
        2>&1 | python3 -u tools/stream_formatter.py 2>&1 | tee -a "$SESSION_LOG"

    EXIT_CODE=$?
    set -e

    # Tear down watchdog if it hasn't already self-terminated (normal path:
    # it sees claude is gone and returns 0 on its own).
    if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi

    echo "=== Session $SESSION_NUM exited with code $EXIT_CODE at $(date) ===" | tee -a "$SESSION_LOG"

    # Log post-session memory telemetry (runs on every iteration regardless of
    # exit-code dispatch branch; no-op if data/memories/ does not exist yet).
    log_memory_telemetry "$SESSION_LOG"

    # Check for victory (check both new and legacy state files, case-insensitive)
    if grep -qi "status.*VICTORY" data/state.md data/session_state.md 2>/dev/null; then
        echo "================================================" | tee -a "$SESSION_LOG"
        echo "  AGI LAB: VICTORY — All benchmarks surpassed!" | tee -a "$SESSION_LOG"
        echo "  Total sessions: $SESSION_NUM" | tee -a "$SESSION_LOG"
        echo "  Check data/eval/scorecard.md for final scores" | tee -a "$SESSION_LOG"
        echo "================================================" | tee -a "$SESSION_LOG"
        osascript -e 'display notification "All benchmarks surpassed!" with title "AGI Lab: VICTORY"' 2>/dev/null || true
        break
    fi

    # Check for catastrophic stop
    if grep -qi "status.*CATASTROPHIC" data/state.md data/session_state.md 2>/dev/null; then
        echo "================================================" | tee -a "$SESSION_LOG"
        echo "  AGI LAB: STOPPED — Catastrophic constraint hit" | tee -a "$SESSION_LOG"
        echo "  Check data/decisions_recent.md for details" | tee -a "$SESSION_LOG"
        echo "================================================" | tee -a "$SESSION_LOG"
        osascript -e 'display notification "Catastrophic constraint hit. Human review needed." with title "AGI Lab: STOPPED"' 2>/dev/null || true
        break
    fi

    # Runner-owned close-out (RO-CO, Stage R6 / Task 16). Reads
    # data/session_exit.md and applies log/current/queue mutations from the
    # structured JSON block (if Director wrote one). Falls back to silent-death
    # recovery when session_exit.md is absent. Errors are logged to
    # data/infra/post_director_telemetry.jsonl, not raised — runner stays alive
    # if the finalizer crashes. Existing reason: line parsing below continues
    # to work alongside.
    if [ -f tools/post_director.py ]; then
        python3 tools/post_director.py >> data/infra/post_director.log 2>&1 || \
            echo "post_director.py failed (non-fatal); see data/infra/post_director.log" \
                >> data/infra/post_director.log
    fi

    # Semantic memory: keep lab_memory.db fresh with this session's outputs so
    # agents can semantic-search against just-written files in the next iteration.
    _run_lab_memory_reindex
    _run_retrieval_graph_reindex

    # Operator review check (L3.3 / D′): if Director hit redispatch ceiling, halt redispatch.
    # NOTE: macOS notification is sent by post_director.py (with full session_id + count
    # context) — we do NOT send a second one here to avoid training the operator to dismiss
    # duplicate alerts. Log line kept for audit trail.
    if [ -f data/operator_review_pending.md ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [OPERATOR-REVIEW] Director redispatch ceiling exceeded; awaiting operator action" \
            | tee -a data/infra/operator_review.log
        # Skip Director spawn until operator clears the file
        sleep 60
        continue
    fi

    # Determine exit reason
    EXIT_REASON=$(grep "reason:" data/session_exit.md 2>/dev/null | awk '{print $2}' || echo "UNKNOWN")

    # Detect rate limit from session output (hard kill case — no exit file written)
    if tail -5 "$SESSION_LOG" 2>/dev/null | grep -qi "hit your limit\|rate limit\|resets.*am"; then
        EXIT_REASON="RATE_LIMIT"
        echo "Detected rate limit from session output." | tee -a "$SESSION_LOG"
    fi

    # Structural teeth for Evaluator: if evaluator reported FAIL and Director did not
    # acknowledge FAIL items in session_exit.md, reject GRACEFUL_CHECKPOINT and force
    # immediate re-entry so the Director addresses the FAIL on the next cycle.
    if [ "$EXIT_REASON" = "GRACEFUL_CHECKPOINT" ] && [ -f data/evaluator_report.md ]; then
        if grep -qiE "^(Overall|verdict)[:]*[[:space:]]*FAIL\b" data/evaluator_report.md 2>/dev/null; then
            if ! grep -qiE "evaluator|unresolved|acknowledg" data/session_exit.md 2>/dev/null; then
                echo "================================================" | tee -a "$SESSION_LOG"
                echo "  BLOCKED: Evaluator reported FAIL and session_exit.md did not" | tee -a "$SESSION_LOG"
                echo "  acknowledge the FAIL items. Forcing immediate re-entry." | tee -a "$SESSION_LOG"
                echo "================================================" | tee -a "$SESSION_LOG"
                EXIT_REASON="EVALUATOR_FAIL_UNADDRESSED"
                osascript -e 'display notification "Evaluator FAIL not addressed. Director re-entering to fix." with title "AGI Lab: Eval Block"' 2>/dev/null || true
            fi
        fi
    fi

    # Structural teeth for signature forgery (D-110): before accepting GRACEFUL_CHECKPOINT,
    # run verify_signatures.py. Any signature in program docs / state.md / decisions without
    # a matching episodic record is a forgery. On detection, force re-entry with
    # SIGNATURE_FORGERY_UNADDRESSED so Director must remediate (dispatch real agent, append
    # D-N REMEDIATION appendix to offending docs, update accountability_ledger.md).
    if [ "$EXIT_REASON" = "GRACEFUL_CHECKPOINT" ] && [ -x tools/verify_signatures.py ]; then
        if ! source .venv/bin/activate && python3 tools/verify_signatures.py >> "$SESSION_LOG" 2>&1; then
            echo "================================================" | tee -a "$SESSION_LOG"
            echo "  BLOCKED: verify_signatures.py detected forged signatures." | tee -a "$SESSION_LOG"
            echo "  Director must dispatch real agent(s), update accountability_ledger.md," | tee -a "$SESSION_LOG"
            echo "  and remediate offending documents. Forcing immediate re-entry." | tee -a "$SESSION_LOG"
            echo "================================================" | tee -a "$SESSION_LOG"
            EXIT_REASON="SIGNATURE_FORGERY_UNADDRESSED"
            osascript -e 'display notification "Forged signature detected. Director re-entering to remediate." with title "AGI Lab: Forgery Block"' 2>/dev/null || true
        fi
    fi

    SESSION_DURATION=$(( $(date +%s) - SESSION_START_TIME ))

    # Intra-session diminishing-returns detector (D-233, 2026-04-27 — Lab
    # Weakness #5 mitigation). Within-session analog of the holding-loop
    # detector below. Compares session-log size to the size of durable
    # changes made (data/ + programs/ git diff). High log + low diff =
    # the agent generated tokens without producing artifacts. Warning
    # accumulates in data/diagnostics/low_productivity_sessions.md for the
    # next Director session to read on startup.
    if [ -x tools/session_productivity.py ] || [ -f tools/session_productivity.py ]; then
        python3 tools/session_productivity.py "$SESSION_LOG" "$SESSION_NUM" \
            2>/dev/null | tee -a "$SESSION_LOG" || true
    fi

    case "$EXIT_REASON" in
        CONTEXT_FULL)
            echo "Context full. Restarting immediately..." | tee -a "$SESSION_LOG"
            CONSECUTIVE_FAILS=0
            sleep 5
            ;;
        GRACEFUL_CHECKPOINT)
            echo "Cycle checkpointed. Starting next cycle..." | tee -a "$SESSION_LOG"
            CONSECUTIVE_FAILS=0
            sleep 5
            ;;
        EVALUATOR_FAIL_UNADDRESSED)
            echo "Evaluator FAIL unaddressed. Re-entering immediately to force resolution..." | tee -a "$SESSION_LOG"
            CONSECUTIVE_FAILS=0
            sleep 5
            ;;
        SIGNATURE_FORGERY_UNADDRESSED)
            echo "Signature forgery detected. Re-entering immediately to force remediation..." | tee -a "$SESSION_LOG"
            CONSECUTIVE_FAILS=0
            sleep 5
            ;;
        RATE_LIMIT)
            echo "Rate limited. Invoking precise wait..." | tee -a "$SESSION_LOG"
            CONSECUTIVE_FAILS=0
            osascript -e 'display notification "Rate limit hit. Waiting for reset." with title "AGI Lab: Paused"' 2>/dev/null || true
            wait_for_rate_limit_reset
            ;;
        *)
            # Unknown exit — use duration to decide
            if [ $SESSION_DURATION -lt 10 ]; then
                # Crashed almost immediately — something is wrong
                CONSECUTIVE_FAILS=$((CONSECUTIVE_FAILS + 1))
                echo "Quick crash (${SESSION_DURATION}s). Consecutive fails: $CONSECUTIVE_FAILS/$MAX_CONSECUTIVE_FAILS" | tee -a "$SESSION_LOG"
                if [ $CONSECUTIVE_FAILS -ge $MAX_CONSECUTIVE_FAILS ]; then
                    echo "=== TOO MANY CONSECUTIVE FAILURES. STOPPING. ===" | tee -a "$SESSION_LOG"
                    osascript -e 'display notification "Too many crashes. Lab stopped." with title "AGI Lab: ERROR"' 2>/dev/null || true
                    break
                fi
                sleep 120
            elif [ $SESSION_DURATION -gt 60 ]; then
                # Ran for a while — probably did real work, just didn't write exit file
                echo "Session ran ${SESSION_DURATION}s without exit file. Restarting..." | tee -a "$SESSION_LOG"
                CONSECUTIVE_FAILS=0
                sleep 5
            else
                # Short but not instant — probe to be safe
                echo "Short session (${SESSION_DURATION}s). Probing readiness..." | tee -a "$SESSION_LOG"
                CONSECUTIVE_FAILS=0
                sleep 30
                while true; do
                    if probe_ready; then
                        echo "Ready to resume." | tee -a "$SESSION_LOG"
                        break
                    fi
                    echo "Not ready. Retrying in 5m... ($(date))" | tee -a "$SESSION_LOG"
                    sleep 300
                done
            fi
            ;;
    esac

    # Holding-loop guard (post D-202..D-210 incident, 2026-04-25). Runs after
    # the per-exit-reason dispatch so it can override the short sleep above
    # when we detect the Director is in a "no-op iteration" pattern. Burns
    # zero tokens and zero compute — pure idle wait until state changes.
    detect_holding_loop_backoff
    if [ "${HOLDING_BACKOFF_SECS:-0}" -gt 0 ]; then
        echo "[HOLDING-LOOP] $(date) — detected no-op iteration pattern (5+ short sessions in last 2h). Backing off ${HOLDING_BACKOFF_SECS}s before next iteration. State unchanged → no point spinning. User intervention will break the loop." | tee -a "$SESSION_LOG"
        osascript -e "display notification \"Director in holding loop. Backing off $((HOLDING_BACKOFF_SECS / 60)) min — likely waiting for user input.\" with title \"AGI Lab: Holding\"" 2>/dev/null || true
        sleep "$HOLDING_BACKOFF_SECS"
    fi
done

echo "=== AGI Lab stopped after $SESSION_NUM sessions at $(date) ===" | tee -a "$SESSION_LOG"
