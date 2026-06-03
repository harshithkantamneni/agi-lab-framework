#!/bin/bash
# tools/restart_when_idle.sh
#
# Polls every POLL_INTERVAL seconds waiting for the lab to reach idle
# (no active claim AND no claude subprocess), then restarts the lab via
# `make lab-stop && make lab-start` so the new RO-CO v1 code becomes live.
#
# Bounded by MAX_POLLS — bails out if idle never reached, so this script
# can't sit running forever in the background.
#
# Idle condition (must hold for IDLE_REQUIRED consecutive polls):
#   1. data/work_queue/claimed.jsonl is empty (no Director currently working).
#   2. No `claude --print` subprocess alive (no in-flight Director session).
#
# Two consecutive idle polls = ~2 minutes of confirmed quiet, well under any
# skip-when-stable window the runner uses (default 4h, deep-stable 8h).

set -euo pipefail

# --- Resolve repo root (script is in tools/, repo is one level up) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- Config (overridable via env) ---
POLL_INTERVAL="${POLL_INTERVAL:-20}"      # seconds between polls (short = catches brief idle gaps)
MAX_POLLS="${MAX_POLLS:-1080}"            # 1080 × 20s = 6-hour ceiling
IDLE_REQUIRED="${IDLE_REQUIRED:-1}"       # consecutive idle polls needed (1 = catch first quiet poll)
VERBOSE="${VERBOSE:-1}"                   # 1 = log every poll (helps debug "lab too busy" cases)

LOG="data/infra/restart_when_idle.log"
mkdir -p "$(dirname "$LOG")"

_log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') restart-when-idle: $*" | tee -a "$LOG"
}

_log "starting watcher (poll=${POLL_INTERVAL}s, max=${MAX_POLLS}, idle_required=${IDLE_REQUIRED})"

idle_streak=0
last_state=""
for i in $(seq 1 "$MAX_POLLS"); do
    # Capture per-poll state for verbose logging
    claim_count=0
    if [ -s "data/work_queue/claimed.jsonl" ]; then
        claim_count=$(grep -c '^.' "data/work_queue/claimed.jsonl" 2>/dev/null || echo 0)
    fi
    claude_count=$(pgrep -f "claude --print" 2>/dev/null | wc -l | tr -d ' ')
    state="claims=${claim_count} claude=${claude_count}"

    # Condition 1: no active claim
    if [ "$claim_count" -gt 0 ]; then
        if [ "$idle_streak" -gt 0 ]; then
            _log "poll $i — $state — claim appeared during idle streak, resetting"
        elif [ "$VERBOSE" = "1" ] && [ "$state" != "$last_state" ]; then
            _log "poll $i — $state"
        fi
        idle_streak=0
        last_state="$state"
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Condition 2: no claude --print subprocess (Director session in flight)
    if [ "$claude_count" -gt 0 ]; then
        if [ "$idle_streak" -gt 0 ]; then
            _log "poll $i — $state — claude subprocess started during idle streak, resetting"
        elif [ "$VERBOSE" = "1" ] && [ "$state" != "$last_state" ]; then
            _log "poll $i — $state"
        fi
        idle_streak=0
        last_state="$state"
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Both conditions met — count this as an idle poll
    idle_streak=$((idle_streak + 1))
    _log "poll $i — $state — IDLE (${idle_streak}/${IDLE_REQUIRED})"
    last_state="$state"

    if [ "$idle_streak" -ge "$IDLE_REQUIRED" ]; then
        _log "idle threshold reached — restarting lab now"

        if make lab-stop 2>&1 | tee -a "$LOG"; then
            _log "lab-stop succeeded"
        else
            _log "lab-stop returned non-zero (probably already stopped); continuing"
        fi

        sleep 5

        if make lab-start 2>&1 | tee -a "$LOG"; then
            _log "lab-start succeeded — RO-CO v1 is now live"
            osascript -e 'display notification "RO-CO v1 active. Watch data/infra/post_director_telemetry.jsonl" with title "AGI Lab: Restart Complete"' 2>/dev/null || true
            exit 0
        else
            _log "lab-start FAILED — manual intervention required"
            osascript -e 'display notification "lab-start failed. Check restart_when_idle.log" with title "AGI Lab: Restart FAILED"' 2>/dev/null || true
            exit 2
        fi
    fi

    sleep "$POLL_INTERVAL"
done

_log "timeout — never reached idle in $MAX_POLLS polls; restart skipped"
osascript -e 'display notification "Idle window not reached. Restart skipped — run make lab-stop/start manually when ready" with title "AGI Lab: Restart Watcher Timeout"' 2>/dev/null || true
exit 1
