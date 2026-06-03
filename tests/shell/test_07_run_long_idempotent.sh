#!/usr/bin/env bash
# test_07_run_long_idempotent.sh — live pid file -> run_long.py no-ops and
# returns 0 (tools/run_long.py:101-105). Probe read_pid + pid_alive directly;
# we do NOT invoke main() so no detached child ever spawns near
# data/runs/dense_a|moe_rev2. Throwaway dir + sleep get cleaned up in trap.
set -euo pipefail
NAME="test_07_run_long_idempotent"
RUN_DIR="data/runs/test_idempotent_$$"
PID_FILE="$RUN_DIR/pid"

cleanup() {
    [[ -f "$PID_FILE" ]] && {
        pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null || true
    }
    rm -rf "$RUN_DIR"
}
trap cleanup EXIT

mkdir -p "$RUN_DIR"
# Backgrounded in THIS shell (not a $() subshell, which invalidates $!).
sleep 30 &
SLEEP_PID=$!
echo "$SLEEP_PID" >"$PID_FILE"
sleep 0.1

if ! kill -0 "$SLEEP_PID" 2>/dev/null; then
    echo "FAIL: $NAME (helper sleep pid $SLEEP_PID not alive)"; exit 1
fi

out="$(python3 - <<PY 2>&1
import sys; sys.path.insert(0, "tools")
import run_long
existing = run_long.read_pid("$PID_FILE")
alive = run_long.pid_alive(existing) if existing else False
assert existing == $SLEEP_PID, f"expected $SLEEP_PID, got {existing}"
assert alive, "expected pid_alive True"
print("ok")
PY
)"
rc=$?

if [[ $rc -ne 0 ]] || ! printf '%s\n' "$out" | grep -q '^ok$'; then
    echo "FAIL: $NAME (probe exit=$rc, out=$out)"; exit 1
fi
echo "PASS: $NAME (read_pid+pid_alive both satisfy the no-op branch)"
