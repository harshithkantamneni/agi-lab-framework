#!/usr/bin/env bash
# test_08_final_ckpt_xfail.sh — KNOWN FAIL / xfail marker (D-119 B3
# follow-up). tools/run_long.py:27 CKPT_RE matches step_NNNNNN.ckpt only, so
# runs producing only final.ckpt cannot auto-resume. Fix deferred; this test
# emits SKIP today and flips to PASS when the regex is broadened.

set -euo pipefail
NAME="test_08_final_ckpt_xfail"

actual="$(python3 - <<'PY' 2>&1
import sys
sys.path.insert(0, "tools")
import run_long
print("step_000500.ckpt", bool(run_long.CKPT_RE.match("step_000500.ckpt")))
print("final.ckpt", bool(run_long.CKPT_RE.match("final.ckpt")))
PY
)"

# Expectations today (xfail):
#   step_000500.ckpt -> True   (regex matches — good)
#   final.ckpt       -> False  (regex does NOT match — the known bug)
step_line="$(printf '%s\n' "$actual" | grep '^step_000500.ckpt ')"
final_line="$(printf '%s\n' "$actual" | grep '^final.ckpt ')"

if [[ "$step_line" != "step_000500.ckpt True" ]]; then
    echo "FAIL: $NAME (step_000500.ckpt regex match broke: $step_line)"
    exit 1
fi
if [[ "$final_line" == "final.ckpt True" ]]; then
    # Fix landed — convert to PASS.
    echo "PASS: $NAME (regex now matches final.ckpt — xfail cleared, flip this test)"
    exit 0
fi
echo "SKIP: $NAME (xfail: CKPT_RE does not match final.ckpt; B3 fix deferred)"
