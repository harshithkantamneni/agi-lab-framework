#!/usr/bin/env bash
# test_05_run_long_checkpoint_dir_eq.sh — B2 regression (D-118):
# --checkpoint-dir=PATH must parse identically to --checkpoint-dir PATH.
# Python probe only; does not invoke the launcher.

set -euo pipefail
NAME="test_05_run_long_checkpoint_dir_eq"

out="$(python3 - <<'PY' 2>&1
import sys
sys.path.insert(0, "tools")
import run_long
eq_form = run_long.extract_ckpt_dir(
    "build/scale_experiment --model dense50m --checkpoint-dir=data/ckpt_probe"
)
sp_form = run_long.extract_ckpt_dir(
    "build/scale_experiment --model dense50m --checkpoint-dir data/ckpt_probe"
)
missing = run_long.extract_ckpt_dir("build/scale_experiment --model dense50m")
print(f"eq={eq_form}")
print(f"sp={sp_form}")
print(f"missing={missing}")
assert eq_form == "data/ckpt_probe", f"eq-form broken: {eq_form!r}"
assert sp_form == "data/ckpt_probe", f"sp-form broken: {sp_form!r}"
assert missing is None, f"missing should be None, got {missing!r}"
print("ok")
PY
)"
rc=$?

if [[ $rc -ne 0 ]]; then
    echo "FAIL: $NAME (python probe exit $rc)"
    printf '%s\n' "$out"
    exit 1
fi
if ! printf '%s\n' "$out" | grep -q '^ok$'; then
    echo "FAIL: $NAME (probe did not emit ok)"
    printf '%s\n' "$out"
    exit 1
fi
echo "PASS: $NAME (both --checkpoint-dir forms parse to the same path)"
