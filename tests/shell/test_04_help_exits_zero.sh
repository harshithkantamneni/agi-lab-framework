#!/usr/bin/env bash
# test_04_help_exits_zero.sh — --help prints usage, exits 0 (scale_experiment.c:259-283).
# NOTE: no-args starts real training — we deliberately do NOT test that path.

set -euo pipefail
NAME="test_04_help_exits_zero"
BIN="build/scale_experiment"

if [[ ! -x "$BIN" ]]; then
    echo "SKIP: $NAME (binary not built)"
    exit 0
fi

out_std="$(mktemp)"
trap 'rm -f "$out_std"' EXIT

set +e
"$BIN" --help >"$out_std" 2>&1
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
    echo "FAIL: $NAME (expected exit 0, got $rc)"
    exit 1
fi
if ! grep -q "Usage: scale_experiment" "$out_std"; then
    echo "FAIL: $NAME (usage line not printed)"
    exit 1
fi
# Sanity check that a few key flags are advertised.
for flag in '--model' '--backprop' '--checkpoint-dir' '--stream'; do
    if ! grep -q -- "$flag" "$out_std"; then
        echo "FAIL: $NAME (help missing $flag)"
        exit 1
    fi
done
echo "PASS: $NAME (exit=0, usage printed with key flags)"
