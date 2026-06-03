#!/usr/bin/env bash
# test_01_unknown_model.sh — `--model bogus123` must exit non-zero + FATAL.
# D-116 guardrail (scale_experiment.c:549). Takes ~1s (data load then fail).

set -euo pipefail
NAME="test_01_unknown_model"
BIN="build/scale_experiment"

if [[ ! -x "$BIN" ]]; then
    echo "SKIP: $NAME (binary not built; run 'make scale' first)"
    exit 0
fi

out_err="$(mktemp)"
trap 'rm -f "$out_err"' EXIT

set +e
"$BIN" --model bogus123 >/dev/null 2>"$out_err"
rc=$?
set -e

if [[ $rc -eq 0 ]]; then
    echo "FAIL: $NAME (expected non-zero exit, got 0)"
    exit 1
fi
if ! grep -q "FATAL: --model 'bogus123' is not recognized" "$out_err"; then
    echo "FAIL: $NAME (missing FATAL message on stderr)"
    exit 1
fi
echo "PASS: $NAME (exit=$rc, FATAL on stderr)"
