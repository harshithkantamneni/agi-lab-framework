#!/usr/bin/env bash
# test_03_entropy_penalty_requires_backprop.sh — --entropy-penalty (or
# --temp-anneal) without --backprop must exit non-zero (scale_experiment.c:403-410).

set -euo pipefail
NAME="test_03_entropy_penalty_requires_backprop"
BIN="build/scale_experiment"

if [[ ! -x "$BIN" ]]; then
    echo "SKIP: $NAME (binary not built)"
    exit 0
fi

out_err="$(mktemp)"
trap 'rm -f "$out_err"' EXIT

set +e
"$BIN" --entropy-penalty >/dev/null 2>"$out_err"
rc=$?
set -e

if [[ $rc -eq 0 ]]; then
    echo "FAIL: $NAME (expected non-zero exit, got 0)"
    exit 1
fi
if ! grep -q "FATAL: --entropy-penalty and --temp-anneal require --backprop" "$out_err"; then
    echo "FAIL: $NAME (missing entropy-penalty/backprop FATAL message)"
    exit 1
fi
echo "PASS: $NAME (exit=$rc, requires-backprop guardrail fired)"
