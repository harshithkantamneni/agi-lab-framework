#!/usr/bin/env bash
# test_02_moe_flag_on_dense.sh — dense50m + MoE-only flag rejected.
# D-116 guardrail (scale_experiment.c:564-584). The parser recognises four
# MoE-only flags; we pick --loss-free-balance (no extra value to parse).
# Task spec's `--load-balance-weight` example is not a real flag — use this.

set -euo pipefail
NAME="test_02_moe_flag_on_dense"
BIN="build/scale_experiment"

if [[ ! -x "$BIN" ]]; then
    echo "SKIP: $NAME (binary not built)"
    exit 0
fi

out_err="$(mktemp)"
trap 'rm -f "$out_err"' EXIT

set +e
"$BIN" --model dense50m --loss-free-balance >/dev/null 2>"$out_err"
rc=$?
set -e

if [[ $rc -eq 0 ]]; then
    echo "FAIL: $NAME (expected non-zero exit, got 0)"
    exit 1
fi
if ! grep -q "FATAL: --model dense50m is incompatible with MoE-specific" "$out_err"; then
    echo "FAIL: $NAME (missing MoE-on-dense FATAL message)"
    exit 1
fi
echo "PASS: $NAME (exit=$rc, MoE-on-dense guardrail fired)"
