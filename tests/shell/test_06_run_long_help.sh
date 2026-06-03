#!/usr/bin/env bash
# test_06_run_long_help.sh — `tools/run_long.py --help` must exit 0 and
# advertise the documented flags. Pure argparse call, no Popen.

set -euo pipefail
NAME="test_06_run_long_help"

out="$(mktemp)"
trap 'rm -f "$out"' EXIT

set +e
python3 tools/run_long.py --help >"$out" 2>&1
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
    echo "FAIL: $NAME (expected exit 0, got $rc)"
    cat "$out"
    exit 1
fi
for flag in '--run-id' '--cmd' '--foreground'; do
    if ! grep -q -- "$flag" "$out"; then
        echo "FAIL: $NAME (help missing $flag)"
        exit 1
    fi
done
echo "PASS: $NAME (help exits 0, advertises --run-id/--cmd/--foreground)"
