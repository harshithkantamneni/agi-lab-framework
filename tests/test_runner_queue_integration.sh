#!/bin/bash
# Smoke test for the Q3 runner queue integration.
# Tests the helper functions in isolation by sourcing the runner's helper
# definitions in a test harness.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# Test 1: _queue_has_pending returns false on missing file
echo "Test 1: missing pending.jsonl"
cd "$TMP"
mkdir -p data/work_queue
# Source the helper definition (extract from runner script).
# Use a temp file rather than process substitution to avoid set -e
# treating the awk subshell's exit code as a test failure.
TMPFN=$(mktemp)
awk '/^_queue_has_pending\(\)/,/^}/' "$REPO/run_agi_lab.sh" > "$TMPFN"
source "$TMPFN"
rm -f "$TMPFN"
if _queue_has_pending; then
    echo "FAIL: should return 1 on missing file"; exit 1
fi
echo "PASS"

# Test 2: _queue_has_pending returns false on empty file
echo "Test 2: empty pending.jsonl"
touch data/work_queue/pending.jsonl
if _queue_has_pending; then
    echo "FAIL: should return 1 on empty file"; exit 1
fi
echo "PASS"

# Test 3: _queue_has_pending returns true with one item
echo "Test 3: pending.jsonl with item"
echo '{"id":"wq-test","type":"phase_advance","priority":"normal"}' > data/work_queue/pending.jsonl
if ! _queue_has_pending; then
    echo "FAIL: should return 0 with 1 item"; exit 1
fi
echo "PASS"

echo
echo "All Q3 runner integration tests passed."
