#!/usr/bin/env bash
# tests/shell/run_all.sh — CLI guardrail shell-test harness (D-120).
# Verifies binary end-to-end behavior of scale_experiment CLI guardrails and
# run_long.py launcher paths. See tests/shell/test_*.sh for individual cases.
# Usage: bash tests/shell/run_all.sh    Exit: 0 on all PASS/SKIP; non-zero on FAIL.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

P=0; F=0; S=0; FAILED=()
TESTS=()
while IFS= read -r -d '' f; do TESTS+=("$f"); done \
    < <(find "$SCRIPT_DIR" -maxdepth 1 -name 'test_*.sh' -type f -print0 | sort -z)
[[ ${#TESTS[@]} -gt 0 ]] || { echo "FAIL: no test_*.sh in $SCRIPT_DIR"; exit 1; }

echo "=== shell-guardrail harness: ${#TESTS[@]} test(s) ==="
for t in "${TESTS[@]}"; do
    name="$(basename "$t" .sh)"
    set +e; output="$(bash "$t" 2>&1)"; rc=$?; set -e
    # Each test self-reports via PASS:/FAIL:/SKIP:. Fall back on exit code.
    line="$(printf '%s\n' "$output" | grep -E '^(PASS|FAIL|SKIP):' | tail -n 1 || true)"
    if [[ -z "$line" ]]; then
        [[ $rc -eq 0 ]] && line="PASS: $name (exit 0, no status line)" \
                       || line="FAIL: $name (no status line, exit $rc)"
    fi
    echo "$line"
    case "$line" in
        PASS:*) P=$((P+1)) ;;
        SKIP:*) S=$((S+1)) ;;
        FAIL:*) F=$((F+1)); FAILED+=("$name")
                printf '  --- %s output ---\n' "$name"
                printf '%s\n' "$output" | sed 's/^/  | /' ;;
    esac
done
echo "=== results: ${P} PASS, ${F} FAIL, ${S} SKIP ==="
[[ $F -eq 0 ]] || { echo "failed: ${FAILED[*]}"; exit 1; }
exit 0
