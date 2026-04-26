#!/usr/bin/env bash
# scripts/run_benchmarks.sh — sequentially execute the v3.0/v3.1 published
# benchmarks. Each is up to 30 min wall-clock on M3 Max. Results cached
# under test/benchmarks/.cache/ keyed by source-tree SHA-256 so reruns
# after unrelated source changes are instant.
#
# Usage:
#     ./scripts/run_benchmarks.sh                # all
#     ./scripts/run_benchmarks.sh ghia moser     # subset (substring match)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ALL_BENCHES=(
    "ghia_re400"
    "moser_re180"
    "rayleigh_benard_1e4"
    "martin_moyce_dam_break"
    "sod_shock_tube"
)

if [ "$#" -eq 0 ]; then
    SELECTED=("${ALL_BENCHES[@]}")
else
    SELECTED=()
    for arg in "$@"; do
        for bench in "${ALL_BENCHES[@]}"; do
            if [[ "$bench" == *"$arg"* ]]; then
                SELECTED+=("$bench")
            fi
        done
    done
fi

if [ "${#SELECTED[@]}" -eq 0 ]; then
    echo "No benchmarks matched '$*'" >&2
    echo "Available: ${ALL_BENCHES[*]}" >&2
    exit 1
fi

LOG_DIR="$REPO_ROOT/test/benchmarks/.logs"
mkdir -p "$LOG_DIR"

echo "Running ${#SELECTED[@]} benchmark(s): ${SELECTED[*]}"
echo "Logs:   $LOG_DIR"
echo "Cache:  $REPO_ROOT/test/benchmarks/.cache"
echo

PASS=0
FAIL=0
DEFER=0

for bench in "${SELECTED[@]}"; do
    log_file="$LOG_DIR/${bench}.log"
    echo "=== $bench ==="
    start=$(date +%s)
    if FVM_RUN_BENCHMARKS=true julia --project=test "test/benchmarks/${bench}.jl" \
        > "$log_file" 2>&1; then
        elapsed=$(( $(date +%s) - start ))
        if grep -q "deferred_compute" "$log_file"; then
            DEFER=$((DEFER + 1))
            echo "  ⊘ deferred (${elapsed}s) — see $log_file"
        else
            PASS=$((PASS + 1))
            echo "  ✓ pass (${elapsed}s)"
        fi
    else
        elapsed=$(( $(date +%s) - start ))
        FAIL=$((FAIL + 1))
        echo "  ✗ fail (${elapsed}s) — see $log_file"
        tail -20 "$log_file" | sed 's/^/    /'
    fi
done

echo
echo "Summary: $PASS passed, $FAIL failed, $DEFER deferred (of ${#SELECTED[@]})"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
