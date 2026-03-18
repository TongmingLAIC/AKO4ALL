#!/bin/bash
# Quick benchmark runner (Modal B200 backend)
# Usage: bash scripts/bench.sh [label] [--force-baseline]
# (This is bench_modal.sh in the template repo, deployed as bench.sh in child environments)
cd "$(dirname "$0")/.."

# Parse arguments: first positional arg is label, --force-baseline is a flag
ARGS=()
LABEL=""
for arg in "$@"; do
    if [ "$arg" = "--force-baseline" ]; then
        ARGS+=(--force-baseline)
    elif [[ "$arg" == --* ]]; then
        echo "Warning: Unknown flag '$arg' ignored." >&2
    elif [ -z "$LABEL" ]; then
        LABEL="$arg"
        ARGS+=(--label "$arg")
    else
        echo "Warning: Extra positional argument '$arg' ignored (already using label '$LABEL')." >&2
    fi
done

conda run -n fi-bench --no-capture-output modal run scripts/run_modal.py "${ARGS[@]}"
