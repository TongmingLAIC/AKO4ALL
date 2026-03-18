#!/bin/bash
# Quick benchmark runner
# Usage: bash scripts/bench.sh [label]
if [ -z "$FIB_DATASET_PATH" ]; then
    echo "Error: FIB_DATASET_PATH is not set. Set it to the path of your flashinfer-bench trace set."
    exit 1
fi
cd "$(dirname "$0")/.."

# Pass optional label parameter to run_local.py
if [ $# -eq 0 ]; then
    conda run -n fi-bench --no-capture-output python scripts/run_local.py
else
    conda run -n fi-bench --no-capture-output python scripts/run_local.py --label "$1"
fi
