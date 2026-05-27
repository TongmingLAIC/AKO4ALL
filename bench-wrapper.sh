#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."

# Defensive: if CUDA_HOME is unset and we're in a conda env, point CUDA_HOME at
# the env's CUDA toolkit. Otherwise torch.cpp_extension can fall back to a system
# /usr/local/cuda symlink that may point at an older CUDA than torch was built
# against (e.g., cu130 torch + symlink → cuda-11.8 → cudaDeviceProp ABI mismatch
# → SIGFPE in load_inline). Common on multi-CUDA hosts.
if [ -z "$CUDA_HOME" ] && [ -n "$CONDA_PREFIX" ]; then
    export CUDA_HOME="$CONDA_PREFIX"
    echo "[bench-wrapper] Auto-set CUDA_HOME=$CUDA_HOME (conda env). Override by exporting CUDA_HOME yourself." >&2
fi

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Bench command ---
# Run benchmark without exiting on failure — we need trajectory even for failed runs
set +e
{{BENCH_COMMAND}}
BENCH_EXIT=$?
set -e
# --- End bench command ---

# --- Trajectory ---
if [ -n "$LABEL" ]; then
    TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"
else
    TRAJ_DIR="trajectory/${TIMESTAMP}"
fi
mkdir -p "$TRAJ_DIR"
cp -r solution/* "$TRAJ_DIR/"
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"

exit $BENCH_EXIT
