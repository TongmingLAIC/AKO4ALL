#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."

# Detect toolkit-vs-torch ambiguity and warn — don't auto-set CUDA_HOME.
# Auto-set is unreliable when active conda env != target python env (e.g.,
# base shell running envs/X/bin/python directly), and a wrong CUDA_HOME
# silently produces ABI mismatches in torch.cpp_extension / load_inline
# (e.g., cu130 torch + cu117 nvcc → cudaDeviceProp mismatch → SIGFPE).
# Common on multi-CUDA hosts.
if ! [ -x "${CUDA_HOME%%:*}/bin/nvcc" ]; then
    TORCH_CU=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
    if [ -n "$TORCH_CU" ]; then
        echo "[bench-wrapper] CUDA_HOME=${CUDA_HOME:-(unset)} → no nvcc found; torch built with CUDA $TORCH_CU." >&2
        echo "  For load_inline / cpp_extension, export CUDA_HOME to the env whose nvcc matches CUDA $TORCH_CU, e.g.:" >&2
        echo "    export CUDA_HOME=\$(python -c 'import sys; print(sys.prefix)')" >&2
    fi
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
