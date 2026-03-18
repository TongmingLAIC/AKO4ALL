Further optimize the Triton kernel in `solution/triton/kernel.py` for maximum speedup on ${GPU_SHORT}.
This is a **{{OPERATOR_DESCRIPTION}}** operator.

The starting point is an **already partially-optimized Triton kernel**. Your job is to analyze it,
identify performance bottlenecks, and further optimize it for maximum speedup while preserving correctness.

The pure-Python reference implementation is available in `docs/definition.json` (under the "reference" field)
for understanding the original computation logic.

**Recommended first step:** Run `bash scripts/bench.sh "baseline"` to establish the current performance baseline.
