# Kernel Optimization Task

You are a GPU kernel optimization expert. Your task is to optimize the kernel in `solution/` for maximum performance.

Read `HINTS.md` before starting for user-provided hints and constraints.

## Kernel

- **Language**: Triton
- **Entry point**: `solution/kernel.py` -> `softmax_kernel()` function
- **Functionality**: Fused softmax over the last dimension of a 2D tensor
- **Inputs**: `x` (float32 tensor, shape `[M, N]`)
- **Outputs**: `out` (float32 tensor, shape `[M, N]`)

## Benchmark

```bash
bash scripts/bench.sh [label]
```

Runs the benchmark script and saves a trajectory snapshot.

- The bench script prints per-case results with latency and correctness status
- **PASSED** = correct output within tolerance. **FAILED** = incorrect or crash
- The primary metric is latency (lower is better) or speedup vs baseline
- Label runs for tracking: `bash scripts/bench.sh "v1_tiling"` saves to `trajectory/YYYYMMDD_HHMMSS_v1_tiling/`

### Example output

```
Case M=1024, N=4096: PASSED | 0.312 ms | speedup=2.1x
Case M=4096, N=4096: PASSED | 1.024 ms | speedup=1.8x
...
Mean speedup: 1.95x
```

## Editable Files

Only modify files in `solution/`. You may create temporary scripts (e.g., `debug.py`) for analysis, but only `solution/` is benchmarked.

Do NOT modify files in `bench/` or `scripts/`.

## Workflow

1. Run `bash scripts/bench.sh "baseline"` to establish baseline performance
2. Read and analyze the kernel in `solution/`
3. Identify optimization opportunities (memory access, tiling, fusion, etc.)
4. Modify kernel -> `bash scripts/bench.sh "description"` -> analyze results -> iterate
5. If a change causes regression or FAILED, revert using git: `git checkout solution/`
6. Check per-case breakdown to target the weakest cases
7. Stop when no further improvements are found; summarize final results

<!-- =================================================================
     THIS IS A REFERENCE EXAMPLE for Session 1 (CLAUDE.md generation).

     When creating a child CLAUDE.md, include these required sections:
       1. Role       — one-line expert description
       2. Hints      — "Read HINTS.md before starting"
       3. Kernel     — language, entry point, inputs, outputs, what it does
       4. Benchmark  — how to run, output format, pass/fail criteria
       5. Editable   — which files the agent may modify
       6. Workflow   — baseline -> analyze -> modify -> bench -> iterate

     Adapt all details to the actual kernel and bench script provided
     by the user. The above is a hypothetical Triton softmax example.
     ================================================================= -->
