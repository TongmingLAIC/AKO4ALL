# Kernel Optimization Task

You are a GPU kernel optimization expert. Your task is to optimize the kernel in `solution/` for maximum performance.

Read `HINTS.md` before starting for user-provided hints and constraints.

## Kernel

- **Language**: Triton
- **Entry point**: `solution/kernel.py` -> `softmax_kernel()` function
- **Functionality**: Fused softmax over the last dimension of a 2D tensor
- **Inputs**: `x` (float32 tensor, shape `[M, N]`)
- **Outputs**: `out` (float32 tensor, shape `[M, N]`)
- **Computation pattern**: Row-wise reduction (max + sum for numerical stability), then elementwise exp and normalize
- **Output allocation**: The `run()` function must allocate and return output tensors (not write into pre-allocated buffers passed as arguments)

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

### Benchmark Internals

- **Timing**: CUDA events with synchronization
- **L2 cache**: Warm (no explicit flushing between iterations)
- **Config**: warmup_runs=5, iterations=100, num_trials=3 (3 random inputs, 100 iters each, median per trial, average across trials)
- **Reference**: The reference implementation is the unoptimized `solution/kernel.py` baseline (the starting point)

### Correctness Tolerance

- **atol** = 1e-3 (absolute error tolerance)
- **rtol** = 1e-3 (relative error tolerance)
- PASSED if all output elements satisfy `|actual - expected| < atol + rtol * |expected|`
- This tolerance is moderately strict — bf16 internal accumulation may cause failures on large N; prefer fp32 accumulation with bf16 loads where needed

### Workload Distribution

6 workloads covering small to large matrix sizes:
- 2 small:  M=128, N=512 and M=256, N=1024
- 2 medium: M=1024, N=4096 and M=2048, N=4096
- 2 large:  M=4096, N=4096 and M=4096, N=8192

Optimize for all workload sizes; mean speedup across all workloads is the primary metric.

### Known Limitations

- Benchmark output does not include detailed error messages on failure — only PASSED/FAILED status is shown. If a workload fails, check kernel output shapes and dtypes manually.

## Editable Files

Only modify files in `solution/`. You may create temporary scripts (e.g., `debug.py`) for analysis, but only `solution/` is benchmarked.

Do NOT modify files in `bench/` or `scripts/`.

## Workflow

1. Run `bash scripts/bench.sh "baseline"` to establish baseline performance
2. Analyze baseline output: note workload count, latency distribution, and any
   patterns (e.g., latency tiers suggesting different input sizes). This informs
   which cases to prioritize.
3. Read and analyze the kernel in `solution/`
4. Identify optimization opportunities and rewrite/optimize the kernel
5. Modify kernel -> `bash scripts/bench.sh "description"` -> analyze results -> iterate
6. If a change causes FAILED:
   a. Read the benchmark output to identify the failure type
   b. For numerical errors: try targeted fixes (e.g., more fp32 accumulation)
   c. For crashes: check shape mismatches, OOM, or compilation issues
   d. If the cause is unclear or unfixable, revert: `git checkout solution/`
7. Check per-workload breakdown to target the weakest cases
8. Stop when no further improvements are found; summarize final results

<!-- =================================================================
     THIS IS A REFERENCE EXAMPLE for Session 1 (CLAUDE.md generation).

     REQUIRED SECTIONS (always include):
       1. Role       — one-line expert description
       2. Hints      — "Read HINTS.md before starting"
       3. Kernel     — language, entry point, inputs, outputs, what it does,
                       computation pattern, output allocation
       4. Benchmark  — how to run, output format, pass/fail criteria
       5. Editable   — which files the agent may modify
       6. Workflow   — use the detailed template from mother CLAUDE.md

     CONDITIONAL SECTIONS (include if found during analysis):
       - Benchmark Internals — timing method (CUDA events, CUPTI, etc.),
         L2 cache policy (cold/warm), config (warmup, iterations, trials),
         result aggregation (mean/median/min), reference baseline level
       - Correctness Tolerance — atol/rtol values and what they imply
         (e.g., loose tolerance = aggressive bf16 is acceptable)
       - Workload Distribution — actual parameter values and counts
         (e.g., "10 workloads with num_tokens=1-2, 13 with num_tokens=4-8")
       - Known Limitations — opaque libraries, missing error diagnostics,
         infrastructure failure modes, or other Session 2 constraints

     KERNEL SECTION DEPTH — beyond the basics, include if discovered:
       - Reference implementation pitfalls (patterns to avoid copying)
       - Computation pattern (reduction, gather/scatter, matmul, etc.)
       - Special semantics (padding/sentinel values, edge case outputs)
       - Output allocation (returns new tensors vs writes into buffers)

     This example demonstrates ALL section types with realistic data.
     Adapt all details to the actual kernel and bench script provided
     by the user. The above is a hypothetical Triton softmax example.
     ================================================================= -->
