# Kernel Optimization Task

{{PROMPT}}

**Read `Info.md` for additional context and user-provided hints before starting.**

## Objective

{{OBJECTIVE}}

The evaluated files are `solution/triton/kernel.py` (and `config.toml` if switching to destination_passing_style).
You may create temporary scripts for debugging or profiling, but only the above files are evaluated.
**Do NOT** look at, reference, or search for any kernel files outside this directory.
**Do NOT** use any web search tools to find other kernel implementations.

## Environment

{{ENVIRONMENT}}

## How to Run Benchmark

```bash
bash scripts/bench.sh
```

- **PASSED** = correct output within tolerance, only PASSED results count for scoring. **FAILED** = incorrect or crash.
- **speedup_factor** = reference_latency / your_latency. Higher is better.

## Trajectory Tracking

Benchmark runs are saved to `trajectory/` for analysis:
```bash
bash scripts/bench.sh                    # Run and save with timestamp only
bash scripts/bench.sh "initial_attempt"  # Run and save with descriptive label
bash scripts/bench.sh "optimized_v2"     # Track different optimization versions
```
Each run creates `trajectory/YYYYMMDD_HHMMSS[_label]/` with kernel.py, config.toml, and results.json.

## Specification

Operator: `{{OPERATOR}}`

{{OPERATOR_DESCRIPTION}}

See `docs/definition.json` for the full operator definition (inputs, outputs, constraints, reference code)
and `docs/workloads.jsonl` for all {{NUM_WORKLOADS}} benchmark workloads.

{{SHAPE_SUMMARY}}

{{WORKLOAD_SUMMARY}}

## config.toml

- `destination_passing_style = false`: `run()` allocates and returns output tensors.
- Set to `true` to receive pre-allocated output tensors as additional arguments to `run()`.

## Scoring

**Final score = arithmetic mean of speedup_factor across all {{NUM_WORKLOADS}} workloads.**
This is the primary optimization metric. The benchmark script prints this after each run.

**All {{NUM_WORKLOADS}} workloads must PASS.** If any workload FAILS, the kernel is invalid and receives no score.
Correctness is a hard gate — always ensure all workloads pass before focusing on performance.

**Don't optimize solely on the final score — check the per-group breakdown and target weak workloads specifically.**

## Correctness

Output must match reference within numerical tolerance (the benchmark checks this).
