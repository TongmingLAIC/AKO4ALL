# Kernel Optimization Task

You are a GPU kernel optimization expert. Your task is to optimize the {{LANGUAGE_NAME}} kernel in `solution/{{LANGUAGE}}/` for maximum speedup.

Read `HINTS.md` for user-provided hints and constraints before starting.

## Objective

{{OBJECTIVE}}

Only files in `solution/{{LANGUAGE}}/` (and `config.toml` for destination_passing_style) are evaluated. Your kernel entry point is specified in `config.toml`.
You may create temporary scripts for debugging, but only the above files are evaluated.

## Environment

- GPU: {{GPU_NAME}}
{{BACKEND}}

## Benchmark

```bash
bash scripts/bench.sh
```

- **PASSED** = correct output within tolerance. **FAILED** = incorrect or crash.
- **speedup_factor** = reference_latency / your_latency. Higher is better.
- First run profiles the reference and caches to `baseline.json`. Subsequent runs use the cache with higher iteration count.
- Use `--force-baseline` to re-profile: `bash scripts/bench.sh --force-baseline`

Label runs for tracking: `bash scripts/bench.sh "description"` saves to `trajectory/YYYYMMDD_HHMMSS_description/`.

## Operator

`{{OPERATOR}}`: {{OPERATOR_DESCRIPTION}}

Full definition: `docs/definition.json` | Workloads: `docs/workloads.jsonl` ({{NUM_WORKLOADS}} total)

{{SHAPE_SUMMARY}}

{{WORKLOAD_SUMMARY}}

## config.toml

- `destination_passing_style = false`: your entry point function allocates and returns output tensors.
- Set to `true` to receive pre-allocated outputs as additional arguments.

## Scoring & Workflow

**Score = mean speedup across all {{NUM_WORKLOADS}} workloads.** All must PASS for a valid score.

1. Run `bash scripts/bench.sh "baseline"` to establish baseline
2. Analyze kernel + operator spec in `docs/definition.json`
3. Modify kernel -> bench -> analyze -> iterate
4. Roll back on regression; check per-group breakdown to target weak workloads
5. Stop when no further improvements found; summarize results
