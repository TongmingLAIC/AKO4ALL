# AKO4ALL

Optimize the kernel in `solution/` for maximum performance, measured by `bash bench.sh`. The optimized kernel must produce outputs identical to the golden reference.

## Setup

Ensure the user has populated:
- `input/` — kernel files and optionally a reference implementation
- `context/` — reference materials **(optional)**
- `bench/` — benchmark script and its dependencies **(optional — if empty, uses KernelBench eval)**

Then:
1. **Analyze inputs:** Read `input/`, `context/`, `bench/`, and `HINTS.md`. Detect bench mode — if `bench/` contains files besides `kernelbench/`, use the user-provided benchmark; otherwise use default bench mode (see `bench/kernelbench/GUIDE.md`). Confirm that input shapes can be determined; if not, **stop and ask the user**.
2. **Initialize solution:** Create `solution/` and `scripts/` directories. Copy kernel files from `input/` to `solution/`.
3. **Generate bench.sh:** Build the bench command with adjusted paths, pipe through `2>&1 | tee _bench_output.txt`. Replace `{{BENCH_COMMAND}}` in `bench-wrapper.sh` to produce `bench.sh`.
4. **Verify environment:** Run `bash bench.sh`. Expected: `CORRECT=True`. If it fails, diagnose and fix before proceeding. Then `git add -A && git commit`.

## Optimization

- Use `bash bench.sh` to measure performance.
- Use `ncu` to profile and identify bottlenecks — do not optimize blindly.
- Leverage all available information: `context/`, `HINTS.md`, prior attempts, web search, etc.
- Follow stall rules defined in `HINTS.md`.

### Iteration Protocol

Every modification to `solution/` code followed by a `bash bench.sh` run counts as one iteration — regardless of whether the result is an improvement, regression, or failure. Number iterations sequentially (1, 2, 3, …).

**Do NOT start the next iteration until ALL steps below are completed:**

1. **Run benchmark** — `bash bench.sh iter-N` (label is required, must match `iter-N` format).
2. **Update `ITERATIONS.md`** 
3. **Git commit** — `[iter N] Short description of optimization direction`.
