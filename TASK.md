# AKO4ALL

Optimize the kernel in `solution/` for maximum performance, measured by `bash scripts/bench.sh`. The optimized kernel must produce outputs identical to the golden reference.

Your goal is genuine latency reduction — not maximizing the reported speedup ratio. Do not use techniques that have no value in production: CUDA stream injection to evade timing, thread/process injection, monkey-patching timing functions or the benchmark script, or any other form of reward hacking.

## Setup

Ensure the user has populated:
- `input/` — kernel files and optionally a reference implementation
- `context/` — reference materials **(optional)**
- `bench/` — benchmark script and its dependencies **(optional — if empty, uses KernelBench eval)**

Then:
1. **Analyze inputs:** Read `input/`, `context/`, `bench/`, and `HINTS.md`. Detect bench mode — if `bench/` contains files besides `kernelbench/`, use the user-provided benchmark; otherwise use default bench mode (see `bench/kernelbench/GUIDE.md`). Confirm that input shapes can be determined; if not, **stop and ask the user**.
2. **Create branch:** Create and switch to a new branch (e.g., `opt/<kernel-name>`).
3. **Initialize solution:** Create `solution/` and `scripts/` directories. Copy kernel files from `input/` to `solution/`.
4. **Generate bench.sh:** Build the bench command with adjusted paths, pipe through `2>&1 | tee _bench_output.txt`. Replace `{{BENCH_COMMAND}}` in `bench-wrapper.sh` to produce `scripts/bench.sh`.
5. **Verify environment:** Run `bash scripts/bench.sh`. Expected: `CORRECT=True`. If it fails, diagnose and fix before proceeding. Then `git add -A && git commit -m "[baseline] Initialize solution and benchmark"`.

## Optimization

- Use `bash scripts/bench.sh` to measure performance.
- Use `ncu` to profile and identify bottlenecks — do not optimize blindly.
- Leverage all available information: `context/`, `HINTS.md`, prior attempts, web search, etc.
- Follow stall rules defined in `HINTS.md`.

### Iteration Protocol

Every modification to `solution/` code followed by a `bash scripts/bench.sh` run counts as one iteration — regardless of whether the result is an improvement, regression, or failure. Number iterations sequentially (1, 2, 3, …).

**Do NOT start the next iteration until ALL steps below are completed:**

1. **Run benchmark** — `bash scripts/bench.sh iter-N` (label is required, must match `iter-N` format).
2. **Update `ITERATIONS.md`** 
3. **Git commit** — `[iter N] Short description of optimization direction`.
