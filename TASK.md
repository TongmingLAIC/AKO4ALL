# AKO4ALL

Optimize the kernel in `solution/` for maximum performance, measured by `bash bench.sh`. The optimized kernel must produce outputs identical to the golden reference.

## Setup

Ensure the user has populated:
- `input/` — kernel files and optionally a reference implementation
- `context/` — reference materials **(optional)**
- `bench/` — benchmark script and its dependencies **(optional — if empty, uses KernelBench eval)**

Then:
1. Read and analyze `input/`, `context/`, `bench/`, and `HINTS.md`.
2. **Detect bench mode:**
   - If `bench/` contains files besides `kernelbench/` → **custom bench mode** (user-provided benchmark, skip to step 3).
   - Otherwise (only `kernelbench/` present) → **default bench mode** (KernelBench eval). Read `bench/kernelbench/GUIDE.md` for setup instructions, then continue at step 3.
3. **Verify input shapes:** Confirm that at least one set of input shapes for testing can be determined from the kernel, reference, bench script, or additional files in `input/` / `bench/`. If no shapes can be found, **stop and ask the user** — do not guess.
4. Create `solution/` and `scripts/` directories.
5. Copy kernel files from `input/` to `solution/`.
6. Build the bench command with adjusted paths, pipe through `2>&1 | tee _bench_output.txt`.
7. Generate `bench.sh` from `bench-wrapper.sh` — replace `{{BENCH_COMMAND}}` with the command from step 5.
8. **Verify bench environment:** Run `bash bench.sh`. Since `solution/` is a copy of the reference, this validates the full toolchain. Expected result: `CORRECT=True`, speedup ≈ 1.0x. If it fails, diagnose the error and fix it before proceeding.
9. `git add -A && git commit` the initial state.

## Optimization

- Use `bash bench.sh` to measure performance.
- Use `ncu` to profile kernels and identify bottlenecks — do not optimize blindly.
- Leverage all available information: `context/`, `HINTS.md`, prior attempts, web search, etc.

### Iteration Tracking

After each optimization iteration, record the attempt:

1. **Update `iterations.md`** — append an entry for this iteration (see template in that file).
2. **Git commit** with the following message format:
   ```
   [iter N] Short description of optimization direction

   Direction: What was tried (e.g. "Triton autotune + 128x256 tiles")
   Result: Performance data (e.g. "0.633ms, 0.92x baseline")
   Status: improved | no-change | regression
   Analysis: One line on why it worked/failed and what to try next
   ```
3. **Git tag** — `git tag iter-N` for easy navigation between iterations.

## Directory Layout

- `input/` — user-provided original files, read-only. Must contain the kernel to optimize. May contain `reference.py` (or similar) as the correctness golden; if absent, the original kernel in `input/` serves as the golden reference.
- `context/` — user-provided reference materials, read-only. Algorithm descriptions, papers, design docs, or other background knowledge for the agent. Optional.
- `bench/` — benchmark script and dependencies **(optional)**. If empty, `bench/kernelbench/bench.py` is used (self-contained, no external KernelBench dependency).
- `solution/` — editable, optimization target. Agent copies kernel here from `input/` and iterates.
- `bench.sh` — generated benchmark wrapper, read-only.
- `scripts/` — workspace for profiling/debug tools.
- `HINTS.md` — directives for the agent: optimization constraints, focus areas, and behavior controls.

For default bench details (output format, CLI args, tolerances), see `bench/kernelbench/GUIDE.md`.
