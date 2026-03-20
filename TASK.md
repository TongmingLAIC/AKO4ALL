# AKO4ALL

Optimize the kernel in `solution/` for maximum performance, measured by `bash bench.sh`. The optimized kernel must produce outputs identical to the golden reference.

## Setup

Ensure the user has populated:
- `input/` — kernel files and optionally a reference implementation
- `bench/` — benchmark script and its dependencies **(optional — if empty, uses KernelBench eval)**

Then:
1. Read and analyze `input/`, `bench/`, and `HINTS.md`.
2. **Detect bench mode:**
   - If `bench/` contains files (besides `.gitkeep` and `kernelbench/`) → **custom bench mode** (existing behavior, skip to step 3).
   - If `bench/` is empty → **default bench mode** (KernelBench eval). Read `bench/kernelbench/GUIDE.md` for setup instructions, then continue at step 3.
3. Create `solution/` and `scripts/` directories.
4. Copy kernel files from `input/` to `solution/`.
5. Build the bench command with adjusted paths, pipe through `2>&1 | tee _bench_output.txt`.
6. Generate `bench.sh` from `bench-wrapper.sh` — replace `{{BENCH_COMMAND}}` with the command from step 5.
7. `git add -A && git commit` the initial state.

## Directory Layout

- `input/` — user-provided original files, read-only. Must contain the kernel to optimize. May contain `reference.py` (or similar) as the correctness golden; if absent, the original kernel in `input/` serves as the golden reference.
- `bench/` — benchmark script and dependencies **(optional)**. If empty, `bench/kernelbench/bench.py` is used (self-contained, no external KernelBench dependency).
- `solution/` — editable, optimization target. Agent copies kernel here from `input/` and iterates.
- `bench.sh` — generated benchmark wrapper, read-only.
- `scripts/` — workspace for profiling/debug tools.
- `HINTS.md` — optimization hints from the user.

For default bench details (output format, CLI args, tolerances), see `bench/kernelbench/GUIDE.md`.
