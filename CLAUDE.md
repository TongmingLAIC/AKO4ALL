# AutoKernelForge

Optimize the kernel in `solution/` for maximum performance, measured by `bash bench.sh`. The optimized kernel must produce outputs identical to the golden reference.

## Setup

Ensure the user has populated:
- `input/` — kernel files and optionally a reference implementation
- `bench/` — benchmark script and its dependencies

Then:
1. Read and analyze `input/`, `bench/`, and `HINTS.md`.
2. Create `solution/` and `scripts/` directories.
3. Copy kernel files from `input/` to `solution/`.
4. Build the bench command with adjusted paths, pipe through `2>&1 | tee _bench_output.txt`.
5. Generate `bench.sh` from `bench-wrapper.sh` — replace `{{BENCH_COMMAND}}` with the command from step 4.
6. `git add -A && git commit` the initial state.

## Directory Layout

- `input/` — user-provided original files, read-only. Must contain the kernel to optimize. May contain `reference.py` (or similar) as the correctness golden; if absent, the original kernel in `input/` serves as the golden reference.
- `bench/` — benchmark script and dependencies, read-only.
- `solution/` — editable, optimization target. Agent copies kernel here from `input/` and iterates.
- `bench.sh` — generated benchmark wrapper, read-only.
- `scripts/` — workspace for profiling/debug tools.
- `HINTS.md` — optimization hints from the user.
