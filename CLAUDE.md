# kernel-opt-agent — Session 1 Action Guide

You are operating in the **template repository**. Your job is to gather information from the user, analyze their kernel and benchmark script, and create an isolated child environment for optimization.

## Conversation Protocol

### 1. Gather inputs

Ask the user for:
- **(a) Kernel path** — file or directory containing the kernel to optimize
- **(b) Bench script path** — the script that benchmarks the kernel
- **(c) Hints** (optional) — a file with optimization hints, or verbal hints to include

If the user provides all three upfront, skip asking.

### 2. Analyze the kernel

Read the kernel file(s). Identify:
- **Language**: Python, Triton, CUDA, C++, TileLang, or other
- **Entry point**: the main function name and file
- **Inputs/outputs**: tensor shapes, dtypes, arguments
- **Functionality**: what the kernel computes (one sentence)

### 3. Analyze the bench script

Read the bench script **and any libraries/configs it imports**. Identify:

**Required**:
- **Invocation**: how to run it (e.g., `python bench.py --kernel kernel.py`)
- **How it references the kernel**: import path, command-line arg, hardcoded path, etc.
- **Output format**: what it prints (latency, speedup, pass/fail, etc.)
- **Environment needs**: conda env, pip deps, special setup

**Investigate if possible** — look into the bench script's source, imported packages,
and config objects. Include whatever you find in the child CLAUDE.md; if something
is not available (e.g., opaque library), note that limitation so Session 2 doesn't
waste time investigating:

- **Correctness checking**: Does it validate output correctness? What tolerance
  (atol/rtol)? Is it strict (1e-6) or loose (1e-2)? This determines how aggressively
  the agent can trade precision for speed.
- **Timing method**: How is kernel time measured? (Python time.time, CUDA events,
  CUPTI tracing, etc.) Is there L2 cache flushing between iterations?
- **Benchmark config**: Number of warmup runs, iterations, trials. How are results
  aggregated (mean, median, min)?
- **Workload parameters**: What varies across test cases (input sizes, shapes, dtypes)?
  Can you extract the actual parameter values and their distribution?
- **Failure diagnostics**: When a test fails, does the output include error messages or
  tracebacks? If not, note this limitation.
- **Reference baseline**: Is speedup measured against a reference implementation or a
  fixed target? What level is the reference (naive Python, optimized CUDA, vendor library)?
- **Infrastructure**: Does the benchmark run locally or remotely? Any known failure modes
  (network, resource contention, image build)?

### 4. Confirm with user

Present findings. Ask the user to confirm or correct:
- Kernel summary (language, entry point, functionality)
- Bench invocation and how paths will be adjusted in the child env
- Anything that couldn't be inferred (optimization goal, GPU, environment name, etc.)
- Optional: a short label for the run directory

### 5. Create child environment

Follow the procedure in [Child Environment Creation](#child-environment-creation) below.

### 6. Done

Tell the user:
```
cd <child-dir> && claude
```

## Child Environment Creation

### Naming

Create the child directory as a **sibling** of this repo (i.e., `../<name>`):
- With label: `kernel-opt-agent-run-{label}`
- Without: `kernel-opt-agent-run-{YYYYMMDD_HHMMSS}`

### Directory structure

```
kernel-opt-agent-run-xxx/
├── CLAUDE.md                   # Generated task spec (self-contained)
├── HINTS.md                    # Optimization hints
├── .gitignore
├── solution/                   # Kernel files (editable by agent)
├── bench/                      # Bench script + deps (read-only)
├── scripts/
│   └── bench.sh                # Generated wrapper
└── .claude/
    └── settings.local.json     # Agent permissions
```

### Steps

1. **Create directory tree**: `mkdir -p <child>/{solution,bench,scripts,.claude}`

2. **Copy kernel files to `solution/`**: Copy the kernel file(s) into `solution/`. Preserve filenames. If it's a directory, copy contents.

3. **Copy bench script + local deps to `bench/`**: Copy the bench script and any local files it imports into `bench/`. If the bench script is in a directory with helpers, copy the directory contents.

4. **Adjust bench command for new paths**: Figure out the command to run the bench script from the child root, with paths adjusted:
   - Kernel path becomes `solution/<filename>`
   - Bench script path becomes `bench/<filename>`
   - Example: if original was `python bench.py --kernel my_kernel.py`, becomes `python bench/bench.py --kernel solution/my_kernel.py`
   - Pipe through tee to capture output: `<command> 2>&1 | tee _bench_output.txt`

5. **Generate `scripts/bench.sh`**: Read `templates/bench-wrapper.sh` from this repo, replace `{{BENCH_COMMAND}}` with the adjusted command from step 4, write to `<child>/scripts/bench.sh`, make executable.

6. **Write `.claude/settings.local.json`**: Read `templates/agent/claude.json` from this repo, extract the `permissions` object, write as `<child>/.claude/settings.local.json`.

7. **Copy `.gitignore`**: Copy this repo's `.gitignore` to `<child>/.gitignore`.

8. **Write `HINTS.md`**: If user provided a hints file, copy it. If user gave verbal hints, write them. Otherwise copy `templates/hints.md` from this repo.

9. **Write `CLAUDE.md`**: Generate the child's CLAUDE.md following the [required sections](#child-claudemd-required-sections) below. Use `templates/task.md` as a reference example for format and detail level.

10. **Initialize git**: `cd <child> && git init && git add -A && git commit -m "Initial environment"`

### Child CLAUDE.md Required Sections

The generated CLAUDE.md must be **self-contained** — Session 2 will not read this template repo. Include these sections:

1. **Role** — one line: "You are a GPU kernel optimization expert. Your task is to optimize the kernel in `solution/` for maximum performance."

2. **Hints** — "Read `HINTS.md` before starting for user-provided hints and constraints."

3. **Kernel** — language, entry point, input/output shapes and types, what it computes. Be specific — this is what the agent needs to understand the code.

4. **Benchmark** — how to run (`bash scripts/bench.sh [label]`), output format, what PASSED/FAILED means, what the primary metric is.

5. **Editable files** — "Only modify files in `solution/`. Do NOT modify files in `bench/` or `scripts/`."

6. **Workflow** — baseline -> analyze -> modify -> bench -> iterate -> rollback on regression -> stop when done.

## Constraints

- Never modify files outside the child directory.
- The child CLAUDE.md must be self-contained (Session 2 does not read this template repo).
- Always init git in the child.
- Adjust all paths in the bench command so they work from the child root directory.
- If the bench script has complex dependencies (conda env, pip packages, etc.), document the environment setup in the child CLAUDE.md's Benchmark section.
