# AutoKernelForge

Automated GPU kernel optimization powered by Claude Code. Provide any kernel and its benchmark script — Claude Code analyzes them, creates an isolated optimization environment, and iteratively rewrites the kernel for maximum performance.

**Why use this?** Manually optimizing GPU kernels is time-consuming and requires deep expertise in hardware-specific tricks (tiling, memory coalescing, warp-level primitives, etc.). This framework automates the trial-and-error loop: analyze → modify → benchmark → iterate, while tracking every attempt so you can review and compare.

## Supported Kernel Types

Triton, CUDA, C++, TileLang, Python — any kernel that can be benchmarked by a script.

## Prerequisites

**Framework dependencies** (required):

- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)
- Git

**Benchmark environment** (depends on your setup):

The framework itself does not require a GPU — it only analyzes code and orchestrates the optimization loop. Whether you need a local GPU, a remote cluster, or a cloud API depends entirely on your benchmark script. Make sure your bench script runs successfully in your current environment before starting. Examples:

- Local GPU benchmark → CUDA toolkit, PyTorch, etc. installed locally
- [Modal](https://modal.com/) remote benchmark → Modal CLI + API token configured
- Remote cluster → SSH access, job submission tools, etc.

## Quick Start

```bash
cd AutoKernelForge
claude
```

Claude Code will ask you for:
1. **Kernel path** — file or directory containing the kernel to optimize
2. **Bench script path** — the script that benchmarks the kernel
3. **Hints** (optional) — optimization hints or constraints

It analyzes both files, creates an isolated child environment, and tells you to:

```bash
cd ../akf-run-xxx
claude
```

The second session reads the generated `CLAUDE.md` and optimizes the kernel autonomously.

### Example Walkthrough

Suppose you have a Triton softmax kernel and a benchmark script:

```
my-project/
├── softmax_kernel.py    # kernel to optimize
└── bench.py             # benchmarks the kernel
```

**Session 1** — setup (in this repo):

```
> claude
You: Kernel: ../my-project/softmax_kernel.py
     Bench: ../my-project/bench.py
     Hint: Focus on large N cases, they are the bottleneck

Claude: [analyzes kernel and bench script, summarizes findings]
        [creates ../akf-run-softmax/]
        Done! Run: cd ../akf-run-softmax && claude
```

**Session 2** — optimization (in the child env):

```
> cd ../akf-run-softmax && claude
You: Optimize the kernel for maximum performance.

Claude: [runs baseline benchmark]
        [reads and analyzes kernel]
        [rewrites kernel with tiling optimization]
        [benchmarks → 1.8x speedup, all PASSED]
        [tries vectorized loads → 2.3x speedup]
        [tries warp-level reduction → 2.5x speedup, but 1 case FAILED]
        [reverts, tries alternative → 2.4x, all PASSED]
        [no further improvements found]
        Final result: 2.4x mean speedup across all workloads
```

Every benchmark run is saved to `trajectory/` so you can review the full optimization history.

## How It Works

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│  Session 1 (this repo)          │       │  Session 2 (child env)          │
│                                 │       │                                 │
│  1. Gather kernel + bench paths │       │  1. Run baseline benchmark      │
│  2. Analyze kernel code         │──────>│  2. Read & analyze kernel       │
│  3. Analyze bench script        │creates│  3. Optimize kernel             │
│  4. Confirm with user           │ child │  4. Benchmark → check results   │
│  5. Create child environment    │  env  │  5. Iterate until converged     │
└─────────────────────────────────┘       └─────────────────────────────────┘
```

**Why two sessions?** The child environment is fully self-contained — its `CLAUDE.md` has everything Session 2 needs. This means:

- **Isolation**: kernel edits are sandboxed; your original files are untouched
- **Reproducibility**: the child env is a git repo; every optimization attempt is tracked
- **Restartable**: you can `cd` into the child env and run `claude` again to continue optimizing

### Child Environment Structure

```
akf-run-xxx/
├── CLAUDE.md                   # Task spec with kernel/bench details (self-contained)
├── HINTS.md                    # Optimization hints and constraints
├── .gitignore
├── solution/                   # Kernel files — only these are edited
├── bench/                      # Bench script + deps (read-only)
├── scripts/
│   └── bench.sh                # Wrapper: runs benchmark + saves trajectory
├── trajectory/                 # Created automatically on first run
│   ├── 20260319_143022_baseline/
│   │   ├── kernel.py           # Kernel snapshot at this point
│   │   └── output.txt          # Benchmark output
│   ├── 20260319_143145_v1_tiling/
│   └── ...
└── .claude/
    └── settings.local.json     # Agent permissions
```

## Optimization Tracking

Every `bash scripts/bench.sh "label"` run automatically saves:
- A **snapshot** of all files in `solution/` at that point
- The **benchmark output** (stdout/stderr)

This gives you a full history of what was tried and how each version performed. Use labels to annotate attempts:

```bash
bash scripts/bench.sh "baseline"
bash scripts/bench.sh "v1_tiling"
bash scripts/bench.sh "v2_vectorized_loads"
```

Results are saved to `trajectory/YYYYMMDD_HHMMSS_label/`.

## Hints

Hints let you guide the optimization agent with domain knowledge, constraints, or directions. There are three ways to provide them:

1. **During Session 1** — pass hints verbally or as a file path when Claude asks for inputs. These are written into the child environment's `HINTS.md`.
2. **Edit before Session 2** — after the child environment is created, edit `<child-env>/HINTS.md` directly to add or refine hints before starting optimization.
3. **Default** — if no hints are provided, a minimal default is used.

Example hints:

```markdown
- Focus on the large-N workloads (N > 4096), they dominate runtime
- Use fp32 accumulation to avoid precision failures
- Try shared memory tiling with tile size 128
- Do not use inline PTX — keep it portable
```

> **Note**: Session 1 never modifies files in this template repo. The `templates/hints.md` here is only a default template — to customize hints for a specific run, edit the child environment's `HINTS.md`.

## Advanced

### Web Search

Web search is **disabled by default**. To enable it when the agent hits an optimization plateau, edit `HINTS.md` before starting Session 2:

```markdown
- If 3 consecutive optimization rounds show no speedup improvement, use WebSearch
  to research optimization techniques specific to this kernel type, then apply
  what you find.
```

You can adjust the round threshold or add constraints on what to search for.

### Template Customization

| File | Purpose |
|------|---------|
| `templates/agent/claude.json` | Agent permissions for child environments |
| `templates/hints.md` | Default hints (used when no hints provided) |
| `templates/bench-wrapper.sh` | Benchmark wrapper with trajectory tracking |
| `templates/task.md` | Reference example of a generated child `CLAUDE.md` |

## FAQ

**Can I run multiple optimization tasks in parallel?**
Yes. Each child environment is independent. Run Session 1 multiple times with different kernels to create separate child environments, then optimize them concurrently.

**What if the benchmark fails after an optimization?**
Session 2 handles this automatically — it reads the failure output, attempts targeted fixes, and reverts (`git checkout solution/`) if the fix doesn't work.

**Can I resume an interrupted optimization?**
Yes. Just `cd` into the child environment and run `claude` again. The `CLAUDE.md` and git history provide full context.

**My bench script uses a remote evaluation service (e.g., Modal). Does that work?**
Yes. The framework is agnostic to how benchmarks are executed. As long as your bench script can be invoked from the command line and prints results to stdout, it works — whether execution happens locally, on a remote GPU, or through a cloud API.

**Can I manually edit the kernel between agent sessions?**
Yes. Edit files in `solution/`, then re-run `claude` in the child environment. The agent will pick up your changes.
