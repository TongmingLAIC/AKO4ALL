# AutoKernelForge

Automated GPU kernel optimization powered by Claude Code. Provide any kernel and its benchmark script — Claude Code analyzes them and iteratively rewrites the kernel for maximum performance.

Supports Triton, CUDA, C++, TileLang, Python — any kernel that can be benchmarked by a script.

## Prerequisites

- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)
- Git
- A working benchmark environment (local GPU, [Modal](https://modal.com/), remote cluster, etc.) — make sure your bench script runs before starting

## Quick Start

1. Place your files:

```
AutoKernelForge/
├── input/                       # Your kernel files (required)
│   ├── kernel.py                # The kernel to optimize
│   └── reference.py             # Optional — correctness golden
├── bench/                       # Your benchmark script + deps (required)
│   └── bench.py
└── HINTS.md                     # Optimization hints (edit to add your own)
```

2. Run:

```bash
cd AutoKernelForge
claude
```

Claude Code reads your files, sets up the workspace, and begins optimizing.

If no `reference.py` is provided, the original kernel in `input/` is used as the correctness golden.

### Example

```
> cd AutoKernelForge && claude

Claude: [reads input/, bench/, and HINTS.md]
        [copies kernel to solution/, runs baseline benchmark]
        [rewrites kernel with tiling optimization]
        [benchmarks -> 1.8x speedup, all PASSED]
        [tries vectorized loads -> 2.3x speedup]
        [tries warp-level reduction -> 2.5x speedup, but 1 case FAILED]
        [reverts, tries alternative -> 2.4x, all PASSED]
        Final result: 2.4x mean speedup across all workloads
```

## Workspace Layout

After setup, the full repo looks like:

```
AutoKernelForge/
├── CLAUDE.md
├── input/                      # User-provided originals (read-only)
├── bench/                      # Benchmark script + deps (read-only)
├── HINTS.md                    # Optimization hints
├── bench.sh                    # Generated benchmark wrapper (read-only)
├── solution/                   # Kernel files — only these are edited
├── scripts/                    # Workspace for profiling/debug tools
└── trajectory/                 # Auto-created on first benchmark run
    ├── 20260319_143022_baseline/
    │   ├── kernel.py           # Kernel snapshot
    │   └── output.txt          # Benchmark output
    └── ...
```

Every `bash bench.sh "label"` snapshots `solution/` and benchmark output into `trajectory/`.

## Hints

Edit `HINTS.md` in the repo root to add optimization hints or constraints:

```markdown
- Focus on the large-N workloads (N > 4096), they dominate runtime
- Use fp32 accumulation to avoid precision failures
- Try shared memory tiling with tile size 128
- Do not use inline PTX — keep it portable
```

### Web Search

Disabled by default. To enable when the agent hits a plateau, include in your hints:

```markdown
- If 3 consecutive optimization rounds show no speedup improvement, use WebSearch
  to research optimization techniques specific to this kernel type, then apply
  what you find.
```

## FAQ

**What if the benchmark fails after an optimization?**
The agent reads the failure, attempts fixes, and reverts if needed.

**My bench script uses a remote service (e.g., Modal). Does that work?**
Yes. As long as your bench script runs from the command line and prints results to stdout.

**Can I manually edit the kernel between runs?**
Edit files in `solution/`, then tell Claude to continue.
