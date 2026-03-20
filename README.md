# AKO4ALL: Agentic Kernel Optimization for All

Automated GPU kernel optimization powered by coding agents. Provide any kernel — the agent iteratively rewrites it for maximum performance. Works with any coding agent; examples below use [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

## What You Provide

Only a kernel is required — everything else is optional.

- **Kernel** (required) — The kernel to optimize. Can be a single file or a directory. Supports Triton, CUDA, C++, TileLang, Python, or any language that can be benchmarked.
- **Reference implementation** (optional) — Used as the correctness golden. If absent, the original kernel is used.
- **Benchmark script** (optional) — Your own benchmark script. If not provided, the built-in [KernelBench](https://github.com/ScalingIntelligence/KernelBench) evaluator is used automatically (no setup needed beyond PyTorch). You can add a `GUIDE.md` to describe its usage, or let the agent figure it out by reading the script.
- **Hints** (optional) — Optimization hints, constraints, and agent behavior controls (e.g., whether to allow web search).

## Requirements

- A coding agent (e.g., [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code))
- Git
- Benchmark environment:
  - Built-in evaluator: Python >= 3.10, PyTorch with CUDA, NVIDIA GPU
  - Custom bench script: whatever your script requires

## Quick Start

1. Place your files:

```
AKO4ALL/
├── input/                       # Place your kernel files here
│   ├── kernel.py                # Example — can be any file(s) or subdirectory
│   └── reference.py             # Example — optional
├── bench/                       # Place your benchmark script here
│   ├── bench.sh                 # Example — can be any file(s) or subdirectory
│   ├── GUIDE.md                 # Optional
│   └── kernelbench/             # Built-in evaluator — delete if using your own
├── HINTS.md
```

2. Run:

```bash
cd AKO4ALL && claude
```

3. Start optimization (e.g., `Follow the instructions in TASK.md`).

## What Happens

The agent reads your files, copies the kernel to `solution/`, and enters a benchmark loop — optimizing, measuring, and reverting on failure. Each iteration is snapshot into `trajectory/` with the kernel source and benchmark output.

## Hints

Edit `HINTS.md` to guide the optimization. By default, web search is disabled — remove that line to enable it.

```markdown
- If 3 consecutive rounds show no improvement, use WebSearch for optimization ideas
- Focus on the large-N workloads (N > 4096), they dominate runtime
- Do not use inline PTX — keep it portable
```

## Permissions

The optimization loop involves running shell commands (compiling, benchmarking, profiling). By default, most coding agents prompt for approval on each command. To run fully unattended, grant the necessary permissions through your agent's configuration.

For Claude Code, the simplest option is to bypass all permission checks:

```bash
cd AKO4ALL && claude --dangerously-skip-permissions
```

For more granular control, create `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(*)", "Read(*)", "Write(*)", "Edit(*)",
      "Glob(*)", "Grep(*)", "Agent(*)"
    ]
  }
}
```

For other agents, consult their documentation on permission / auto-approve settings.

## FAQ

**What if the benchmark fails after an optimization?**
The agent reads the failure, attempts fixes, and reverts if needed.

**My bench script uses a remote service (e.g., Modal). Does that work?**
Yes. As long as your bench script runs from the command line and prints results to stdout.

**Can I manually edit the kernel between runs?**
Edit files in `solution/`, then tell the agent to continue.
