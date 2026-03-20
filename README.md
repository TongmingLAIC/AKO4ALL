# AKO4ALL: Agentic Kernel Optimization for All

Automated GPU kernel optimization powered by coding agents. Provide any kernel — the agent iteratively rewrites it for maximum performance. Works with any coding agent; examples below use [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

## What You Provide

Only a kernel is required — everything else is optional.

- **Kernel** (required) — The kernel to optimize. Can be a single file or a directory. Supports Triton, CUDA, C++, TileLang, Python, or any language that can be benchmarked. Must include at least one set of input shapes for testing (in the kernel itself, reference, bench script, or hints) — the agent will ask if none can be determined.
- **Reference implementation** (optional) — Used as the correctness golden. If absent, the original kernel is used.
- **Benchmark script** (optional) — Your own benchmark script, with an optional `GUIDE.md` describing its usage. If omitted, the built-in [KernelBench](https://github.com/ScalingIntelligence/KernelBench) evaluator is used automatically (no setup needed beyond PyTorch).
- **Context** (optional) — Reference materials for the agent: algorithm descriptions, papers, design docs, or any background knowledge that helps inform the optimization. Place in `context/`.
- **Hints** (optional) — Directives for the agent: optimization constraints, focus areas, and behavior controls (e.g., whether to allow web search).

## Requirements

- A coding agent (e.g., [Claude Code](https://docs.anthropic.com/en/docs/claude-code))
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
├── context/                     # Place reference materials here (optional)
│   ├── algorithm.md             # Example — algorithm description
│   └── paper.pdf                # Example — related paper
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

Edit `HINTS.md` to guide the optimization. Examples:

```markdown
- Focus on the large-N workloads (N > 4096), they dominate runtime
- Do not use inline PTX — keep it portable
- Stay in Triton — do not rewrite the kernel in CUDA or other languages
- If 3 consecutive rounds show no improvement, use WebSearch for optimization ideas
```

> **Tips:**
> - **Language switching** — The agent may rewrite your kernel in a different language (e.g., Triton → CUDA) to chase performance. Add a constraint in `HINTS.md` if you want to keep the original language.
> - **Web search** — Web search is enabled by default. The agent will search for optimization ideas online after consecutive rounds without improvement. Edit `HINTS.md` to disable or adjust this behavior.

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
      "Glob(*)", "Grep(*)", "Agent(*)",
      "WebFetch(*)", "WebSearch(*)"
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

**Can I intervene during optimization?**
Yes. You can interrupt the agent at any time to give guidance, discuss strategy, or manually edit files in `solution/`. Then tell the agent to continue.
