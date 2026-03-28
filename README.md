<h1 align="center">AKO4ALL</h1>
<p align="center"><b>Agentic Kernel Optimization for All</b></p>

<p align="center">
  <a href="https://tongminglaic.github.io/AKO"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://github.com/TongmingLAIC/AKO4FIB"><img src="https://img.shields.io/badge/GitHub-AKO4FIB-blue?logo=github" alt="AKO4FIB"></a>
  <img src="https://img.shields.io/badge/Tech%20Report-Coming%20Soon-gray" alt="Tech Report">
</p>

<p align="center"><b>If you find our work useful, please consider giving us a star 🌟</b></p>

Agentic Kernel Optimization for All — automated GPU kernel optimization powered by coding agents. Provide any kernel — the agent iteratively rewrites it for maximum performance. Works with any coding agent; examples below use [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

> Looking for standardized optimization against flashinfer-bench operators? Check out [AKO4FIB](https://github.com/TongmingLAIC/AKO4FIB).

<p align="center">
  <img src="assets/sol_001_optimization.png" alt="AKO4ALL optimization trajectory on SOL-ExecBench L1-001" width="100%">
</p>

*Optimization trajectory on [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) L1-001 (Attention Softmax+Dropout+Value Matmul Backward). 41 iterations, 8.93× average speedup over the PyTorch reference. Developed on A100-SXM4-80GB, ~2 hours total.*

## News

- 🎉 **[2026.03.28]** [AKO4FIB](https://github.com/TongmingLAIC/AKO4FIB) is now open source!
- 🚀 **[2026.03.24]** AKO4ALL is released. Check out the [project page](https://tongminglaic.github.io/AKO).

## What You Provide

Only a kernel is required — everything else is optional.

- **Kernel** (required) — The kernel to optimize. Can be a single file or a directory. Supports Triton, CUDA, C++, TileLang, Python, or any language that can be benchmarked.
- **Reference implementation** (optional) — Used as the correctness golden. If absent, the original kernel is used.
- **Benchmark script** (optional) — Your own benchmark script. A `GUIDE.md` can be included to describe its usage. If no benchmark script is provided, the built-in [KernelBench](https://github.com/ScalingIntelligence/KernelBench) evaluator is used automatically (no setup needed beyond PyTorch).
- **Context** (optional) — Reference materials for the agent: algorithm descriptions, papers, design docs, or any background knowledge that helps inform the optimization.
- **Hints** (optional) — Directives for the agent: optimization constraints, focus areas, and behavior controls (e.g., whether to allow web search).

> **Notes:** At least one set of input shapes for testing must be determinable — hardcoded in the kernel, reference, or bench script, or provided as additional files. The agent will ask if none can be found.

## Requirements

- A Coding Agent (e.g., [Claude Code](https://docs.anthropic.com/en/docs/claude-code))
- NVIDIA Nsight Compute (`ncu`)
- Benchmark environment:
  - Built-in evaluator: Python >= 3.10, PyTorch with CUDA, NVIDIA GPU
  - Custom bench script: whatever your script requires
- Git

> **Note:** The agent may install packages (`pip install`, `apt install`, etc.) to resolve missing dependencies. Running in a container or virtual environment is recommended. To restrict this behavior, use [Hints](#hints) (e.g., `Do not install any packages`) or [Permissions](#permissions).


## Quick Start

1. Place your files:

```
AKO4ALL/
├── input/                       # Place your kernel files here
│   ├── kernel.py                # Example — can be any file(s) or subdirectory
│   └── reference.py             # Optional
├── bench/                       # Place your benchmark script here (optional)
│   ├── bench.sh                 # Example — can be any file(s) or subdirectory
│   ├── GUIDE.md                 # Optional
│   └── kernelbench/             # Built-in evaluator — delete if using your own
├── context/                     # Place reference materials here (optional)
├── HINTS.md
```

2. Run (from the environment where dependencies are installed):

```bash
cd AKO4ALL && claude
```

3. Start optimization (e.g., `Follow the instructions in TASK.md`).

## What Happens

1. **Setup** — The agent reads your files, **creates an optimization branch**, copies the kernel to `solution/`, generates `bench.sh`, and verifies correctness.
2. **Profile** — Runs `ncu` on the baseline kernel to identify bottlenecks before optimizing.
3. **Iterate** — Each iteration: modify kernel → benchmark → log results to `ITERATIONS.md` → git commit. The agent uses profiling data, web search, and prior results to guide each attempt.
4. **Track** — Every iteration is saved to `trajectory/` with the kernel source and benchmark output.

## Hints

`HINTS.md` controls agent behavior. You can add directives such as:

- **Optimization constraints** — focus areas, language restrictions, techniques to avoid
- **Strategies** — specific approaches to try or skip
- **Agent behavior** — web search, verbosity, iteration limits
- **Dependency policies** — whether the agent may install packages

> **Notes:**
> - **Language switching** — The agent may rewrite your kernel in a different language (e.g., Triton → CUDA) to chase performance.
> - **Web search** — Web search is enabled by default. The agent will search for optimization ideas online after consecutive rounds without improvement.
>
> Edit `HINTS.md` to adjust these behaviors.

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

## Tips

- **Agent laziness** — Agents may default to conservative strategies: staying in PyTorch, only tuning configurations, or skipping profiling. If progress stalls, intervene with specific guidance.
- **Model matters** — Model capability strongly influences optimization quality. We recommend [Claude Opus 4.6](https://docs.anthropic.com/en/docs/about-claude/models).
- **Iteration limits** — By default, there is no iteration cap — the agent decides when to stop on its own. To enforce a limit, add a directive to `HINTS.md`, or your prompt (e.g., `Optimize for up to 30 iterations. Stop early only if all viable approaches are exhausted.`). For guidance on structuring open-ended agent tasks, see [autoresearch](https://github.com/karpathy/autoresearch)'s [`program.md`](https://github.com/karpathy/autoresearch/blob/master/program.md).

## Anti-Cheat

The agent is instructed via `TASK.md` to pursue genuine latency reduction and avoid reward hacking (stream injection, timing manipulation, returning uninitialized results, etc.). The built-in KernelBench evaluator also provides runtime defenses: excessive speedup flagging (>10× triggers a warning) and input shape protection (the solution's `get_inputs`/`get_init_inputs` are replaced by the reference's to prevent trivializing the workload).

For stricter enforcement, add directives to `HINTS.md` or provide a custom bench script in `bench/` with built-in static analysis. KernelBench's [`kernel_static_checker.py`](https://github.com/ScalingIntelligence/KernelBench) is a good starting point.

## Example: SOL-ExecBench

[SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) contains 235 real-world DL kernel problems from NVIDIA. This example shows how to optimize any of them with AKO4ALL — no file copying needed.

1. Clone both repositories. Set up the SOL-ExecBench environment and install `ncu` if needed.

2. Activate the SOL-ExecBench environment, then start the agent in the AKO4ALL directory:

```bash
cd AKO4ALL && claude
```

3. Give it a prompt (replace `N` and the paths):

```
Follow the instructions in TASK.md. Save HINTS.md to memory.
Optimize for up to N iterations. Stop early only if all viable approaches are exhausted.
Bench is SOL-ExecBench: <path/to/SOL-ExecBench>.
Input is <path/to/SOL-ExecBench>/data/benchmark/L1/001_attention_softmax_dropout_value_matmul_backward.
Benchmark with SOL-ExecBench. All dependencies for SOL-ExecBench are already installed — use them directly.
```

The agent handles the rest — reads the problem definition, sets up the benchmark, and starts iterating.

## FAQ

**What if the benchmark fails after an optimization?**
The agent reads the failure, attempts fixes, and reverts if needed.

**My bench script uses a remote service (e.g., Modal). Does that work?**
Yes. As long as your bench script runs from the command line and prints results to stdout.

**Can I intervene during optimization?**
Yes. You can interrupt the agent at any time to give guidance, discuss strategy, or manually edit files in `solution/`. Then tell the agent to continue.

## Tech Report

Coming soon — our tech report will detail why we advocate agentic approaches over fixed-pipeline methods for kernel optimization.

## Acknowledgments

We would like to thank the following open-source projects that inspired and supported the development of AKO:

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) — for providing the benchmark and evaluation format used by AKO4ALL's built-in evaluator.
- [autoresearch](https://github.com/karpathy/autoresearch) and [autokernel](https://github.com/RightNow-AI/autokernel) — AKO's design was inspired by their work on autonomous optimization loops.
