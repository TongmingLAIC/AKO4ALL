<h1 align="center">AKO4ALL</h1>
<p align="center"><b>Agentic Kernel Optimization for All</b></p>

<p align="center">
  <a href="https://tongminglaic.github.io/AKO"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://github.com/TongmingLAIC/AKO4FIB"><img src="https://img.shields.io/badge/GitHub-AKO4FIB-blue?logo=github" alt="AKO4FIB"></a>
  <img src="https://img.shields.io/badge/Tech%20Report-Coming%20Soon-gray" alt="Tech Report">
</p>

<p align="center"><b>If you find our work useful, please consider giving us a star 🌟</b></p>

Agentic Kernel Optimization for All — automated GPU kernel optimization powered by coding agents. Provide any kernel — the agent iteratively rewrites it for maximum performance. Ships as a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill; other coding agents can drive the loop by following `SKILL.md` as a protocol document.

**AKO4ALL is the lightweight drop-in-a-kernel-get-it-optimized path** — single-session, minimal infra; your benchmark (BYO script or built-in KernelBench) is a shell command the agent invokes directly. For **batch evaluation against the standardized flashinfer-bench operator suite** (comparable speedups across many kernels), see the sibling project [AKO4FIB](https://github.com/TongmingLAIC/AKO4FIB) (coming soon — open-sourced after the [MLSys 2026 competition](https://mlsys26.flashinfer.ai/)).

## What You Provide

Only a kernel is required — everything else is optional.

- **Kernel** (required) — The kernel to optimize. Can be a single file or a directory. Supports Triton, CUDA, C++, TileLang, Python, or any language that can be benchmarked.
- **Reference implementation** (optional) — Used as the correctness golden. If absent, the original kernel is used.
- **Inputs** (optional) — Sample input data, in any form: hardcoded inside the kernel/reference, a separate Python file defining `get_inputs()`, or raw data files (`.npz`, `.bin`, shape lists, etc.) that the agent wires up itself. The agent assembles whatever is provided into something the bench script can call.
- **Benchmark script** (optional) — Your own benchmark script. A `GUIDE.md` can be included to describe its usage. If no benchmark script is provided, the built-in [KernelBench](https://github.com/ScalingIntelligence/KernelBench) evaluator is used automatically (no setup needed beyond PyTorch).
- **Knowledge** (optional) — Reference materials for the agent: algorithm descriptions, papers, design docs, or any background knowledge that helps inform the optimization.
- **Hints** (optional) — Directives for the agent: optimization constraints, focus areas, and behavior controls (e.g., whether to allow web search).

> **Notes:** At least one set of input shapes must be determinable from what you provide. If none can be found, the agent will ask.

## Install

Clone the repo directly into Claude Code's skills directory:

```bash
git clone https://github.com/TongmingLAIC/AKO4ALL.git ~/.claude/skills/ako4all
```

Or, if you already have it cloned somewhere else, symlink it:

```bash
ln -s /path/to/AKO4ALL ~/.claude/skills/ako4all
```

## How to Use

### Mode 1 — Interactive (default)

Open Claude Code in a directory containing your kernel, then ask it to optimize the kernel. Files in the working directory are inspected automatically; you can also point at external paths. Optional constraints — iteration cap, language preference, dependency policy, etc. — included in the prompt will be merged into `HINTS.md`.

The skill presents a resolved plan, asks if anything is genuinely ambiguous, and starts iterating. You can interrupt at any point to redirect.

<details>
<summary>Optional: pre-organize your workspace</summary>

If you want the conventional layout the skill recognizes:

```
<workspace>/
├── source/                      # Kernel + optional reference and inputs
│   ├── kernel.py
│   ├── reference.py             # Optional
│   └── inputs.py / data.npz ... # Optional — input data, any format
├── bench/                       # Your own benchmark script (optional)
│   └── ...
├── knowledge/                   # Reference materials (optional)
└── HINTS.md                     # Agent directives (optional)
```

These are conventions, not requirements. If your files are organized differently, just point Claude at them.

</details>

### Mode 2 — Batch via subagents (advanced)

For optimizing several related kernels in parallel, in Claude Code the main agent can spawn one subagent per kernel via the `Task` tool.

> ⚠️ **Subagents can't ask back.** In Claude Code, subagents lack `AskUserQuestion`. The main agent composes each subagent's prompt from your instructions plus what's in each kernel's directory; any decision you don't pin down there will fall to `SKILL.md` defaults. So include the things you care about (per-kernel paths, optional reference / inputs locations, language or iteration-cap constraints, custom branch names) — and let the rest default.

See the [FAQ](#faq) for an example.

## Requirements

- A coding agent (e.g., [Claude Code](https://docs.anthropic.com/en/docs/claude-code))
- NVIDIA GPU with CUDA-enabled PyTorch (only for the built-in evaluator; a remote-execution bench script may not need either locally)
- Python **>= 3.10** for the built-in evaluator (or whatever your custom bench script requires)
- NVIDIA Nsight Compute (`ncu`) — recommended; its version must match your CUDA / PyTorch CUDA build. If unavailable, the loop still proceeds (no profiling — the agent reasons from runtime stats instead).
- Git

> **Note:** During the loop the agent may run `pip install` / `apt install` etc. to fill in missing dependencies. A container or virtual environment is recommended for isolation; to forbid installs, see [Hints](#hints) or [Permissions](#permissions).

## How It Works

The agent creates an `opt/<kernel>` branch, copies your kernel to `solution/`, generates a bench script, and runs a profile → modify → benchmark → log → commit loop. Profiling uses `ncu`; iteration history lives in `ITERATIONS.md`. See [`SKILL.md`](SKILL.md) for the full protocol.

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
claude --dangerously-skip-permissions
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

## Repo Layout

```
AKO4ALL/
├── SKILL.md              # Skill protocol — entry point the agent loads
├── HINTS.md              # Default behavior directives (copied to workspace)
├── ITERATIONS.md         # Iteration log template (copied to workspace)
├── bench-wrapper.sh      # Template for scripts/bench.sh (copied to workspace)
└── bench/kernelbench/    # Built-in KernelBench evaluator (copied to workspace)
    ├── bench.py
    └── GUIDE.md
```

`SKILL.md` is what Claude Code loads when the skill triggers. The other listed files are scaffold the skill copies into the user's workspace when bootstrapping the loop. Other repo-root files (`README.md`, `LICENSE`, `assets/`) are project metadata, not installed into the workspace.

## Example: SOL-ExecBench

[SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) contains 235 real-world DL kernel problems from NVIDIA. This example shows how to optimize any of them with AKO4ALL — no file copying needed.

1. Install the skill (see [Install](#install)). Clone SOL-ExecBench separately, set up its environment, and install `ncu` if needed.

2. Activate the SOL-ExecBench environment, open Claude Code in any working directory:

```bash
claude
```

3. Give it a prompt (replace `N` and the paths):

```
Optimize the kernel at <path/to/SOL-ExecBench>/data/benchmark/L1/001_attention_softmax_dropout_value_matmul_backward using AKO.
Save HINTS.md to memory.
Optimize for up to N iterations. Stop early only if all viable approaches are exhausted.
Bench is SOL-ExecBench: <path/to/SOL-ExecBench>.
Benchmark with SOL-ExecBench. All dependencies for SOL-ExecBench are already installed — use them directly.
```

The skill triggers, inspects the problem definition, sets up the benchmark, and starts iterating.

## Anti-Cheat

The skill instructs the agent to pursue genuine latency reduction (no stream injection, timing manipulation, or returning uninitialized results), and the built-in KernelBench evaluator flags >10× speedups and strips the solution's module-level tail before eval so it cannot influence its own workload. For stricter enforcement, add directives to `HINTS.md` or provide a custom bench script with static analysis (e.g., KernelBench's [`kernel_static_checker.py`](https://github.com/ScalingIntelligence/KernelBench)). See [`SKILL.md`](SKILL.md) "Gotchas" for the full agent-side rules.

## FAQ

**Can I use AKO4ALL for a whole benchmark suite?**
Not designed for it — single-session, drop-in is the design center. For batch evaluation across many operators with comparable speedups, see [AKO4FIB](https://github.com/TongmingLAIC/AKO4FIB).

**My agent isn't trying hard enough — keeps tuning configs instead of rewriting.**
Agent laziness is a known failure mode (staying in PyTorch, only tuning configurations, skipping profiling). Intervene with specific guidance, or add directives to `HINTS.md` ahead of time. The point of the loop is to *rewrite* the kernel — switch languages (Triton → CUDA, etc.) when it helps.

**Which model should I use?**
Model capability strongly influences optimization quality. We recommend [Claude Opus 4.6](https://docs.anthropic.com/en/docs/about-claude/models).

**How do I cap iterations?**
By default there is no cap — the agent decides when to stop on its own. To enforce a limit, add a directive to `HINTS.md`, or include it in your prompt (e.g., `Optimize for up to 30 iterations. Stop early only if all viable approaches are exhausted.`). For guidance on structuring open-ended agent tasks, see [autoresearch](https://github.com/karpathy/autoresearch)'s [`program.md`](https://github.com/karpathy/autoresearch/blob/master/program.md).

**How do I spawn subagents to optimize multiple kernels in parallel?**
Tell the main agent up-front, with the per-kernel info it needs:

> Use AKO4ALL to optimize three kernels in parallel:
> - `/work/relu/kernel.py` — no separate reference, inline inputs
> - `/work/softmax/kernel.py` — reference at `/work/softmax/torch_ref.py`
> - `/work/matmul/kernel.py` — constraint: stay in Triton
>
> Iter cap 10 each.

In Claude Code, subagents can't ask back, so anything you don't specify in the prompt (or that isn't in the kernel's directory) falls to `SKILL.md` defaults.

**What if the benchmark fails after an optimization?**
The agent reads the failure, attempts fixes, and reverts if needed.

**My bench script uses a remote service (e.g., Modal). Does that work?**
Yes. As long as your bench script runs from the command line and prints results to stdout. The local machine doesn't need a GPU / CUDA / PyTorch in that case — just whatever your wrapper script imports.

**Can I intervene during optimization?**
Yes. You can interrupt the agent at any time to give guidance, discuss strategy, or manually edit files in `solution/`. Then tell the agent to continue.

## Tech Report

Coming soon — our tech report will detail why we advocate agentic approaches over fixed-pipeline methods for kernel optimization.

## Acknowledgments

We would like to thank the following open-source projects that inspired and supported the development of AKO:

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) — for providing the benchmark and evaluation format used by AKO4ALL's built-in evaluator.
- [autoresearch](https://github.com/karpathy/autoresearch) and [autokernel](https://github.com/RightNow-AI/autokernel) — AKO's design was inspired by their work on autonomous optimization loops.
