---
name: ako4all
description: Drive an agentic loop that iteratively optimizes a GPU kernel for maximum speedup. Use this skill whenever the user wants to optimize / speed up / benchmark a GPU kernel (CUDA, Triton, TileLang, C++, Python), mentions AKO / AKO4ALL / AKO4FIB / agentic kernel optimization, asks to "make this kernel faster", or has a kernel they want measured against a PyTorch reference. The skill handles setup, profiling (ncu), correctness checking, iteration logging, and git commits. Bootstraps a workspace in any directory the user points at.
---

# AKO4ALL — Agentic Kernel Optimization

Drive a profile → modify → benchmark → log → commit loop on a GPU kernel until it runs faster than the reference. The user provides at minimum a kernel; everything else (reference, inputs, bench script, hints) is optional.

## When this skill applies

- "optimize this kernel" / "speed up this CUDA / Triton / TileLang kernel"
- "run AKO / AKO4ALL on ..."
- "benchmark this kernel against PyTorch"
- "iterate on this kernel until it's faster"
- mentions of `ncu`, kernel profiling, GPU speedup target

Does NOT apply when:
- User wants to *write* a new kernel from scratch with no optimization target — just write code, no loop.
- User wants Codex / GPT to review or implement — use `codex:rescue` instead.
- User wants generic performance advice for code that isn't a GPU kernel.

## First action

Before doing anything else, establish the **workspace** — the directory the loop runs in. It is typically the user's CWD, or a subdirectory / path they name in the prompt.

### Inventory the workspace + prompt

Browse the workspace (don't run a fixed checklist — look around) and read the user's prompt to identify what the loop needs:

- **Kernel** (required) — the code to optimize
- **Reference** (optional) — correctness golden
- **Input data** (optional) — data files the kernel consumes (`.npz`, `.bin`, shape lists, custom formats, etc.)
- **Knowledge** (optional) — reference materials the user wants you to consult: algorithm notes, papers, design docs, prior PRs. Typically under `knowledge/` but anywhere the user points at.
- **Bench mode** — user-provided bench script vs. default `bench/kernelbench/` evaluator
- **Scaffold presence** — whether `bench-wrapper.sh`, `HINTS.md`, `ITERATIONS.md`, `bench/kernelbench/` are already at workspace root

Whether the workspace follows AKO4ALL's `source/` / `knowledge/` / `bench/` naming or some entirely different layout is **not** the signal. What matters is whether you can identify each item above with confidence.

### Ask only when genuinely uncertain

If the user's prompt + filesystem give you confidence about every required item, **don't ask** — skip straight to presenting the plan. Ask only when a piece's role is genuinely ambiguous (a kernel-shaped file with no obvious reference, two files that could both be the kernel, an input data file in an unfamiliar format you need permission to wire up a custom way, etc.). When in doubt, asking is cheaper than guessing wrong.

### Always present the resolved plan before running anything

Whether you asked the user or not, list back what you decided — so the user can correct you even when you didn't think you needed to ask.

Use the format below. Bold field labels + inline-code path values + the leading emoji marker make the plan visually scannable in any terminal theme (don't flatten to a wall of prose):

**📋 Resolved Plan**

- **Workspace** — `<path>`
- **Kernel** — `<path>`
- **Reference** — `<path>` *(or none — will use original kernel)*
- **Input data** — `<path>` *(or inline in ref, or none)*
- **Knowledge** — `<path>` *(or none)*
- **Bench mode** — default (KernelBench) *(or custom: `<path>`)*
- **Scaffold to copy** — `<list of missing files>` *(or none — already present)*

If anything still feels uncertain at this point, **stop and ask**. Otherwise proceed to Workflow.

### Bringing in scaffold

When copying scaffold (`bench-wrapper.sh`, `bench/kernelbench/`, starter `HINTS.md` / `ITERATIONS.md`) from this skill's own directory into the workspace, **do not overwrite** files that already exist — the user may have edited `HINTS.md`, or `ITERATIONS.md` may carry prior iteration history. Copy only what's missing.

### Persisting user-supplied hints

The user may supply behavior directives in two ways:
- **Inline in the prompt** — e.g., "do not use shared memory" or "prefer Triton".
- **External file reference** — e.g., "follow rules in /tmp/x.md" or "see hints.md".

In **both cases**, merge those directives into `HINTS.md` alongside the defaults
(do not overwrite the defaults). `HINTS.md` is the persistence layer — directives
that only live in the current session's plan are lost on resume.

### Surfacing HINTS.md changes

Whenever you merge directives into `HINTS.md`, tell the user explicitly what
happened. Example phrasings:

> "I merged your 2 directives into HINTS.md alongside the 2 default entries."
> "I added the 3 rules from /tmp/user-hints.md to HINTS.md; the 2 defaults are preserved."

Without this acknowledgment the user can't tell from your reply whether you
added, replaced, or silently dropped their directives. Always name the **source**
("from your prompt" / "from /tmp/x.md") and the **count of defaults retained**.

## Workflow

1. **Analyze inputs.** Building on the inventory above, confirm `class Model` and `get_inputs()` can be assembled for default bench mode; if not, **stop and ask the user**. See `bench/kernelbench/GUIDE.md` for the input assembly contract (KernelBench-format input / raw kernel / kernel + separate data file / external path patterns).
2. **Create branch.** `git checkout -b opt/<kernel-name>`. If the workspace isn't a git repo, init one first.
3. **Initialize solution.** Create `solution/` and `scripts/`. Copy the kernel implementation files into `solution/` (only the kernel itself — reference / inputs helper files stay at their resolved locations).
4. **Generate bench.sh.** Build the bench command with adjusted paths, pipe through `2>&1 | tee _bench_output.txt`. Replace `{{BENCH_COMMAND}}` in `bench-wrapper.sh` to produce `scripts/bench.sh`. For default bench mode the command is `python bench/kernelbench/bench.py --ref <ref> --solution solution/<kernel> [--inputs <inputs-file>] --verbose` — include `--inputs` only when inputs are defined outside the ref file.
5. **Verify baseline.** Run `bash scripts/bench.sh`. Expect `CORRECT=True`. If not, diagnose and fix before iterating. Commit: `git add -A && git commit -m "[baseline] Initialize solution and benchmark"`. Then run `ncu` once on the baseline to inform iter-1 direction.

## Iteration protocol

Every modification to `solution/` followed by a bench run = one iteration. Number sequentially (1, 2, 3, …). **All three steps below MUST complete before you do anything else — including starting the next iteration, replying to the user, or yielding the turn. Skipping any breaks the trajectory log.**

A failed or partial bench is still an iteration — complete steps 2 and 3 anyway. Silent abandonment loses the trajectory and makes the next iter's analysis harder.

1. `bash scripts/bench.sh iter-N` — label is required, must match `iter-N` format.
2. Append a structured entry to `ITERATIONS.md` (template inside that file).
3. `git commit -m "[iter N] <short description of optimization direction>"`.

Use `ncu` to profile and identify bottlenecks — do not optimize blindly. Read `HINTS.md` for stall rules (default: re-profile + WebSearch after 3 stagnant iterations).

## Gotchas

- **Pursue genuine latency reduction, not reward hacking.** No CUDA stream injection to evade timing, no monkey-patching the benchmark, no returning uninitialized results. The built-in evaluator flags >10× speedups for a reason — investigate before celebrating.
- **The solution file must not contain `get_inputs` / `get_init_inputs`.** The bench script strips the solution's module-level tail before eval as an anti-cheat boundary. Inputs come from the reference or `--inputs` file, never the solution.
- **`get_inputs()` must produce fresh data each call.** Bench calls it 5+ times across trials. Module-level cached tensors make correctness checks trivially pass and let timing measure cache-warm performance. Use `torch.randn` or reload from disk on every call.
- **Don't be lazy.** Stay-in-PyTorch, only-tune-configurations, skip-profiling — these are the default low-effort failure modes for agents. The point of the loop is to *rewrite* the kernel — switch languages (Triton → CUDA, etc.) when it helps.
- **Iteration cap is the user's call.** Default: no cap, decide when to stop yourself. If `HINTS.md` or the prompt names a limit, honor it. Stop early only when all viable directions are exhausted.

## Reference files

- `bench/kernelbench/GUIDE.md` — full input assembly patterns, CLI flags, timing methods, tolerances. Read this before writing the bench command if anything about input shape, precision, or backend is non-obvious.
- `HINTS.md` (workspace) — user-editable behavior directives. Read at session start; respect any constraint named there.
- `ITERATIONS.md` (workspace) — your own iteration log. Write to it every iteration.
