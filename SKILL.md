---
name: ako4all
description: Iteratively optimize a GPU kernel with correctness checks, profiling, measured benchmarks, and recoverable experiment history. Use in Codex or Claude Code when the user requests an AKO4ALL optimization run or asks to make a concrete CUDA, Triton, or Metal kernel faster, including Apple Silicon on-device inference. A profiling-only or benchmark-only request does not authorize changing the implementation.
---

# AKO4ALL — Agentic Kernel Optimization

Drive a profile → modify → benchmark → log → commit loop on a GPU kernel. Preserve the best correct implementation within the requested budget; a speedup is an experimental result, not a guaranteed outcome. Identify a runnable kernel and test inputs before measuring.

In Codex, read [references/codex.md](references/codex.md) for installation, invocation, and session recovery. This is a workflow for the current coding agent; it does not require launching Claude Code or another model.

## Hardware and runtime route

For Apple Silicon / Metal kernels or on-device inference, read
[references/metal.md](references/metal.md) and use `bench/metal/evaluate.py` with
the target repository's native correctness and timing commands. That route
replaces the CUDA evaluator, `ncu`, and `solution/`-copy assumptions below;
it retains the shared budget, evidence, recovery, and final-verification rules.
Copy the `bench/metal/` resources from the resolved skill root if needed. Never
import CUDA/PyTorch just to benchmark a Metal task. A kernel-only pass does not
qualify a model change; the inference route adds the repository's output gate.

For CUDA-backed kernels, use the existing KernelBench or user-provided evaluator
below. The chosen runtime and numerical contract stay fixed throughout the run.

## When this skill applies

- "optimize this kernel" / "speed up this CUDA / Triton / TileLang / Metal kernel"
- "optimize Apple Silicon on-device inference with measured output and performance gates"
- "run AKO / AKO4ALL on ..."
- "iterate on this kernel until it's faster"
- a concrete GPU speedup target with permission to modify the kernel

Does NOT apply when:
- User wants to *write* a new kernel from scratch with no optimization target — just write code, no loop.
- User only requests a review, profiling, or a benchmark. Perform that requested work without starting an optimization loop.
- User wants generic performance advice for code that isn't a GPU kernel.

## First action

Establish the **workspace** — the directory the loop runs in — and read its applicable `AGENTS.md` / contribution rules before editing. Follow the user's chosen runtime, repository layout, branch, and authorization for GPU use, dependency installation, commits, and publication. The skill supplies no permission to publish or to run a model. Use an isolated experiment workspace when the repository excludes experiment logs; keep its rules applicable to any source copied there.

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

Present the resolved fields concisely in the host's normal progress format:

- **Workspace** — `<path>`
- **Kernel** — `<path>`
- **Reference** — `<path>` *(or none — will use original kernel)*
- **Input data** — `<path>` *(or inline in ref, or none)*
- **Knowledge** — `<path>` *(or none)*
- **Bench mode** — default (KernelBench) *(or custom: `<path>`)*
- **Budget and acceptance** — iteration/time limit, shapes, numerical gate, measurement boundary, and required GPU authorization
- **Scaffold to copy** — `<list of missing files>` *(or none — already present)*

If anything still feels uncertain at this point, **stop and ask**. Otherwise proceed to Workflow.

### Bringing in scaffold

Resolve scaffold paths from the loaded `SKILL.md` directory, not the task's CWD. When copying scaffold (`bench-wrapper.sh`, `bench/kernelbench/`, starter `HINTS.md` / `ITERATIONS.md`, `workspace.gitignore` → as `.gitignore` in the workspace), **do not overwrite** files that already exist — the user may have edited `HINTS.md`, or `ITERATIONS.md` may carry prior iteration history. Copy only what's missing.

### Persisting user-supplied hints

The user may supply behavior directives in two ways:
- **Inline in the prompt** — e.g., "do not use shared memory" or "prefer Triton".
- **External file reference** — e.g., "follow rules in /tmp/x.md" or "see hints.md".

In **both cases**, merge those directives into `HINTS.md`. It's the persistence
layer — directives that only live in the current session's plan are lost on resume.

### Surfacing HINTS.md changes

Whenever you merge directives into `HINTS.md`, tell the user explicitly what
happened. Example phrasings:

> "I added your 'avoid shared memory' directive from the prompt to HINTS.md."
> "I added the 3 rules from /tmp/user-hints.md to HINTS.md."

Without this acknowledgment the user can't tell from your reply whether you
added, replaced, or silently dropped their directives. Always name the **source**
("from your prompt" / "from /tmp/x.md").

## Workflow

1. **Analyze inputs.** Building on the inventory above, confirm `class Model` and `get_inputs()` can be assembled for default bench mode; if not, **stop and ask the user**. See `bench/kernelbench/GUIDE.md` for the input assembly contract (KernelBench-format input / raw kernel / kernel + separate data file / external path patterns).
2. **Choose branch.** Reuse the task branch when one exists. Otherwise create an experiment branch following repository conventions (`codex/opt-<kernel-name>` in Codex, `opt/<kernel-name>` in Claude Code). Preserve unrelated work. Initialize Git only in a new experiment directory intended for this run.
3. **Initialize solution.** Create `solution/` and `scripts/`. Copy the kernel implementation files into `solution/` (only the kernel itself — reference / inputs helper files stay at their resolved locations). **Do not copy or `mkdir` canonical directories** (`source/`, `input/`, etc.) when the user's files already exist elsewhere. Point bench.sh's `--ref` and `--inputs` flags at the resolved paths in place. `solution/` is the only directory the loop owns.
4. **Generate bench.sh.** Build the bench command with adjusted paths, pipe through `2>&1 | tee _bench_output.txt`. Replace `{{BENCH_COMMAND}}` in `bench-wrapper.sh` to produce `scripts/bench.sh`. For default bench mode the command is `python bench/kernelbench/bench.py --ref <ref> --solution solution/<kernel> [--inputs <inputs-file>] --verbose` — include `--inputs` only when inputs are defined outside the ref file. **Do not hardcode `--backend`** in the rendered command; bench.py auto-detects backend from solution source. Add `--backend` only to override the sniff (explicit HIP labelling or mixed-backend solutions). `scripts/bench.sh` is a starting template — when the bench env needs setup (conda activate, sub-env python paths, multi-CUDA toolkit selection), edit it freely; preserve only the trajectory section (LABEL/TIMESTAMP handling and `cp -r solution/* "$TRAJ_DIR/"`).

   **Common env friction:** base shell often has no `python` on PATH when python lives in a sub-env (e.g. `~/anaconda3/envs/py312/bin/python`). Tools that internally subprocess `python` (sol-execbench CLI, torch `cpp_extension.load_inline`) will then fail with `command not found`. Workaround: put `PATH=<env-bin>:$PATH` at the top of `scripts/bench.sh` (or `source <conda>/etc/profile.d/conda.sh && conda activate <env>`). Discover sub-envs via `ls /home/*/anaconda3/envs/*/bin /root/*/envs/*/bin /opt/conda/envs/*/bin 2>/dev/null`.
5. **Verify baseline.** Run `bash scripts/bench.sh`. Expect `CORRECT=True`. If not, diagnose and fix before iterating. If commits are allowed, stage only this experiment's source and allowed records, inspect the staged diff, then commit with `[baseline] Initialize solution and benchmark`. Do not stage unrelated changes, models, binaries, credentials, or repository-excluded logs. Then run `ncu` once on the baseline to inform iter-1 direction.

## Iteration protocol

Every modification to `solution/` followed by a bench run = one iteration. Number sequentially (1, 2, 3, …). Complete each iteration in this order:

1. `bash scripts/bench.sh iter-N` — label is required, must match `iter-N` format.
2. Append a structured entry to `ITERATIONS.md` (template inside that file).
3. Stage only the experiment's changed files, inspect the staged diff, and commit with `[iter N] <short description of optimization direction>` when the task permits commits. Otherwise keep the code snapshot and record that it is uncommitted.

Finish the log and snapshot/commit before starting the next hypothesis. Tool-call counts are not part of the protocol: Codex may use several calls to read evidence, stage, and commit. A failed or partial bench is still an iteration; preserve its actual output and failure before debugging. Never invent missing results from memory.

After an interruption or context compaction, read `HINTS.md`, `ITERATIONS.md`, the latest trajectory, and Git status/history. Reconcile any measured but unlogged iteration before editing again. Never rerun a completed label over its evidence. Use a fresh retry label and account for it within the original budget.

Profile to identify bottlenecks — see "ncu profiling" below for the ncu workflow and analytical fallback. Do not optimize blindly.

## Keeping the iteration loop fast

A bench run must be cheap enough to iterate against (seconds to low minutes). When it isn't, the cause is almost always an **expensive reference** — it's re-run for every correctness trial *and* re-timed for the speedup denominator, yet it's **invariant across solution edits**, so most of that cost is wasted. This is an eval-*time* problem, not a metric problem: never change *what* you compare against to make the bench cheaper.

Separate the per-iteration **signal** from the final **verdict**:

- **Signal** (every iter): rank candidates by the **solution's own `RUNTIME`** (lower is better) — the reference contributes nothing to comparing two solutions.
- **Verdict** (before committing a winner / a `final`): a full run — full trial counts, reference measured, real `SPEEDUP`, reward-hack check.

**Whose eval is it determines what you may touch:**

- **Default bench** (`bench/kernelbench/bench.py` — the skill owns it): pull levers freely, cheapest-and-safest first — `--no-ref` (skip reference timing; `REF_RUNTIME`/`SPEEDUP` → -1, `COMPILED`/`CORRECT`/`RUNTIME` unaffected) → trim `--num-perf-trials` (e.g. 50→20; latency noise only) → trim `--num-correct-trials` (higher risk: weakens the fresh-input anti-cheat and still runs the ref once per trial, so keep ≥1 in the loop, full at the gate). Keep the input regime **fixed across the whole run**: `--fresh-inputs` (the default) and `--no-fresh-inputs` measure different quantities, so switching mid-run makes iterations incomparable — and the final verdict must use the same regime the loop ranked under.
- **User-provided eval** (custom `{{BENCH_COMMAND}}`): the trial counts, correctness rounds, and reference handling are the **user's contract** — do **not** inject `--no-ref` or cut counts on a script you didn't author (it may have no such flag, break the interface, or silently invalidate the measurement / a leaderboard's required N). Use only the fast-iteration switches the user exposed (flags / env vars documented in the prompt or `HINTS.md`). If iteration is too slow and none exist, **raise it with the user** — don't fabricate one.

Caching the reference's *runtime* across iterations is sound only on a clock-locked GPU; on unlocked clocks (ref and solution timed in different clock states) prefer ranking by the solution's own latency.

## Stall handling

When 3 consecutive iterations show no improvement (≥3% over current best), pause the loop and re-assess before iter N+1. Re-assessment combines:

- **Re-profile** with `ncu` if available, or re-read runtime stats from `ITERATIONS.md` (median vs min/mean, distribution shape) if not.
- Use the host's available search tools for op-specific techniques on the same hardware class when network access is allowed; otherwise use local references.
- **Review `ITERATIONS.md`** for patterns (which axes have been tried, which haven't, where prior wins came from).

Default outcome: pick a new direction and continue. Only escalate to stop (see next section) if re-assessment produces concrete evidence the current state is at a physical floor.

## When to stop

Legitimate triggers:

1. The user stops the task, or the agreed iteration, time, or cost budget is reached (in the prompt or `HINTS.md`). Never extend it automatically.
2. Stall re-assessment produced hard evidence of a floor — e.g., min runtime at cuda-event timer resolution, kernel arithmetic at HBM bandwidth limit, launch overhead dominating compute. Cite the evidence in `ITERATIONS.md`.
3. All viable directions exhausted: document at least 3 distinct directions tried (with their iteration numbers) in `ITERATIONS.md` before invoking this trigger, to prevent premature stops.

A missing optional profiler permits analytical reassessment. A missing required evaluator, GPU authorization, or correctness oracle prevents promotion: preserve the current evidence and state the blocker instead of claiming success.

### HEAD handling on stop

When finishing within the authorized task, restore the best correct implementation, not necessarily the latest. Do not reset Git history or overwrite user edits. If the user explicitly stops work, stop immediately and report the current state; do not run a cleanup benchmark. Procedure:

1. Identify the best iter by reading `ITERATIONS.md` Summary, the bench output for each iter under `trajectory/`, and your own reasoning notes. Useful signals from KernelBench output: median speedup, runtime std (consistency), min runtime (tail), `CORRECT` flag. Other bench harnesses expose different shapes — use what's available. Justify your pick in the commit message (e.g., "iter 4: best mean AND lowest min, while iter 6 ties on mean but has higher std").

2. If best iter ≠ latest iter:
   - `git checkout <best-iter-sha> -- solution/` — verbatim copy, do NOT hand-reconstruct from memory or earlier notes.
   - Run a full final verification with a new label, preserving every earlier run. Budget for this verification before the optimization budget is exhausted.
   - `git commit -m "[final] Restore iter-K (X.XXx) — <one-sentence why>"`.

Restore verbatim from the recorded commit or immutable snapshot; manual reconstruction risks drift. Even when the best iteration is already current, run the full final verification or explicitly report it as pending. A failed final gate leaves the candidate unaccepted.

## ncu profiling — best effort, not a gate

Probe `ncu` once after baseline. If it fails (driver mismatch, missing toolkit, user opt-out via free-text `HINTS.md` directive), proceed analytically for the rest of the loop without re-probing within this session. Don't gate iteration progress on ncu availability; analytical reasoning + runtime stats from the bench harness are a valid substitute for the optimizer's direction picking.

## Gotchas

- **Pursue genuine latency reduction, not reward hacking.** No CUDA stream injection to evade timing, no monkey-patching the benchmark, no returning uninitialized results. The built-in evaluator flags >10× speedups for a reason — investigate before celebrating. It also re-randomizes the timed inputs in place after timing (mutation sentinel): a solution that keys on input identity and replays a stored output fails outright.
- **The solution file must not contain `get_inputs` / `get_init_inputs`.** The bench script strips the solution's module-level tail before eval as an anti-cheat boundary. Inputs come from the reference or `--inputs` file, never the solution.
- **`get_inputs()` must produce fresh data each call.** Bench calls it many times — every correctness trial, and (under the default fresh regime) before every timed trial. Module-level cached tensors make correctness checks trivially pass and let timing measure cache-warm performance. Use `torch.randn` or reload from disk on every call.
- **Don't be lazy.** Stay-in-PyTorch, only-tune-configurations, skip-profiling — these are the default low-effort failure modes for agents. The point of the loop is to *rewrite* the implementation — switch languages (Triton → CUDA, etc.) when it helps.

## Reference files

- `bench/kernelbench/GUIDE.md` — full input assembly patterns, CLI flags, timing methods, tolerances. Read this before writing the bench command if anything about input shape, precision, or backend is non-obvious.
- `HINTS.md` (workspace) — user-editable behavior directives. Read at session start; respect any constraint named there.
- `ITERATIONS.md` (workspace) — your own iteration log. Write to it every iteration.
