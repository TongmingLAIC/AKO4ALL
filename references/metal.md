# Metal kernels and on-device inference

Use this route for Apple Silicon / Metal tasks. Keep the user's native runtime
(for example C++20 + Objective-C++ + MSL in Kapitsa); do not add PyTorch, MLX,
CUDA, or a new serving engine to make the default evaluator work. The CUDA
KernelBench evaluator and its `ncu` path do not apply to Metal.

## Establish the experiment

Read the target repository's `AGENTS.md`, contributor guide, and existing harness.
Identify the actual dispatch that exercises the changed kernel. Record chip/GPU
family, macOS build, SDK/MSL versions, both commits/binaries, model and drafter
revisions, shapes/strides/dtypes, input domains, cache state, prompts, and seeds.
Set a bounded iteration budget and reserve a final verification run.

Choose the requested scope before writing code:

- **Kernel**: an independent CPU/high-precision or trusted implementation checks
  boundary shapes, odd sizes, strides, tails, masks, and special values. Use a
  predeclared numerical contract, including rounding and reduction behavior.
- **Inference**: retain the kernel checks and add the repository's model-output
  gate. Measure prefill, decode, TTFT and follow-up latency separately through the
  real runtime path. A faster isolated kernel is not evidence of faster inference.

Freeze reference inputs, numerical floors, evaluator commands, and acceptance
thresholds. The optimizer changes the candidate implementation, not the gate.
For an in-tree kernel, edit the permitted repository paths in an isolated
worktree and rebuild named targets; copying a `.metal` file to `solution/` is not
enough if the binary embeds kernels from the source tree. List the actual built
binaries and relevant sources in the evaluator's `artifacts` fields.

## Small-size qualification before timing

Model loading or any GPU work requires task authorization. Use the shared
`BENCH_GPU_LOCK` (default `/tmp/kapitsa-gpu.lock`) for the whole comparison, so
other cooperating measurements cannot interleave. A lock does not establish
that the GPU is idle: inspect contention and thermal/power conditions separately.

Exercise new memory/synchronization designs at small sizes first with
`MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`. Do not time under instrumentation.
Then run the full correctness matrix before measuring performance. A CPU build
does not compile or execute runtime-embedded MSL and cannot qualify it.

In Kapitsa, follow the current repository rule for the validation layer's static
threadgroup-memory double counting. On its documented macOS setup, allocations
above 16,384 B cannot run under the layer even though the hardware cap is 32,768 B.
Record the kernel, actual allocation, device limit, and this specific exception;
do not silently disable validation or call it a successful validation run.

## Run and interpret the evaluator

`bench/metal/evaluate.py` is a Python 3.9+ stdlib command adapter. It does not
generate inputs or launch a model on its own. Configuration selects existing
correctness and timing commands; the CLI requires an Apple Silicon Mac and
`--allow-gpu` for execution. Validate configuration without executing commands:

```bash
python3 /path/to/ako4all/bench/metal/evaluate.py experiment.json \
  --out /path/to/evidence/iter-1 --dry-run
```

When the task authorizes these GPU measurements:

```bash
python3 /path/to/ako4all/bench/metal/evaluate.py experiment.json \
  --out /path/to/evidence/iter-1 --allow-gpu
```

Paths in a config resolve relative to that config. Commands are argv arrays,
executed without a shell; use an explicit reviewed script when shell features
are needed. `{run_dir}` and `{label}` expand inside arguments. Each command has
a timeout; failure preserves stdout/stderr and terminates its process group.
Use a new output directory per attempt: earlier evidence is never overwritten.

The runner acquires a nonblocking `flock` and passes `NI_LOCK_HELD=1` /
`NACE_GPU_LOCK_HELD=1` only while holding it. Commands must run synchronously and
honor that inherited-lock convention. A nested command that tries to take a
second lock will block; do not wrap the runner in another lock holder.

The full JSON format is illustrated in `bench/metal/kapitsa.example.json`.
For a single kernel, use `scope: "kernel"`, select a latency metric with
`direction: "lower"`, and replace the commands with the repository's microbench.
Each correctness command must exit zero and print exactly one line:

```text
AKO_CHECK={"passed":true,"cases":37}
```

Set `shader_validation: true` on a small-size correctness command to enable the
two Metal validation variables for that command only. Other correctness commands
can implement full numerical/model gates. A nonzero exit, missing record, false
pass, or zero cases stops the run before timing.

Each timing command must exit zero and print exactly one line:

```text
AKO_METRICS={"valid":true,"metrics":{"kernel_ms":0.42},"context":{"shape":[128,4096],"dtype":"float16"}}
```

These numbers are format examples, not results. `valid` must be false for
contention, thermal invalidity, or incomplete work. Required metrics must be
finite and positive. Optional `context` must stay identical across A/B runs.
Synchronize completed GPU work and exclude compile/warmup time as specified by
the contract. For GPU-only time, use the target runtime's completed command-buffer
timestamps; CPU enqueue duration does not measure kernel latency. End-to-end
latency has a different boundary and must remain a separate metric.

The runner performs at least A1 B1 A2 B2, preserves every value, and hashes the
listed artifacts before/after. It reports:

- `accepted`: all configured correctness checks pass, no metric exceeds its
  allowed regression, and the primary gain clears both its configured minimum
  and the observed same-build spread.
- `inconclusive`: the measured gain is too small or noisy; retain the baseline.
- `rejected`: a configured performance regression; retain the baseline.
- `error` / `incomplete`: a failed check, invalid measurement, changed artifact,
  timeout, interruption, or unavailable resource; no acceptance claim.

The spread check is a conservative screening rule, not a statistical significance
test. Reports audit commands and hashes, but cannot prove a custom adapter is
honest or measures the intended dispatch. Inspect that adapter and independently
recheck the selected final artifact. An `accepted` kernel report says nothing
about inference checks that were not configured.

## Kapitsa integration

Read its current `AGENTS.md` and `CONTRIBUTING.md` rather than copying policies
into the skill. `.dsh/skills/verify-kapitsa/scripts/vk checks` identifies affected
models and gates; build the required named CPU targets and run the repository
CPU/lint checks. `vk ref` can prepare a separate baseline build. Follow the
current target list even if an older helper omits tests.

Copy `bench/metal/kapitsa.example.json` to the experiment directory and replace
the paths. The example's 3% primary threshold is illustrative, not Kapitsa policy;
agree the metric/threshold before running and never lower it to admit a candidate.
Add the changed-kernel small-size validation check before the model check. Record
a repository-approved validation exception in the experiment when applicable.

The supplied `kapitsa.py` adapter:

1. Requires the runner's inherited lock and an existing local model directory.
2. Runs old `logit_dump`, new `logit_dump --from` on the same prefixes, and the
   checkout's `tools/logit_compare.py`. Only a successful exit emits a pass.
3. Executes one `kapitsa bench --quick --json --out` per A/B leg, keeps the native
   JSON, rejects flagged/invalid runs, and reports decode, prefill and TTFT metrics.
4. Preserves dumps and reports for inspection; it does not publish or submit them.

Run a separate experiment for each required model. Inspect the native logs for
non-empty output, valid token counts before EOS, and the actual drafter path and
acceptance statistics. Server behavior checks and any additional long-context
coverage named by `vk checks` remain required; the quick adapter does not replace
them. Do not remeasure noise floors, regenerate frozen prompts/results, or change
defaults based only on an isolated-kernel gain. A true K-reduction reassociation
uses `--reorder` only with the repository's required evidence; this adapter uses
the normal gate and never selects that exception automatically.

## Continue the optimization loop

For each candidate, state one hypothesis, edit the native implementation, rebuild,
then run the checks above. Append the reason and report path to `ITERATIONS.md`,
including rejected and failed attempts. Keep the current best accepted candidate
separate from the latest experiment. After a stall, change the hypothesis rather
than the evaluator. At the budget limit, restore the exact best artifact and run
full qualification; if no candidate qualifies, preserve the baseline and report
that outcome. Follow repository rules for disk-state format bumps, experiment
locations, and one-change PRs. No autonomous merge is part of this workflow.

This integration has CPU fixture coverage only. Real MSL execution, GPU validation,
model-output parity and performance require an explicitly authorized Mac run.
