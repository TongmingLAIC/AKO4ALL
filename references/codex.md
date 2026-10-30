# Using AKO4ALL in Codex

AKO4ALL is executed by the current Codex agent. It does not launch Claude Code,
choose a model, change permissions, or require a second agent process.

## Install and invoke

From an AKO4ALL checkout, run:

```bash
python3 tools/install_codex_skill.py
```

This copies the skill and its runtime resources into `~/.agents/skills/ako4all`.
The installer refuses an existing destination and does not modify configuration.
For a project installation, set `--dest /path/to/project/.agents/skills/ako4all`.
Keep the source checkout outside the destination. `--dry-run` prints the path
without writing. An installed copy is a snapshot: move an old installation
outside the skill discovery directories before installing an update.

Open the kernel workspace in Codex and use:

```text
Use $ako4all to optimize source/kernel.py against source/reference.py.
Use the existing benchmark and its correctness thresholds. Run at most
10 optimization iterations, then a full final verification. GPU use for
these tests is authorized. Preserve the best correct candidate and all results.
```

The skill is discoverable on the next turn; restart Codex if it is not listed.
The root `SKILL.md` is shared with Claude Code. Refer to its resolved installation
directory when copying scaffolds; do not assume the skill lives in the task CWD.

## Run from a script

When a noninteractive invocation is useful, Codex CLI can drive the same skill:

```bash
cd /path/to/kernel-workspace
codex exec --sandbox workspace-write --json \
  'Use $ako4all to optimize source/kernel.py with our existing benchmark. Run at most 10 iterations plus final verification; use the GPU only as authorized in HINTS.md.' \
  > codex-events.jsonl
```

Use the permissions required by the actual environment. A sandbox that cannot
access the GPU or a remote benchmark is an environment limitation, not a reason
to bypass the user's controls. Do not invoke this nested CLI from an already
running Codex task just to use the skill.

For scripts, retain the thread ID from the JSONL `thread.started` event and resume
that specific run:

```bash
codex exec resume <SESSION_ID> 'Continue the AKO4ALL run within its remaining budget.'
```

Codex's JSONL event schema differs from Claude Code's `stream-json`; no Claude
transcript parser or externally generated Claude session ID should be reused.

## Recover the experiment

After interruption, reconcile `HINTS.md`, `ITERATIONS.md`, trajectory snapshots,
and Git status/history before editing. An unlogged measurement still counts
against the original budget. Keep previous labels intact and give retries new
labels. Record pending tests as pending, never as passing.

Keep experiment logs outside the source repository if its contribution rules
exclude them. Stage only task-owned changes. Follow repository branch naming,
required tests, numerical gates, and publication rules over the scaffold defaults.

Use normal Codex editing, terminal, and search tools. The ordering invariant is
measure → preserve evidence → record/commit → next hypothesis; it does not depend
on a fixed number of tool calls. Parallel worker agents are optional and require
task authorization. Measurements on the same GPU remain serialized.

## Verification scope

The installer and packaging have CPU-only regression tests:

```bash
python3 -m unittest discover -s tests -v
```

These tests do not validate generated GPU code, model quality, or speedups. The
historical performance results in the README are upstream Claude/CUDA runs, not
Codex measurements. The built-in evaluator still requires CUDA-enabled PyTorch;
changing the coding agent does not change the evaluator's hardware requirements.

Official references:

- [Codex skill discovery](https://learn.chatgpt.com/docs/build-skills)
- [Codex noninteractive execution and resume](https://developers.openai.com/codex/noninteractive)
