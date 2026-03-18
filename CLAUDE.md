# kernel-opt-agent — Developer Guide

This is the **template repository** for spawning isolated GPU kernel optimization environments.
It is NOT an optimization environment itself — use `setup.sh` to create one.

## Repository Purpose

Provides scaffolding to create child environments where a Code Agent (e.g. Claude Code)
autonomously optimizes a Triton kernel, using the flashinfer-bench SDK for evaluation.

Supports any flashinfer-bench compatible dataset, including user-defined operators.

## Directory Structure

```
kernel-opt-agent/
├── CLAUDE.md                          # This file (developer guide for the template repo)
├── README.md                          # Human-facing usage guide
├── setup.sh                           # Creates isolated child environments
├── .gitignore                         # Git ignore rules (copied to children)
├── templates/
│   ├── prompt.md                      # Default system prompt (embedded into CLAUDE.md at spawn)
│   ├── info.md                        # Default Info.md template (copied to children)
│   ├── claude-md/
│   │   ├── common.md                  # CLAUDE.md skeleton with placeholders
│   │   ├── objective-scratch.md       # Objective for scratch mode
│   │   ├── objective-existing.md      # Objective for existing mode
│   │   ├── environment-local.md       # A100 environment description
│   │   └── environment-modal.md       # B200 environment description
│   └── settings/
│       ├── local.json                 # .claude/settings.local.json for local backend
│       └── modal.json                 # .claude/settings.local.json for modal backend
└── scripts/
    ├── generate_context.py            # Extracts operator metadata for template rendering
    ├── bench.sh                       # Local benchmark script
    ├── bench_modal.sh                 # Modal benchmark script
    ├── run_local.py                   # Local A100 benchmark runner
    ├── run_modal.py                   # Modal B200 benchmark runner
    └── pack_solution.py               # Pack kernel for evaluation
```

## How setup.sh Works

1. Parses `--operator`, `--mode`, `--backend`, `--kernel`, `--name`, `--dataset`, `--prompt`, `--info`
2. If `--operator` not provided, lists available operators from the dataset and exits
3. Auto-discovers operator definition from `<dataset>/definitions/*/<operator>.json`
4. Runs `generate_context.py` to extract template variables (shapes, workload summary, etc.)
5. Auto-increments run number -> creates `kernel-opt-agent-run-NNN[-label]/`
6. Copies definition.json and workloads.jsonl from the dataset
7. Auto-generates `config.toml` with operator name and default build settings
8. Copies backend-specific bench script and runner (`run_local.py` or `run_modal.py`)
9. Copies settings: `templates/settings/{backend}.json` -> `.claude/settings.local.json`
10. Extracts reference kernel from definition.json (scratch) or copies user-provided kernel (existing)
11. Copies `Info.md` (from `--info` or `templates/info.md`)
12. Assembles CLAUDE.md from templates: embeds prompt.md content + replaces all placeholders
13. Initializes a git repo in the child directory

## Template Placeholders

Templates use these placeholders (replaced at spawn time):

| Placeholder | Source | Example |
|-------------|--------|---------|
| `{{PROMPT}}` | prompt.md content | (multi-line system instructions) |
| `{{OPERATOR}}` | definition name | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` |
| `{{OPERATOR_DESCRIPTION}}` | first sentence of definition description | `Batched Native Sparse Attention (DSA)...` |
| `{{NUM_WORKLOADS}}` | count of workloads | `23` |
| `{{SHAPE_SUMMARY}}` | auto-formatted input/output shapes | `Key shapes: ...` |
| `{{WORKLOAD_SUMMARY}}` | distribution of variable axes | `Workloads: num_tokens in {1(x1),...}` |
| `{{OBJECTIVE}}` | objective-{mode}.md content | (multi-line) |
| `{{ENVIRONMENT}}` | environment-{backend}.md content | (multi-line) |
| `${GPU_SHORT}` | backend -> GPU name | `A100 GPU` or `B200 GPU` |

## How to Modify Templates

- **System prompt**: Edit `templates/prompt.md` — this is embedded at the top of the child's CLAUDE.md
- **Agent task instructions**: Edit files under `templates/claude-md/`
  - `common.md` — shared structure (prompt embed point, benchmark, scoring, spec, etc.)
  - `objective-scratch.md` / `objective-existing.md` — mode-specific objectives
  - `environment-local.md` / `environment-modal.md` — backend-specific environment descriptions
- **Agent permissions**: Edit `templates/settings/local.json` or `templates/settings/modal.json`
- **Info template**: Edit `templates/info.md`
- **Config defaults**: Edit the `config.toml` generation block in `setup.sh`
- **Benchmark scripts**: Edit `scripts/bench.sh`, `scripts/bench_modal.sh`, etc.

## Key Constraints

- Agents only edit `solution/triton/kernel.py` (and optionally `config.toml`)
- `config.toml`: `destination_passing_style = false` by default
- Operator data (definition.json, workloads.jsonl, reference kernel) comes from the dataset at spawn time, not from static files in this repo
