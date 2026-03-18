# kernel-opt-agent — Developer Guide

This is the **template repository** for spawning isolated GPU kernel optimization environments.
It is NOT an optimization environment itself — use `setup.py` to create one.

## Repository Purpose

Provides scaffolding to create child environments where a Code Agent (e.g. Claude Code)
autonomously optimizes a GPU kernel (Python, Triton, CUDA, C++, or TileLang), using the flashinfer-bench SDK for evaluation.

Supports any flashinfer-bench compatible dataset, including user-defined operators.

## Directory Structure

```
kernel-opt-agent/
├── CLAUDE.md                          # This file (developer guide for the template repo)
├── README.md                          # Human-facing usage guide
├── setup.py                           # Creates isolated child environments
├── .gitignore                         # Git ignore rules (copied to children)
├── templates/
│   ├── task.md                        # Main task template (assembled into CLAUDE.md at spawn)
│   ├── hints.md                       # Default HINTS.md template (copied to children)
│   ├── fragments/
│   │   ├── objective-scratch.md       # Objective for scratch mode
│   │   ├── objective-existing.md      # Objective for existing mode
│   │   ├── gpu-a100.md                # A100 hardware specs
│   │   ├── gpu-b200.md                # B200 hardware specs
│   │   ├── backend-local.md           # Local execution instructions
│   │   └── backend-modal.md           # Modal execution instructions
│   ├── stubs/
│   │   └── cuda/                      # Stub templates for CUDA scratch mode
│   │       ├── kernel.cu
│   │       └── binding.py
│   └── agent/
│       └── claude.json                # Claude Code agent config (output filenames + permissions)
└── scripts/
    ├── generate_context.py            # Extracts operator metadata for template rendering
    ├── bench_utils.py                 # Shared benchmark utilities (baseline caching, scoring)
    ├── bench.sh                       # Local benchmark script
    ├── bench_modal.sh                 # Modal benchmark script
    ├── run_local.py                   # Local A100 benchmark runner
    ├── run_modal.py                   # Modal B200 benchmark runner
    └── pack_solution.py               # Pack kernel for evaluation
```

## Language Categories

The system distinguishes two language categories that behave differently:

### Python-like Languages (python, triton, tilelang)
- **Entry point**: `kernel.py::run`
- **Scratch mode**: Reference from definition.json is extracted directly to `solution/{language}/kernel.py`
- **Existing mode**: User kernel is renamed to `kernel.py` (matches entry point)
- **Validation**: In existing mode, checks for `def run(` in the kernel file

### Compiled Languages (cuda, cpp)
- **Entry point**: `binding.py::kernel` (Python bindings using TVM FFI)
- **Scratch mode**:
  - Stub templates copied to `solution/{language}/` (kernel.cu + binding.py for CUDA)
- **Existing mode**: User kernel keeps original filename (may have multiple files)
- **Validation**: No `def run(` check (bindings are in Python but kernel logic is compiled)

This distinction is defined in `setup.py` as:
```python
PYTHON_LIKE = {"python", "triton", "tilelang"}
```

## How setup.py Works

1. Parses `--operator`, `--mode`, `--backend`, `--language`, `--gpu`, `--agent`, `--kernel`, `--name`, `--dataset`, `--task`, `--hints`
2. If `--operator` not provided, lists available operators from the dataset and exits
3. Derives `LANGUAGE_NAME` from `--language` (python→Python, triton→Triton, cuda→CUDA, cpp→C++, tilelang→TileLang) and resolves GPU (auto-detect for local, required for modal)
4. Reads agent config from `templates/agent/{agent}.json`:
   - `task_filename`: Name of the generated task file (e.g., "CLAUDE.md")
   - `hints_filename`: Name of the hints file (e.g., "HINTS.md")
   - `permissions`: Claude Code permissions (Bash, Edit, Read, Write, etc.)
5. Auto-discovers operator definition from `<dataset>/definitions/*/<operator>.json`
6. Calls `generate_context()` to extract template variables (shapes, workload summary, etc.)
7. Auto-increments run number -> creates `kernel-opt-agent-run-NNN[-label]/`
8. Copies definition.json and workloads.jsonl from the dataset
9. Auto-generates `config.toml` with operator name, language, GPU, and default build settings
10. Copies backend-specific bench script and runner (`run_local.py` or `run_modal.py`)
11. Generates `.claude/settings.local.json` from agent config permissions
12. **Extracts or prepares kernel code**:
    - **Scratch mode, Python-like languages**: Extracts reference from definition.json → `solution/{language}/kernel.py`
    - **Scratch mode, compiled languages**:
      - Copies stub templates from `templates/stubs/{language}/` → `solution/{language}/`
    - **Existing mode, Python-like languages**: Copies user kernel, renames to `kernel.py`
    - **Existing mode, compiled languages**: Copies user kernel, preserves original filename
13. Copies hints file (from `--hints` or `templates/hints.md`) as `HINTS.md`
14. Assembles task file from `task.md` + fragment placeholders -> output filename from agent config
15. Initializes a git repo in the child directory

## Template Placeholders

Templates use these placeholders (replaced at spawn time):

| Placeholder | Source | Example |
|-------------|--------|---------|
| `{{LANGUAGE}}` | `--language` flag (default: triton) | `triton`, `cuda`, `cpp` |
| `{{LANGUAGE_NAME}}` | derived from `--language` | `Triton`, `CUDA`, `C++` |
| `{{GPU}}` | `fragments/gpu-{gpu}.md` content (or fallback `- GPU: {NAME}`) | (hardware specs line) |
| `{{GPU_NAME}}` | `uppercase(--gpu)` | `A100`, `H100` |
| `{{BACKEND}}` | `fragments/backend-{backend}.md` content | (execution instructions) |
| `{{OBJECTIVE}}` | `fragments/objective-{mode}.md` content | (multi-line) |
| `{{OPERATOR}}` | definition name | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` |
| `{{OPERATOR_DESCRIPTION}}` | first sentence of definition description | `Batched Native Sparse Attention (DSA)...` |
| `{{NUM_WORKLOADS}}` | count of workloads | `23` |
| `{{SHAPE_SUMMARY}}` | auto-formatted input/output shapes | `Key shapes: ...` |
| `{{WORKLOAD_SUMMARY}}` | distribution of variable axes | `Workloads: num_tokens in {1(x1),...}` |

## How to Modify Templates

- **Task template**: Edit `templates/task.md` — the main template assembled into the child's task file
- **Objective fragments**: Edit `templates/fragments/objective-scratch.md` or `objective-existing.md`
- **GPU hardware specs**: Edit or add `templates/fragments/gpu-{name}.md` (optional; if missing, `{{GPU}}` falls back to `- GPU: {NAME}`)
- **Backend execution instructions**: Edit `templates/fragments/backend-local.md` or `backend-modal.md`
- **Agent config**: Edit `templates/agent/claude.json` (output filenames + permissions)
- **Hints template**: Edit `templates/hints.md`
- **Config defaults**: Edit the `config.toml` generation in `setup.py` (`populate_child` function)
- **Benchmark scripts**: Edit `scripts/bench.sh`, `scripts/bench_modal.sh`, etc.

## Baseline Caching

The benchmark system caches reference implementation performance on the first run:
- First `bench.sh` run profiles the reference (20 iterations) and saves metrics to `baseline.json`
- Subsequent runs skip reference profiling and use cached values with 100 iterations for the solution
- This provides stable reference metrics and more accurate solution timing
- Use `--force-baseline` to re-profile: `bash scripts/bench.sh --force-baseline`
- `baseline.json` auto-invalidates when workloads change

## Key Constraints

- Agents only edit files in `solution/{language}/` (and optionally `config.toml`)
- `config.toml`: `destination_passing_style = false` by default
- Operator data (definition.json, workloads.jsonl, reference kernel) comes from the dataset at spawn time, not from static files in this repo
- Benchmark scripts hardcode conda environment name `fi-bench` (the `conda run -n fi-bench` line in `scripts/bench.sh` and `scripts/bench_modal.sh`)
- Local benchmarking uses CUPTI profiling via `cupti-python` (requires CUDA 13.0+ driver; falls back to CUDA events on older drivers)

## Entry Point Conventions

Entry points vary by language (configured in config.toml `entry_point` field):

- **python, triton, tilelang**: `"kernel.py::run"` — single Python file with `run()` function
- **cuda**: `"binding.py::kernel"` — Python bindings file with `kernel()` function calling CUDA kernel
- **cpp**: `"binding.py::kernel"` — Python bindings file with `kernel()` function calling C++ code

For compiled languages, the `binding.py` file uses TVM FFI (`from tvm.ffi import register_func`) to expose compiled code to Python.

## ⚠️ Custom Dataset Requirements

**CRITICAL**: If you create a custom dataset, the `definition.json` **must** contain a `reference` field with **plain PyTorch code**:

```json
{
  "name": "my_operator",
  "reference": "import torch\n\n@torch.no_grad()\ndef run(...):\n    # Plain PyTorch implementation\n    return ...",
  ...
}
```

The `reference` is the mathematical specification — flashinfer-bench executes it as Python (`language=PYTHON`, `entry_point=main.py::run`) for:
- Generating reference outputs (correctness baseline)
- Profiling reference latency (performance baseline)

**Do NOT use CUDA/C++/Triton code in `reference`** — the framework will fail. Those languages belong in the `Solution` being optimized, not the `Definition.reference`.
