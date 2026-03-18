# kernel-opt-agent

A tool for automated GPU kernel optimization using Code Agents (e.g. Claude Code). Built on the [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) SDK, it supports any flashinfer-bench compatible dataset, including user-defined operators.

## Quick Start

```bash
# List available operators
python setup.py

# Create an optimization environment (local, auto-detect GPU, Triton, from scratch)
python setup.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64

# Enter the child environment and start your code agent
cd ../kernel-opt-agent-run-001
claude
```

## Setup Options

```bash
python setup.py \
  --operator <name> \
  [--dataset /path/to/dataset] \    # default: $FIB_DATASET_PATH
  [--mode scratch|existing] \       # default: scratch
  [--backend local|modal] \         # default: local
  [--language python|triton|cuda|cpp|tilelang] \  # default: triton
  [--gpu <name>] \                  # local: auto-detected; modal: required
  [--agent claude] \                # default: claude
  [--kernel /path/to/kernel.py] \   # required for existing mode
  [--name <label>] \                # optional label for the run
  [--task /path/to/task.md] \       # custom task template
  [--hints /path/to/hints.md]       # custom hints for the agent
```

### Examples

```bash
# Local, auto-detect GPU, from scratch, with label
python setup.py --operator gdn_decode_qk4_v8_d128_k_last --name "gdn_exp_1"

# Local with explicit GPU
python setup.py --operator gdn_decode_qk4_v8_d128_k_last --gpu h100

# Local, from existing kernel
python setup.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
    --mode existing --kernel /path/to/kernel.py

# Modal B200, from scratch (--gpu required for modal)
python setup.py --operator moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
    --backend modal --gpu b200 --name "moe_b200"

# Custom dataset
python setup.py --operator my_custom_op --dataset /path/to/my/traceset

# Custom task template and hints
python setup.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
    --task ./my_task.md --hints ./my_hints.md

# CUDA kernel from scratch (creates stub templates)
python setup.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 --language cuda

# TileLang kernel (extracts reference, like triton)
python setup.py --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 --language tilelang

# Legacy flag aliases still work (--prompt → --task, --info → --hints)
python setup.py --operator my_op --prompt ./my_task.md --info ./my_hints.md
```

### Existing Mode Behavior

When using `--mode existing --kernel /path/to/your_kernel.py`:

- **Python-like languages** (python/triton/tilelang): Your kernel is renamed to `kernel.py` to match the entry point
- **Compiled languages** (cuda/cpp): Your kernel keeps its original filename (e.g., `best_cuda_kernel.cu`)

If a `config.toml` exists in the same directory as your kernel, it will be copied to the child environment (useful for pre-configuring `destination_passing_style`).

## How It Works

1. `python setup.py --operator <name>` discovers the operator definition and workloads from the dataset
2. Creates a child environment `kernel-opt-agent-run-NNN[-label]/` with everything the agent needs
3. You start your code agent (e.g. `claude`) in the child environment
4. The agent reads `CLAUDE.md` and `HINTS.md`, then iteratively optimizes the kernel

### Child Environment Structure

```
kernel-opt-agent-run-NNN/
├── CLAUDE.md                   # Auto-generated task spec (operator details + instructions)
├── HINTS.md                    # User-editable hints and constraints
├── config.toml                 # Operator configuration
├── .gitignore
├── docs/
│   ├── definition.json         # Operator definition
│   ├── workloads.jsonl         # Benchmark workloads
├── solution/
│   ├── triton/kernel.py        # Triton/Python/TileLang (entry: kernel.py::run)
│   └── cuda/                   # CUDA (entry: binding.py::kernel)
│       ├── kernel.cu           #   - CUDA kernel implementation
│       └── binding.py          #   - Python bindings (TVM FFI)
├── scripts/
│   ├── bench_utils.py          # Shared benchmark utilities
│   ├── bench.sh                # Benchmark script
│   ├── run_local.py/run_modal.py  # Benchmark runner
│   └── pack_solution.py        # Pack kernel for evaluation
├── baseline.json               # Auto-created, cached reference performance
├── trajectory/                 # Auto-created, stores each bench run
└── .claude/
    └── settings.local.json     # Permission fallback
```

### Entry Points by Language

The entry point depends on the kernel language:

- **Python, Triton, TileLang**: `solution/{language}/kernel.py::run`
  - Agent edits a single Python file
  - Entry function is `run(...)`

- **CUDA**: `solution/cuda/binding.py::kernel`
  - Agent edits `kernel.cu` (CUDA kernel code) and `binding.py` (Python bindings using TVM FFI)
  - Entry function is `kernel(...)` in binding.py

- **C++**: `solution/cpp/binding.py::kernel`
  - Agent edits C++ source files and `binding.py` (Python bindings using TVM FFI)
  - Entry function is `kernel(...)` in binding.py

**Note for compiled languages (CUDA, C++):** The PyTorch reference implementation is in the `reference` field of `docs/definition.json`. Study it to understand the computation semantics before implementing the compiled kernel.

## Customization

### Custom Task Template

The default task template (`templates/task.md`) defines the agent's instructions. Override it with `--task`:

```bash
python setup.py --operator my_op --task ./my_task.md
```

### Custom Hints

`HINTS.md` is placed in each child environment for ad-hoc hints. Override the template with `--hints`:

```bash
python setup.py --operator my_op --hints ./my_analysis.md
```

Or edit `HINTS.md` directly in the child environment before launching.

### Custom Dataset

Any flashinfer-bench compatible trace set works. The expected structure:

```
dataset/
├── definitions/
│   └── <type>/
│       └── <operator>.json
└── workloads/
    └── <type>/
        └── <operator>.jsonl
```

**Important:** Your `<operator>.json` must contain a `reference` field with **plain PyTorch code**:

```json
{
  "name": "my_operator",
  "reference": "import torch\n\n@torch.no_grad()\ndef run(input, weight):\n    return torch.matmul(input, weight)",
  ...
}
```

The `reference` serves as the mathematical specification and correctness baseline. FlashInfer-Bench will:
1. Execute it as Python to generate reference outputs for correctness checks
2. Profile it to measure baseline performance (when `profile_baseline=True`)

⚠️ **Critical Constraint**: The `reference` must be executable Python/PyTorch code. FlashInfer-Bench hardcodes reference execution as `language=PYTHON, entry_point=main.py::run`. Using CUDA/C++/Triton code in the `reference` field will cause compilation errors and benchmark failures.

## Benchmark Output

The benchmark script prints per-workload results and a final score:

```
operator_name:
  Workload 0c23b10...: PASSED | 0.492 ms | 4.17x speedup | abs_err=1.2e-03, rel_err=2.1e-04
  ...

══════════════════════════════════════════════════
  FINAL SCORE (mean speedup): 4.17x
  Passed: 23/23 | Min: 2.10x | Max: 8.34x
  By num_tokens:  1→4.17x  2→3.85x  ...
══════════════════════════════════════════════════
```

The `score` field (= `FINAL SCORE` = mean speedup across all workloads) is the primary metric. All workloads must PASS for a valid score.

### Baseline Caching

The first benchmark run profiles the reference implementation and caches the result to `baseline.json`. Subsequent runs skip reference profiling for faster, more stable comparisons and use higher iteration counts (100 vs 20) for more accurate solution timing.

```bash
bash scripts/bench.sh "baseline"          # First run: profiles reference, caches baseline
bash scripts/bench.sh "opt_v1"            # Uses cached baseline, 100 iterations for solution
bash scripts/bench.sh --force-baseline    # Force re-profile reference
```

## Installation

### Prerequisites

- **NVIDIA GPU**: CUDA-capable GPU (A100, H100, B200, etc.)
- **CUDA Driver**: 13.0+ recommended for CUPTI profiling via `cupti-python` (falls back to CUDA events on older drivers)
- **Python**: 3.10 – 3.13 (3.12 recommended)
- **Conda**: For environment management

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/flashinfer-ai/kernel-opt-agent.git
cd kernel-opt-agent

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate fi-bench

# Install flashinfer-bench from source
pip install git+https://github.com/flashinfer-ai/flashinfer-bench.git@main

# Set dataset path (required for local backend)
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/flashinfer-ai/kernel-opt-agent.git
cd kernel-opt-agent

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install flashinfer-bench from source
pip install git+https://github.com/flashinfer-ai/flashinfer-bench.git@main

# Set dataset path (required for local backend)
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

### Modal Backend Setup (Optional)

For cloud benchmarking on Modal:

```bash
# Authenticate with Modal
conda run -n fi-bench --no-capture-output modal setup

# Create and upload trace volume
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace/
```

### Verify Installation

```bash
# Check GPU
nvidia-smi

# Check flashinfer-bench
python -c "import flashinfer_bench; print(flashinfer_bench.__version__)"

# Check CUPTI profiling support (requires CUDA 13.0+ driver; falls back to CUDA events)
python -c "
try:
    from cupti import cupti; cupti.get_timestamp(); print('CUPTI: available')
except Exception as e:
    print(f'CUPTI: unavailable ({e}), will use CUDA events fallback')
"

# List available operators
python setup.py --dataset /path/to/flashinfer-trace
```

**⚠️ Note**: Local benchmarking uses CUPTI profiling by default, which requires a **CUDA 13.0+ driver** and `cupti-python`. On older drivers (CUDA 12.x), profiling automatically falls back to CUDA events. For best accuracy, use a CUDA 13.0+ driver or the Modal backend (`--backend modal`).

## Requirements Summary

See **Installation** section above for detailed setup instructions.

**Quick reference:**

**Local backend (default):**
- NVIDIA GPU with CUDA 13.0+ driver recommended (CUPTI profiling; falls back to CUDA events)
- Conda environment named `fi-bench` (hardcoded in `scripts/bench.sh`)
- Environment variable `FIB_DATASET_PATH` pointing to the trace set

**Modal backend:**
- `--gpu` flag required (e.g., `--gpu b200`, `--gpu h100`)
- Modal authenticated and volume uploaded
- No local GPU or CUDA driver required

**Note on Environment Name**: The benchmark scripts (`scripts/bench.sh`, `scripts/bench_modal.sh`) hardcode `conda run -n fi-bench`. If you use a different environment name, either:
- Rename your environment to `fi-bench`, or
- Modify the `conda run -n fi-bench` line in `scripts/bench.sh` in spawned child environments
