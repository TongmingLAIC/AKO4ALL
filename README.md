# kernel-opt-agent

A tool for automated GPU kernel optimization using Code Agents (e.g. Claude Code). Built on the [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) SDK, it supports any flashinfer-bench compatible dataset, including user-defined operators.

## Quick Start

```bash
# List available operators
bash setup.sh

# Create an optimization environment (local A100, from scratch)
bash setup.sh --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64

# Enter the child environment and start your code agent
cd ../kernel-opt-agent-run-001
claude

# Then send an optimization instruction, e.g.:
# "Read CLAUDE.md and Info.md, then begin optimizing the kernel."
```

## Setup Options

```bash
bash setup.sh \
  --operator <name> \
  [--dataset /path/to/dataset] \    # default: $FIB_DATASET_PATH
  [--mode scratch|existing] \       # default: scratch
  [--backend local|modal] \         # default: local
  [--kernel /path/to/kernel.py] \   # required for existing mode
  [--name <label>] \                # optional label for the run
  [--prompt /path/to/prompt.md] \   # custom system prompt
  [--info /path/to/info.md]         # custom info for the agent
```

### Examples

```bash
# Local A100, from scratch, with label
bash setup.sh --operator gdn_decode_qk4_v8_d128_k_last --name "gdn_exp_1"

# Local A100, from existing kernel
bash setup.sh --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
    --mode existing --kernel /path/to/kernel.py

# Modal B200, from scratch
bash setup.sh --operator moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
    --backend modal --name "moe_b200"

# Custom dataset
bash setup.sh --operator my_custom_op --dataset /path/to/my/traceset

# Custom prompt and info
bash setup.sh --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
    --prompt ./my_prompt.md --info ./my_hints.md
```

## How It Works

1. `setup.sh --operator <name>` discovers the operator definition and workloads from the dataset
2. Creates a child environment `kernel-opt-agent-run-NNN[-label]/` with everything the agent needs
3. You start your code agent (e.g. `claude`) in the child environment
4. The agent reads `CLAUDE.md` and `Info.md`, then iteratively optimizes the kernel

### Child Environment Structure

```
kernel-opt-agent-run-NNN/
├── CLAUDE.md                   # Auto-generated task spec (prompt + operator details)
├── Info.md                     # User-editable hints and context
├── config.toml                 # Operator configuration
├── .gitignore
├── docs/
│   ├── definition.json         # Operator definition
│   └── workloads.jsonl         # Benchmark workloads
├── solution/
│   └── triton/
│       └── kernel.py           # The kernel to optimize
├── scripts/
│   ├── bench.sh                # Benchmark script
│   ├── run_local.py/run_modal.py  # Benchmark runner
│   └── pack_solution.py        # Pack kernel for evaluation
├── trajectory/                 # Auto-created, stores each bench run
└── .claude/
    └── settings.local.json     # Permission fallback
```

## Customization

### Custom Prompt

The default prompt (`templates/prompt.md`) tells the agent how to work. Override it with `--prompt`:

```bash
bash setup.sh --operator my_op --prompt ./my_prompt.md
```

### Custom Info

`Info.md` is placed in each child environment for ad-hoc hints. Override the template with `--info`:

```bash
bash setup.sh --operator my_op --info ./my_analysis.md
```

Or edit `Info.md` directly in the child environment before launching.

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

## Requirements

**Local backend (default):**
- NVIDIA A100-SXM4-40GB
- Conda env `fi-bench` (Python 3.12, Triton 3.5.1, PyTorch 2.9.1+cu128, `flashinfer-bench`)
- `FIB_DATASET_PATH` pointing to the trace set, or use `--dataset`

**Modal backend:**
- Conda env `fi-bench` (with `modal` and `flashinfer-bench`)
- Modal authenticated (`conda run -n fi-bench --no-capture-output modal setup`)
- Modal volume `flashinfer-trace` with the trace set uploaded
