# KernelBench Default Benchmark

Self-contained evaluation script for AKO4ALL. Inlines core logic from [KernelBench](https://github.com/KernelBench/KernelBench) so no external dependency is needed.

## Setup

When `bench/` is empty (default bench mode), use `bench/kernelbench/bench.py` as the benchmark. Determine the input format:

- **KernelBench-format input**: The input file already has `class Model(nn.Module)` with a `forward()` method, plus top-level `get_inputs()` and `get_init_inputs()` functions. Use the input file directly — no wrapping needed.
- **Raw kernel input** (CUDA, Triton, CuTe-DSL, etc.): The agent must wrap the kernel into KernelBench format:
  1. Create a new `.py` file in `input/` (e.g., `input/kernel_kb.py`) containing:
     - `class Model(nn.Module)` with `forward()` calling the kernel (e.g., via `torch.utils.cpp_extension.load_inline` for CUDA)
     - Top-level `get_inputs()` returning sample inputs
     - Top-level `get_init_inputs()` returning constructor args (often `[]`)
  2. This wrapped file becomes the reference.

If the user provides `input/reference.py`, use that as the reference (`--ref`). Otherwise, the input kernel file (original or wrapped) serves as the reference.

**Bench command** (for TASK.md step 5):
```
python bench/kernelbench/bench.py --ref input/<ref>.py --solution solution/<kernel>.py --verbose
```

The solution file keeps `class Model` — the bench script transparently renames it to `class ModelNew` before evaluation.

## Output Format

Each run prints structured lines (parsed by the agent):

```
COMPILED: True
CORRECT: True
RUNTIME: 0.4523
REF_RUNTIME: 1.2301
SPEEDUP: 2.7197x
```

- **COMPILED** — whether the solution compiled successfully
- **CORRECT** — whether outputs match the reference (within precision tolerance)
- **RUNTIME** — solution kernel mean execution time in milliseconds
- **REF_RUNTIME** — reference kernel mean execution time in milliseconds
- **SPEEDUP** — `REF_RUNTIME / RUNTIME`

Exit code: `0` = correct, `1` = incorrect or failed.

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--ref` | (required) | Path to reference kernel |
| `--solution` | (required) | Path to optimized kernel |
| `--timing-method` | `cuda_event` | `cuda_event`, `host_time` |
| `--precision` | `float32` | `float32`, `float16`, `bfloat16` |
| `--backend` | `cuda` | `cuda`, `triton`, `tilelang`, `cute`, `hip` |
| `--num-correct-trials` | `5` | Number of correctness check iterations |
| `--num-perf-trials` | `100` | Number of performance timing iterations |
| `--verbose` | off | Print detailed debug info |
| `--self-test` | off | Run source transformation self-test and exit |

## Solution File Requirements

- The solution file must contain `class Model(nn.Module)` with a `forward()` method matching the reference's signature.
- The bench script handles `Model` -> `ModelNew` renaming transparently — **do not** rename the class in the solution file.
- Do not include `get_inputs()` or `get_init_inputs()` in the solution file — the reference provides them.

## Correctness Tolerances

Inspired by [torchbench](https://github.com/pytorch/benchmark):

| Precision | Tolerance (atol & rtol) |
|-----------|------------------------|
| float32   | 1e-4                   |
| float16   | 1e-2                   |
| bfloat16  | 1e-2                   |

## Timing Methods

- **cuda_event** (default): Uses `torch.cuda.Event` for device-side timing. Measures cold-cache performance (L2 thrashed before each trial). Most accurate for GPU kernel time.
- **host_time**: Host-side wall-clock timing via `time.perf_counter()`. Includes Python overhead, CUDA launch costs, and synchronization. Results may be longer than device-side timings.
