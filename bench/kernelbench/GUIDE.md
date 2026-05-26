# KernelBench Default Benchmark

Self-contained evaluation script for AKO4ALL. Inlines core logic from [KernelBench](https://github.com/KernelBench/KernelBench) so no external dependency is needed.

## Setup

When `bench/` is empty (default bench mode), use `bench/kernelbench/bench.py` as the benchmark.

### Inputs to assemble

Three pieces must be reachable at eval time:

| Piece | Source priority | Required |
|-------|-----------------|----------|
| `class Model(nn.Module)` with `forward()` | `--ref` file | Yes |
| `get_inputs()` returning sample input tensors | `--inputs` file > `--ref` file | Yes |
| `get_init_inputs()` returning constructor args | `--inputs` file > `--ref` file > defaults to `[]` | No |

The agent's job during Setup is to assemble these three pieces from whatever the user provided — by writing files into `input/` (or pointing `--ref` / `--inputs` at existing paths) so the bench command has what it needs. How to assemble them is up to the agent.

### Common assembly patterns

These are typical paths, not rules. Use whichever shape fits the situation.

- **KernelBench-format input** — A single file already contains all three pieces. Use it directly as `--ref`; omit `--inputs`.
- **Raw kernel** (CUDA / Triton / CuTe-DSL / TileLang / ...) — Wrap the kernel into a `class Model` (e.g., `torch.utils.cpp_extension.load_inline` for CUDA) and define `get_inputs()`. Either keep both in one file used as `--ref`, or split: `class Model` in `--ref`, `get_inputs()` in a separate `input/inputs.py` passed via `--inputs`.
- **Kernel + separate input data** — User provides a kernel plus data in some non-Python format (`.npz`, `.bin`, shape lists in `.txt`, etc.). Wrap the kernel into `class Model` in `--ref`, then write `input/inputs.py` whose `get_inputs()` loads the data file (`np.load`, `torch.load`, custom parser — agent's call) and returns tensors. Pass it via `--inputs`.
- **Kernel referenced by external path** — User points at a kernel outside this repo (e.g., a path into a KernelBench dataset). `--ref` and `--inputs` accept arbitrary paths; no need to copy files into `input/`.

### Pitfall: `get_inputs()` must produce fresh data per call

The bench script calls `get_inputs()` once per correctness trial (5 by default) and again for timing. If `get_inputs()` returns a module-level cached tensor, every trial sees identical data — correctness checks pass trivially and timing reflects cache-warm performance. Make `get_inputs()` regenerate inputs each call (`torch.randn`, re-`np.load`, etc.).

### Bench command

```
python bench/kernelbench/bench.py --ref input/<ref>.py --solution solution/<kernel>.py [--inputs input/inputs.py] --verbose
```

If the file passed to `--inputs` also defines `get_init_inputs()`, it overrides the ref's version too. The solution file keeps `class Model` — the bench script transparently renames it to `class ModelNew` before evaluation.

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
| `--ref` | (required) | Path to reference kernel (must define `class Model`) |
| `--solution` | (required) | Path to optimized kernel |
| `--inputs` | (none) | Optional file defining `get_inputs()` (required) and `get_init_inputs()` (optional); overrides definitions in `--ref` |
| `--timing-method` | `cuda_event` | `cuda_event`, `host_time` |
| `--precision` | `float32` | `float32`, `float16`, `bfloat16` |
| `--backend` | `cuda` | `cuda`, `triton`, `tilelang`, `cute`, `hip` (see note below) |
| `--num-correct-trials` | `5` | Number of correctness check iterations |
| `--num-perf-trials` | `100` | Number of performance timing iterations |
| `--verbose` | off | Print detailed debug info |
| `--self-test` | off | Run source transformation self-test and exit |

### Backend selection matters

The default `--backend cuda` loads the solution via `exec(<source>, context)` —
which **rejects `@triton.jit` functions** with `@jit functions should be defined
in a Python file` (Triton requires source from a real `.py`). If your solution
uses Triton / TileLang / CuTe, pass `--backend triton` (or the matching name).
The Triton-family backends load via tempfile + `importlib` so `@jit` source
inspection works.

| Solution language | Required `--backend` |
|-------------------|----------------------|
| Raw CUDA via `torch.utils.cpp_extension.load[_inline]` | `cuda` (default) |
| Triton (`@triton.jit`) | `triton` |
| TileLang | `tilelang` |
| CuTe | `cute` |
| HIP | `hip` |

## Solution File Requirements

- The solution file must contain `class Model(nn.Module)` with a `forward()` method matching the reference's signature.
- The bench script handles `Model` -> `ModelNew` renaming transparently — **do not** rename the class in the solution file.
- Do not include `get_inputs()` or `get_init_inputs()` in the solution file. The bench script strips the solution's module-level tail (variables and functions following the last class) as an anti-cheat boundary — any such definitions would be silently dropped, and the solution cannot influence which inputs it is tested against.

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
