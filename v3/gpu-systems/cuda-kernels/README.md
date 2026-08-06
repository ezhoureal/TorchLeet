# Project 3 — CUDA Kernels

Write real CUDA kernels, JIT-compiled and tested against PyTorch references.
Do them in order — each one builds on techniques from the previous.

| # | Exercise | Difficulty |
|---|----------|------------|
| 1 | `01_vector_add/` | easy |
| 2 | `02_reduction/` | harder |
| 3 | `03_softmax/` | important |
| 4 | `04_layernorm/` | very important |
| 5 | `05_flash_attention/` | advanced |

## Rules

- Implement the `TODO`s in each `.cu` file — kernel body **and** launch code.
- Don't touch `test.py` or `common.py`.
- Your kernel must pass every correctness shape, not just the benchmark one.

## Setup

Already done for this repo: `nvidia-cuda-nvcc` (nvcc 13.0) and friends are in
the uv venv, and `common.py` points `CUDA_HOME` at them automatically. If the
venv is ever recreated, re-run:

```
uv add nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88 nvidia-cuda-cccl==13.3.3.4.1
```

## Running

From the repo root:

```
cd v3/gpu-systems/cuda-kernels/01_vector_add
uv run python test.py
```

The first run takes ~1 min (nvcc compile); later runs reuse the cached build.
Each test prints correctness per shape, then times your kernel against the
PyTorch reference. Try to at least match the reference.
