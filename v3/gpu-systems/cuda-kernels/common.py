"""Shared utilities for the CUDA kernel exercises.

Handles locating the pip-installed CUDA toolkit (nvcc) inside the uv
virtualenv, JIT-compiling .cu files with torch.utils.cpp_extension, and
checking / benchmarking kernels against PyTorch references.
"""

import glob
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _setup_cuda_home() -> None:
    """Point CUDA_HOME at the nvcc toolkit installed in the venv."""
    if "CUDA_HOME" in os.environ:
        return
    matches = glob.glob(
        os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "cu13")
    )
    if not matches:
        raise RuntimeError(
            "CUDA toolkit not found in the venv. Run: "
            "uv add nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 "
            "nvidia-nvvm==13.0.88 nvidia-cuda-cccl==13.3.3.4.1"
        )
    cuda_home = matches[0]
    os.environ["CUDA_HOME"] = cuda_home
    # the runtime wheel ships libcudart.so.13 but not the linker symlink
    lib = Path(cuda_home) / "lib"
    target, link = lib / "libcudart.so.13", lib / "libcudart.so"
    if target.exists() and not link.exists():
        link.symlink_to(target.name)


_setup_cuda_home()

from torch.utils.cpp_extension import load  # noqa: E402


def build(name: str, source: Path):
    """JIT-compile a .cu file into a Python extension module."""
    return load(
        name=name,
        sources=[str(source)],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def check(label: str, out: torch.Tensor, ref: torch.Tensor, atol: float = 1e-4) -> None:
    max_err = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=atol, rtol=1e-4)
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label:<24} max_err={max_err:.3e}")
    if not ok:
        raise AssertionError(f"{label}: mismatch vs reference (max_err={max_err:.3e})")


def bench(label: str, fn, iters: int = 100) -> float:
    """Time fn() with CUDA events, returns median milliseconds."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"  {label:<24} {ms:8.3f} ms")
    return ms
