import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bench, build, check

ext = build("reduction", Path(__file__).with_name("reduction.cu"))

torch.manual_seed(0)
print("correctness:")
for n in [1, 1000, 1 << 20, (1 << 24) + 7]:
    x = torch.randn(n, device="cuda")
    check(f"n={n}", ext.reduce_sum(x).reshape(()), x.sum(), atol=1e-2)

print("benchmark (n=16M):")
x = torch.randn(1 << 24, device="cuda")
bench("yours", lambda: ext.reduce_sum(x))
bench("torch", lambda: x.sum())
