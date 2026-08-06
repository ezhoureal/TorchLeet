import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bench, build, check

ext = build("vector_add", Path(__file__).with_name("vector_add.cu"))

torch.manual_seed(0)
print("correctness:")
for n in [1, 1000, 1 << 20, (1 << 24) + 7]:
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    check(f"n={n}", ext.vector_add(a, b), a + b)

print("benchmark (n=16M):")
a = torch.randn(1 << 24, device="cuda")
b = torch.randn(1 << 24, device="cuda")
bench("yours", lambda: ext.vector_add(a, b))
bench("torch", lambda: a + b)
