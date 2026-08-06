import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bench, build, check

ext = build("softmax", Path(__file__).with_name("softmax.cu"))

torch.manual_seed(0)
print("correctness:")
for rows, cols in [(1, 1), (7, 33), (128, 1024), (1000, 4097)]:
    x = torch.randn(rows, cols, device="cuda") * 10
    check(f"({rows}, {cols})", ext.softmax(x), torch.softmax(x, dim=-1), atol=1e-5)

print("benchmark (4096 x 4096):")
x = torch.randn(4096, 4096, device="cuda")
bench("yours", lambda: ext.softmax(x))
bench("torch", lambda: torch.softmax(x, dim=-1))
