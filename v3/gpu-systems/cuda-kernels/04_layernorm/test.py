import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bench, build, check

ext = build("layernorm", Path(__file__).with_name("layernorm.cu"))

torch.manual_seed(0)
print("correctness:")
for rows, cols in [(1, 1), (7, 33), (128, 768), (1000, 4097)]:
    x = torch.randn(rows, cols, device="cuda") * 10
    gamma = torch.randn(cols, device="cuda")
    beta = torch.randn(cols, device="cuda")
    ref = F.layer_norm(x, (cols,), gamma, beta, eps=1e-5)
    check(f"({rows}, {cols})", ext.layernorm(x, gamma, beta, 1e-5), ref, atol=1e-4)

print("benchmark (4096 x 4096):")
x = torch.randn(4096, 4096, device="cuda")
gamma = torch.randn(4096, device="cuda")
beta = torch.randn(4096, device="cuda")
bench("yours", lambda: ext.layernorm(x, gamma, beta, 1e-5))
bench("torch", lambda: F.layer_norm(x, (4096,), gamma, beta, eps=1e-5))
