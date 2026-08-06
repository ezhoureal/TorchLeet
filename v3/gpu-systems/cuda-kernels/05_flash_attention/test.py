import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bench, build, check

ext = build("flash_attention", Path(__file__).with_name("flash_attention.cu"))


def reference(q, k, v):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    return torch.softmax(scores, dim=-1) @ v


torch.manual_seed(0)
print("correctness:")
for b, h, s, d in [(1, 1, 16, 16), (1, 2, 128, 64), (2, 4, 513, 64)]:
    q = torch.randn(b, h, s, d, device="cuda")
    k = torch.randn(b, h, s, d, device="cuda")
    v = torch.randn(b, h, s, d, device="cuda")
    check(f"b{b} h{h} s{s} d{d}", ext.flash_attention(q, k, v), reference(q, k, v), atol=1e-4)

print("benchmark (b=4, h=8, s=2048, d=64):")
q = torch.randn(4, 8, 2048, 64, device="cuda")
k = torch.randn(4, 8, 2048, 64, device="cuda")
v = torch.randn(4, 8, 2048, 64, device="cuda")
bench("yours", lambda: ext.flash_attention(q, k, v))
bench("naive torch", lambda: reference(q, k, v))
