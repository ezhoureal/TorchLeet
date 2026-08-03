import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.param = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + 1e-5)
        return x * self.param + self.bias

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.param = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        rms = torch.sum(x**2, dim=-1, keepdim=True) / float(x.shape[-1])
        x = x * rms.rsqrt()
        return x * self.param + self.bias

class AdaRMSNorm(nn.Module):
    def __init__(self, dim, cond_dim=2):
        super().__init__()
        self.scale = nn.Linear(cond_dim, dim)
        self.shift = nn.Linear(cond_dim, dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, x, cond):
        rms = torch.sum(x**2, dim=-1, keepdim=True) / float(x.shape[-1])
        x = x * rms.rsqrt()
        return x * (self.scale(cond) + 1) + self.shift(cond)

torch.manual_seed(5)
data = torch.rand(1, 5) * 17 + 6
norm = LayerNorm(5)
ref = nn.LayerNorm(5)


output = norm(data)
ref_output = ref(data)
assert torch.allclose(output, ref_output, atol=1e-6), f'output = {output}, ref = {ref(data)}'

rms_norm = RMSNorm(5)
rms_output = rms_norm(data)
ref_rms = nn.RMSNorm(5)
ref_rms_output = ref_rms(data)
assert torch.allclose(ref_rms_output, rms_output, atol=1e-6), f'output = {rms_output}, ref = {ref_rms_output}'

ada = AdaRMSNorm(5)
time_cond = torch.rand(2,)
output = ada(data, time_cond)