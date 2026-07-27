import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        assert d_model % 2 == 0
        inv_freq = 1 / (10000 ** torch.arange(0, d_model, 2).float() / float(d_model))
        seq = torch.arange(0, max_len)
        seq = torch.einsum('i,j->ij', seq, inv_freq)

        seq = torch.cat([seq, seq], dim=1)
        print(f'seq dimension = {seq.shape}')

        self.register_buffer('cos_cache', torch.cos(seq))
        self.register_buffer('sin_cache', torch.sin(seq))
        self.d_model = d_model

    def _rotate_half(self, x):
        half_d = self.d_model // 2
        x1, x2 = x[:, :, :half_d], x[:, :, half_d:]
        return torch.concat([-x2, x1], dim=-1)

    def forward(self, x):
        batch_size, len, d = x.shape
        assert d == self.d_model
        cos = self.cos_cache[:len]
        sin = self.sin_cache[:len]

        return cos * x + sin * self._rotate_half(x)

B = 2
T = 17
D = 64
data = torch.rand(B, T, D)
rope = RoPE(20, D)
roped = rope(data)
print(f'roped = {roped}')