import torch.nn as nn
import torch

IN = 3
PATCH_SIZE=16
D = 16 * PATCH_SIZE ** 2
MAX_T = 4096

class RoPE(nn.Module):
    def __init__(self):
        super().__init__()
        inv_freq = 1 / 10000 ** (torch.arange(0, D, 2) / D)

        pos = torch.arange(0, MAX_T)
        pairs = torch.einsum('i,j->ij', pos, inv_freq) # [T, D / 2]
        pairs = torch.stack([pairs, pairs], dim=-1).flatten(1) # [T, D]
        cos_cache = torch.cos(pairs)
        sin_cache = torch.sin(pairs)
        print(f'sin shape = {sin_cache.shape}')
        self.register_buffer('cos_cache', cos_cache)
        self.register_buffer('sin_cache', sin_cache) # [T, D]
    
    def _rotate_emb(self, x: torch.Tensor):
        assert D % 2 == 0
        x1, x2 = x[..., :D // 2], x[..., D // 2:]
        return torch.concat([-x2, x1], dim=-1)

    def forward(self, x):
        B, T, d = x.shape
        assert d == D
        cos, sin = self.cos_cache[:T], self.sin_cache[:T]
        return x * cos + self._rotate_emb(x) * sin


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_proj = nn.Conv2d(3, D, kernel_size=PATCH_SIZE, stride=4)
        self.qkv = nn.Linear(D, 3 * D)
        self.o = nn.Linear(D, D)
        self.rope = RoPE()
        self.norm1 = nn.RMSNorm(D)
        self.attn = nn
        self.mlp = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.GELU(),
            nn.Linear(D * 4, D)
        )
        self.norm2 = nn.RMSNorm(D)
    
    def forward(self, x):
        # input: [B, C, W, H]
        x = self.conv_proj(x)
        B, d, W, H = x.shape
        assert d == D

        x = x.flatten(2)
        x = x.transpose(1, 2) # -> [B, T, D]

        x = self.rope(x)

        x = self.norm1(x)
        q, k, v = self.qkv(x).split(D, dim=2)
        attn = nn.functional.scaled_dot_product_attention(q, k, v)

        x = x + self.o(attn)

        x = x + self.mlp(self.norm2(x))
        return x

if __name__ == "__main__":
    vit = VisionTransformer()
    data = torch.rand(1, 3, 256, 256)
    x = vit(data)
    print(f'x.shape = {x.shape}')