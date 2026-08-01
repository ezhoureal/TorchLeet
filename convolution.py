import torch
import torch.nn as nn

class Convolution(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size: int, stride: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.stride: int = stride
        self.kernel_size: int = kernel_size
        self.weight = nn.Linear(kernel_size * dim_in, dim_out)

    def forward(self, x):
        B, C, L = x.shape
        assert C == self.dim_in

        new_len = (L - self.kernel_size) // self.stride + 1
        mats = []
        for i in range(new_len):
            pos = i * self.stride
            window = x[:, :, pos:pos+self.kernel_size].flatten(1)
            print(f'window shape = {window.shape}')
            mats.append(window)
        full_mat = torch.stack(mats, dim=-1).transpose(1, 2)
        print(f'full mat shape = {full_mat.shape}')
        return self.weight(full_mat).transpose(1, 2)

data = torch.rand(10, 3, 8)
conv = Convolution(3, 16, 3, 3)

real_conv = nn.Conv1d(3, 16, 3, stride=3)
conved = conv(data)
real_conved = real_conv(data)
print(f'conved shape = {conved.shape}')
assert real_conved.shape == conved.shape, real_conved.shape