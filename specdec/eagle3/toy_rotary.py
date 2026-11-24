"""Small default M-RoPE fixture for CPU tests only; real training uses target rotary."""
import torch
from torch import nn


class ToyRotary(nn.Module):
    def __init__(self, head_dim=8, partial_rotary_factor=0.5, theta=10000.0,
                 mrope_section=(1, 1, 0)):
        super().__init__()
        dim = int(head_dim * partial_rotary_factor)
        self.register_buffer("inv_freq", 1.0 / theta ** (torch.arange(0, dim, 2).float() / dim))
        self.mrope_section = mrope_section

    def forward(self, x, position_ids):
        freqs = position_ids[..., None].float() * self.inv_freq.float()
        base = freqs[0].clone()
        for axis in (1, 2):
            index = slice(axis, self.mrope_section[axis] * 3, 3)
            base[..., index] = freqs[axis, ..., index]
        phase = torch.cat((base, base), dim=-1)
        return phase.cos().to(x.dtype), phase.sin().to(x.dtype)
