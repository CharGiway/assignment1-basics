import math
import torch
from torch import nn
from einops import rearrange


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)
        inv = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, dtype=torch.float32) / float(self.d_k)))
        pos = torch.arange(0, self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        ang = pos * inv.unsqueeze(0)
        cos = torch.cos(ang)
        sin = torch.sin(ang)
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        xe = x[..., :, 0::2]
        xo = x[..., :, 1::2]
        ye = xe * cos - xo * sin
        yo = xo * cos + xe * sin
        y = torch.stack((ye, yo), dim=-1)
        return rearrange(y, "... seq h two -> ... seq (h two)")

