import math
import torch
from torch import nn


class _LinearWeight(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        sigma = math.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("... i, o i -> ... o", x, self.weight)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.w1 = _LinearWeight(in_features=self.d_model, out_features=self.d_ff, device=device, dtype=dtype)
        self.w2 = _LinearWeight(in_features=self.d_ff, out_features=self.d_model, device=device, dtype=dtype)
        self.w3 = _LinearWeight(in_features=self.d_model, out_features=self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = self.w3(x)
        a_silu = a * torch.sigmoid(a)
        y = a_silu * b
        return self.w2(y)

