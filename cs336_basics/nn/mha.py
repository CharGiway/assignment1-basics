import math
import torch
from torch import nn
from einops import rearrange
from cs336_basics.nn.sdpa import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_head = self.d_model // self.num_heads
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        # Store weights as (out_features, in_features)
        self.q_proj = nn.Parameter(torch.empty((self.d_model, self.d_model), **factory_kwargs))
        self.k_proj = nn.Parameter(torch.empty((self.d_model, self.d_model), **factory_kwargs))
        self.v_proj = nn.Parameter(torch.empty((self.d_model, self.d_model), **factory_kwargs))
        self.o_proj = nn.Parameter(torch.empty((self.d_model, self.d_model), **factory_kwargs))

        def _init(param: torch.Tensor):
            sigma = math.sqrt(2.0 / (self.d_model + self.d_model))
            torch.nn.init.trunc_normal_(param, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)

        for p in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            _init(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        seq_len = x.shape[-2]
        q = torch.einsum("... t d, od -> ... t o", x, self.q_proj)
        k = torch.einsum("... t d, od -> ... t o", x, self.k_proj)
        v = torch.einsum("... t d, od -> ... t o", x, self.v_proj)
        q = rearrange(q, "... t (h d) -> ... h t d", h=self.num_heads)
        k = rearrange(k, "... t (h d) -> ... h t d", h=self.num_heads)
        v = rearrange(v, "... t (h d) -> ... h t d", h=self.num_heads)
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        out_heads = scaled_dot_product_attention(q, k, v, mask=causal)
        out = rearrange(out_heads, "... h t d -> ... t (h d)")
        y = torch.einsum("... t d, od -> ... t o", out, self.o_proj)
        return y

