import torch
from torch import nn
import einx
from cs336_basics.nn.rmsnorm import RMSNorm
from cs336_basics.nn.mha import MultiHeadSelfAttention
from cs336_basics.nn.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        use_rope: bool = True,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.use_rope = bool(use_rope)
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        self.theta = float(theta)

        self.ln1 = RMSNorm(d_model=self.d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            device=device,
            dtype=dtype,
            use_rope=self.use_rope,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
        )
        self.ln2 = RMSNorm(d_model=self.d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=self.d_model, d_ff=self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        t = x.shape[-2]
        h = self.ln1(x)
        if self.use_rope and token_positions is None:
            pos = torch.arange(t, device=x.device, dtype=torch.long)
            ones = torch.ones((b,), device=x.device, dtype=torch.long)
            token_positions = einx.multiply("b, t -> b t", ones, pos)
        h = self.attn(h, token_positions=token_positions)
        x = x + h
        h2 = self.ln2(x)
        h2 = self.ffn(h2)
        y = x + h2
        return y
