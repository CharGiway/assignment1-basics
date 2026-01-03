import torch
from torch import nn
import einx
from cs336_basics.nn.embedding import Embedding
from cs336_basics.nn.transformer_block import TransformerBlock
from cs336_basics.nn.rmsnorm import RMSNorm


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)

        self.token_embeddings = Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    use_rope=True,
                    max_seq_len=self.context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=self.d_model, device=device, dtype=dtype)
        self.lm_head = nn.Parameter(torch.empty((self.vocab_size, self.d_model), device=device, dtype=dtype))

        torch.nn.init.trunc_normal_(self.lm_head, mean=0.0, std=(2.0 / (self.vocab_size + self.d_model)) ** 0.5, a=-0.5, b=0.5)

    def forward(self, x_idx: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x_idx)
        b = x.shape[0]
        t = x.shape[1]
        pos = torch.arange(t, device=x.device, dtype=torch.long)
        ones = torch.ones((b,), device=x.device, dtype=torch.long)
        token_positions = einx.multiply("b, t -> b t", ones, pos)
        h = x
        for layer in self.layers:
            h = layer(h, token_positions=token_positions)
        h = self.ln_final(h)
        logits = torch.einsum("... t d, v d -> ... t v", h, self.lm_head)
        return logits

