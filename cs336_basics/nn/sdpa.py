import math
import torch
from cs336_basics.nn.softmax import softmax


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    q = Q.to(torch.float32)
    k = K.to(torch.float32)
    v = V.to(torch.float32)
    d_k = q.shape[-1]
    scores = torch.einsum("... q d, ... k d -> ... q k", q, k) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = softmax(scores, dim=-1)
    out = torch.einsum("... q k, ... k d -> ... q d", probs, v)
    return out.to(V.dtype)

