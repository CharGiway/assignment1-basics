from __future__ import annotations

import numpy as np
import torch
import numpy.typing as npt


def get_batch(dataset: npt.NDArray[np.int_], batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(len(dataset))
    cl = int(context_length)
    num_possible = n - cl
    assert num_possible > 0
    starts = np.random.randint(0, num_possible, size=(batch_size,), dtype=np.int64)
    offsets = np.arange(cl, dtype=np.int64)[None, :]
    idx = starts[:, None] + offsets
    x_np = dataset[idx]
    y_np = dataset[idx + 1]
    x = torch.as_tensor(x_np, dtype=torch.long)
    y = torch.as_tensor(y_np, dtype=torch.long)
    return x.to(device), y.to(device)
