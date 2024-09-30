import torch
from torch import nn
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, N: int = 10000, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.N = N
        self.model_dim = model_dim
        self.register_buffer('matrix', torch.zeros(N, model_dim))
        even = torch.arange(0, model_dim, 2)
        odd = torch.arange(1, model_dim, 2)
        self.matrix[:, even] = torch.sin(torch.arange(N).float().unsqueeze(1) / torch.pow(N, 2 * even / model_dim))
        self.matrix[:, odd] = torch.cos(torch.arange(N).float().unsqueeze(1) / torch.pow(N, 2 * odd / model_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch_size, seq_len, model_dim)
        seq_len = x.size(1)
        return x + self.matrix[:seq_len, :].unsqueeze(0)