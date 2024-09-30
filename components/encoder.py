import torch
from torch import nn
from torch.nn import functional as F
from utils.attention import MultiHeadAttention
from utils.positional import PositionalEncoding

__all__ = ['OriginalTransformerEncoderLayer', 'OriginalTransformerEncoder']

class OriginalTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.self_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x.clone()
        x = self.self_attn.forward(x)
        x = self.layer_norm1(x + original_x)
        original_x = x.clone()
        x = self.fc.forward(x)
        return self.layer_norm2(x + original_x)
    
class OriginalTransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int = 6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([OriginalTransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch_size, seq_len, embed_dim)
        x = self.pos_enc.forward(x)
        for layer in self.layers:
            x = layer.forward(x)
        return x