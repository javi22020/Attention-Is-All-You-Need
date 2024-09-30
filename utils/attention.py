import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        attn_scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        attn_probs = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        out = attn_probs @ v  # (batch_size, num_heads, seq_len, head_dim)
        
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        
        out = self.o_proj.forward(out)  # (batch_size, seq_len, embed_dim)
        
        return out