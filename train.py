import torch
from torch import nn
from components.encoder import OriginalTransformerEncoder

ote = OriginalTransformerEncoder(embed_dim=512, num_heads=8, num_layers=6).cuda()
x = torch.randn(16, 10, 512).cuda()
print(ote.forward(x).shape)