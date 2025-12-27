import torch, torch.nn as nn

x = torch.randn(1, 5, 32)
enc = nn.TransformerEncoderLayer(32, 4)
print(enc(x).shape)
