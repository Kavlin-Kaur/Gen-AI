import torch
import torch.nn as nn

tgt = torch.randn(4, 1, 32)
memory = torch.randn(5, 1, 32)

dec = nn.TransformerDecoderLayer(32, 4)

out = dec(tgt, memory)
print(out.shape)
