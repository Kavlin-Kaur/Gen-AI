import torch
import torch.nn as nn

embed = 16
seq = 6

pos = nn.Parameter(torch.randn(1, seq, embed))
tokens = torch.randn(1, seq, embed)

out = tokens + pos
print(out.shape)
