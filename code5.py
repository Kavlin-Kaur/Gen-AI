import torch
import torch.nn as nn

gru = nn.GRU(input_size=5, hidden_size=3, batch_first=True)
x = torch.randn(2, 4, 5)  
out, h = gru(x)
print(out.shape, h.shape)
