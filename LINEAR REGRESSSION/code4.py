import torch
import torch.nn as nn

model = nn.Linear(1, 1)
x = torch.tensor([[1.0],[2.0],[3.0]])
y = torch.tensor([[2.0],[4.0],[6.0]])

out = model(x)
print(out)
