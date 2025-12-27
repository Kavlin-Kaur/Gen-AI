import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1, 1),
    nn.Sigmoid()
)

x = torch.tensor([[1.],[2.],[3.]])
y = torch.tensor([[0.],[0.],[1.]])

output = model(x)
print(output)
