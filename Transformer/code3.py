import torch
import torch.nn as nn

transformer = nn.Transformer(
    d_model=32,
    nhead=4,
    num_encoder_layers=1,
    num_decoder_layers=1
)

src = torch.randn(5, 1, 32)  # seq=5
tgt = torch.randn(4, 1, 32)  # seq=4

out = transformer(src, tgt)

print(out.shape)
