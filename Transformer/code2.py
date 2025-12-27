import torch
import torch.nn as nn

embed = 32
heads = 4

src = torch.randn(1, 4, embed)
tgt = torch.randn(1, 3, embed)

enc_layer = nn.TransformerEncoderLayer(embed, heads)
dec_layer = nn.TransformerDecoderLayer(embed, heads)

encoder = nn.TransformerEncoder(enc_layer, 1)
decoder = nn.TransformerDecoder(dec_layer, 1)

memory = encoder(src)
out = decoder(tgt, memory)

print(out.shape)
