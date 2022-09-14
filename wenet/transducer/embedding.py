import torch
from torch import nn
class S2cEmbedding(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embed_size,
        embed_dropout,
    ):
        assert len(input_dim) == 2
        super().__init__()
        char_size, syl_size = input_dim
        self.embed_char = nn.Embedding(char_size, embed_size)
        self.embed_syl = nn.Embedding(syl_size, embed_size)
        self.embed_dropout = nn.Dropout(p=embed_dropout)

    def forward(
        self,
        txt_input,
        syl_input,
    ):
        txt_input = self.embed_char(txt_input)
        syl_input = self.embed_syl(syl_input)
        x = torch.cat((txt_input, syl_input), dim=-1)
        x = self.embed_dropout(x)
        return x


    
