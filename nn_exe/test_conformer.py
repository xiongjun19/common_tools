# coding=utf8

import torch
from conformer import ConformerEncoder
from conformer import RelPosEncXL

def test():
    layers = 6
    hidden_dims = 768
    heads = 12
    d_ffn = 3072
    dropout = 0.2
    encoder = ConformerEncoder(layers, hidden_dims, d_ffn, heads, dropout=dropout)
    pe_emb = RelPosEncXL(hidden_dims)
    t_input = torch.rand(2, 80, hidden_dims)
    pe = pe_emb(t_input)
    t_out = encoder(t_input, pos_embs=pe)
    print(t_out)


if __name__ == '__main__':
    test()
