# coding=utf8

import torch
from transformer import TransformerEncoder
from transformer import PositionalEncoding 

def test():
    layers = 6
    hidden_dims = 768
    heads = 12
    d_ffn = 3072
    dropout = 0.2
    encoder = TransformerEncoder(layers, heads, d_ffn, d_model=hidden_dims, dropout=dropout)
    pos_encoder = PositionalEncoding(hidden_dims)
    t_input = torch.rand(4, 80, hidden_dims)
    p_t = pos_encoder(t_input)
    t_input += p_t
    import ipdb; ipdb.set_trace()
    t_out = encoder(t_input)
    print(t_out)



if __name__ == '__main__':
    test()
