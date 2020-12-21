# coding=utf8

import torch


def dropout(x, p=0.5, mode='train'):
    if mode != 'train':
        return x
    cur_p = torch.rand_like(x)
    mask = cur_p > p
    return x * mask / (1 - p)