# coding=utf8

import torch


def batch_norm(x, gamma, beta, moment,  eps, param):
    """
    x shape of [N, D]
    gamma: scale parameter shape of [D,]
    beta: shift parameter shape of [D, ]
    eps: 用来防止除以0的
    param: dictionary {
        mode: train or test
        running_mean: "用来计算移动平均的均值"
        running_var: "用来计算移动平均的方差"
    }
    """
    mode = param["mode"]
    running_mean = param["running_mean"]
    running_var = param["running_var"]
    if mode == "train":
        m = torch.mean(dim=0)
        std = torch.std(dim=0)
        n_x = (x - m) / (std + eps)
        n_x = gamma * n_x + beta
        running_mean = moment * running_mean + (1 - moment) * m
        running_var = moment * running_var + (1 - moment) * std
    else:
        n_x = (x - running_mean) / (running_var + eps)
        n_x = gamma * n_x + beta
    param["running_mean"] = running_mean
    param["running_var"] = running_var
    return n_x
