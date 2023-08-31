# coding=utf8


import numpy as np
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import math


def gen_input(_shape):
    tot_val = np.prod(list(_shape))
    res = torch.arange(0, tot_val, dtype=torch.float32)
    res = res.view(_shape).contiguous()
    print("gen_input is: ", res)
    return res


def naive_matmul_grad():
    _shape = [2, 4]
    x = gen_input(_shape)
    w = nn.Parameter(torch.ones([4, 3]))
    y = torch.matmul(x, w)
    print(y.shape)
    print(y)
    loss = 1 - torch.sum(y)
    loss.backward()
    print("w: ", w)
    print("w's grad: ", w.grad)


def naive_matmul_grad_3d():
    _shape = [2, 3, 4]
    x = gen_input(_shape)
    w = nn.Parameter(torch.ones([4, 3]))
    y = torch.matmul(x, w)
    print(y.shape)
    print(y)
    loss = 1 - torch.sum(y)
    loss.backward()
    print("w: ", w)
    print("w's grad: ", w.grad)
    return w.grad


class Custom3DMat(Function):
    @staticmethod
    def forward(ctx, w, x):
        ctx.data = x.detach()
        y = torch.matmul(x, w)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        dim = ctx.data.shape[-1]
        data = ctx.data.reshape([-1, dim]) 
        data = data.permute(1, 0).contiguous()
        g_dim = grad_out.shape[-1]
        gd = grad_out.reshape([-1, g_dim])
        res = torch.matmul(data, gd)
        return res, None


def cus_mm_grad_3d():
    _shape = [2, 3, 4]
    x = gen_input(_shape)
    w = nn.Parameter(torch.ones([4, 3]))
    y = Custom3DMat.apply(w, x)
    print(y.shape)
    print(y)
    loss = 1 - torch.sum(y)
    loss.backward()
    print("w: ", w)
    print("w's grad: ", w.grad)
    return w.grad


def naive_mm_select():
    _shape = [2, 3, 4]
    x = gen_input(_shape)
    w = nn.Parameter(torch.ones([4, 3]))
    y = torch.matmul(x, w)
    z = y[..., 0:2]
    print(y.shape)
    print(z.shape)
    loss = 1 - torch.sum(z)
    loss.backward()
    print("w: ", w)
    print("w's grad: ", w.grad)
    return w.grad


def cus_mm_select():
    _shape = [2, 3, 4]
    x = gen_input(_shape)
    w = nn.Parameter(torch.ones([4, 3]))
    # y = torch.matmul(x, w)
    # z = y[..., 0:2]
    z = Custom3DMatSelect.apply(w, x, 0, 2)

    loss = 1 - torch.sum(z)
    loss.backward()
    print("w: ", w)
    print("w's grad: ", w.grad)
    return w.grad


class Custom3DMatSelect(Function):
    @staticmethod
    def forward(ctx, w, x, col_begin, col_end):
        ctx.data = x.detach()
        ctx.b = col_begin
        ctx.e = col_end
        ctx.in_dim = w.shape[0]
        ctx.out_dim = w.shape[1]
        y = torch.matmul(x, w)[..., col_begin:col_end]
        return y

    @staticmethod
    def backward(ctx, grad_out):
        in_dim = ctx.in_dim
        out_dim = ctx.out_dim
        data = ctx.data.reshape([-1, in_dim])
        data = data.permute(1, 0).contiguous()
        g_dim = grad_out.shape[-1]
        gd = grad_out.reshape([-1, g_dim])
        res = torch.matmul(data, gd)
        b = ctx.b
        e = ctx.e
        l = b
        r = out_dim - e
        new_res = F.pad(res, (l, r), "constant", 0)
        return new_res, None, None, None


if __name__ == '__main__':
    # naive_grad()
    print("the naive result: ")
    ng = naive_matmul_grad_3d()
    print("*" * 36)
    print('cus 3d matmul: ')
    cg = cus_mm_grad_3d()
    print("diff are: ", ng -cg)
    print("*" * 36)
    print("naive selet test")
    s_ng = naive_mm_select()
    print("ng - s_ng", ng - s_ng)
    print("*" * 36) 
    print("cus_seledt test")
    s_cg = cus_mm_select()


