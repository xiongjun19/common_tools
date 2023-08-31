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
    return res


def naive_2mm_grad():
    """ naive test for calc gradient"""
    _shape = [2, 3, 4]
    x = gen_input(_shape)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    r = 2
    in_dim = 4
    out_dim = 4
    lora_A = nn.Parameter(torch.ones((r, in_dim)))
    lora_B = nn.Parameter(torch.ones((out_dim, r)))
    nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
    nn.init.kaiming_uniform_(lora_B, a=math.sqrt(7))
    y = torch.matmul(lora_B, lora_A)
    y = torch.matmul(x, y)
    loss = 1 - torch.sum(y)
    loss.backward()
    print("lora_A's grad", lora_A.grad)
    print("lora_B's grad", lora_B.grad)
    return lora_A.grad, lora_B.grad


def cus_2mm_grad():
    """ naive test for calc gradient"""
    _shape = [2, 3, 4]
    x = gen_input(_shape)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    r = 2
    in_dim = 4
    out_dim = 4
    lora_A = nn.Parameter(torch.ones((r, in_dim)))
    lora_B = nn.Parameter(torch.ones((out_dim, r)))
    # nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
    y = Custom2MM.apply(lora_A, lora_B, x)
    loss = 1 - torch.sum(y)
    loss.backward()
    print("lora_A's grad", lora_A.grad)
    print("lora_B's grad", lora_B.grad)
    return lora_A.grad, lora_B.grad


class Custom2MM(Function):
    @staticmethod
    def forward(ctx, w_a, w_b, x):
        ctx.data = x.detach()
        ctx.w_a = w_a.detach()
        ctx.w_b = w_b.detach()
        y = torch.matmul(w_b, w_a)
        y = torch.matmul(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        dim = ctx.data.shape[-1]
        data = ctx.data.reshape([-1, dim])
        data = data.permute(1, 0).contiguous()
        g_dim = grad_out.shape[-1]
        gd = grad_out.reshape([-1, g_dim])
        mid_res = torch.matmul(data, gd)
        gd_a = torch.matmul(ctx.w_b.permute(1, 0).contiguous(), mid_res)
        gd_b = torch.matmul(mid_res, ctx.w_a.permute(1, 0).contiguous())
        return gd_a, gd_b, None


def ns_2mm_grad(begin, end):
    """ naive test for calc gradient"""
    _shape = [2, 3, 4]
    x = gen_input(_shape)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    r = 2
    in_dim = 4
    out_dim = 4
    lora_A = nn.Parameter(torch.ones((r, out_dim)))
    lora_B = nn.Parameter(torch.ones((in_dim, r)))
    # nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
    y = torch.matmul(lora_B, lora_A)
    y = torch.matmul(x, y)
    z = y[..., begin:end]
    loss = 1 - torch.sum(z)
    loss.backward()
    print("lora_A's grad", lora_A.grad)
    print("lora_B's grad", lora_B.grad)
    return lora_A.grad, lora_B.grad


def cus_s_2mm_grad(begin, end):
    """ naive test for calc gradient"""
    _shape = [2, 3, 4]
    x = gen_input(_shape)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    r = 2
    in_dim = 4
    out_dim = 4
    lora_A = nn.Parameter(torch.ones((r, in_dim)))
    lora_B = nn.Parameter(torch.ones((out_dim, r)))
    # nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
    nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
    nn.init.kaiming_uniform_(lora_B, a=math.sqrt(7))

    y = CustomS_2MM.apply(lora_A, lora_B, x, begin, end)
    loss = 1 - torch.sum(y)
    loss.backward()
    print("lora_A's grad", lora_A.grad)
    print("lora_B's grad", lora_B.grad)
    return lora_A.grad, lora_B.grad


class CustomS_2MM(Function):
    @staticmethod
    def forward(ctx, w_a, w_b, x, begin, end):
        ctx.data = x.detach()
        ctx.w_a = w_a.detach()
        ctx.w_b = w_b.detach()
        ctx.b = begin
        ctx.e = end
        y = torch.matmul(w_b, w_a)
        ctx.in_dim = w_b.shape[0]
        ctx.out_dim = w_a.shape[1]
        y = torch.matmul(x, y)[..., begin:end]
        return y

    @staticmethod
    def backward(ctx, grad_out):
        in_dim = ctx.in_dim
        out_dim = ctx.out_dim
        data = ctx.data.reshape([-1, in_dim])
        data = data.permute(1, 0).contiguous()
        g_dim = grad_out.shape[-1]
        gd = grad_out.reshape([-1, g_dim])
        b = ctx.b
        e = ctx.e
        l = b
        r = out_dim - e
        mid_res = torch.matmul(data, gd)
        mid_res = F.pad(mid_res, (l, r), "constant", 0)
        gd_a = torch.matmul(ctx.w_b.permute(1, 0).contiguous(), mid_res)
        gd_b = torch.matmul(mid_res, ctx.w_a.permute(1, 0).contiguous())
        return gd_a, gd_b, None, None, None


if __name__ == '__main__':
    # naive_grad()
    print("the naive result: ")
    ng_a, ng_b = naive_2mm_grad()
    print("*" * 36)
    print("cus_2_mm test")
    s_a, s_b = cus_2mm_grad()

    print("*" * 36)
    print("navie_selec test")
    ns_a, ns_b = ns_2mm_grad(0, 2)

    print("*" * 36)
    print("cus_s_2_mm test")
    cs_a, cs_b = cus_s_2mm_grad(0, 2)
    print("*" * 36)
    cs_a2, cs_b2 = cus_s_2mm_grad(2, 4)
    tot_a_g = cs_a + cs_a2
    tot_b_g = cs_b + cs_b2
    print("total_a_g")
    print(tot_a_g)
    print("total_b_g")
    print(tot_b_g)

