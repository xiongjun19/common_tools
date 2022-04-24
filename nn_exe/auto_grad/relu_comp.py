# coding=utf8

'''
usage: python relu_comp.py --option 0  --w-opt 1
'''


import torch
import torch.nn.functional as F
from torch.autograd import Function

def calc_loss(pred, y):
    y = torch.Tensor(y)
    return torch.norm((y - pred))


class NormRelu(object):
    def __init__(self, w):
        self.w = w
        self.w.requires_grad_()

    def calc_gradient(self, x, y):
        h = torch.matmul(self.w, x)
        pred = F.relu(h)
        loss = calc_loss(pred, y)
        print("loss is: ")
        print(loss)
        loss.backward()
        return self.w.grad


class CustomRelu(object):
    def __init__(self, w):
        self.w = w
        self.w.requires_grad_()

    def calc_gradient(self, x, y):
        pred = CusFunc.apply(self.w, x, y)
        loss = calc_loss(pred, y)
        print("loss is: ")
        print(loss)
        loss.backward()
        return self.w.grad


class CusFunc(Function):
    @staticmethod
    def forward(ctx, w, x, y):
        w = w.detach()
        h = torch.matmul(w, x)
        pred = F.relu(h)
        ctx.x_mul = x.detach()
        rl = torch.zeros_like(h)
        _bool = h > 0
        rl.masked_fill_(_bool, 1.)
        ctx.rl = rl
        return pred

    @staticmethod
    def backward(ctx, grad_out):
        return ctx.x_mul * ctx.rl * grad_out, None, None


def run_and_print(cls, w, x, y):
    _obj = cls(w)
    _grad = _obj.calc_gradient(x, y)
    print(f"gradient of norm of {cls.__name__} is: ")
    print(_grad)




def main(args):
    # 本函数用来完成测试，目前有两种选项用来完成测试
    # 1. opiton 用来选择到底是那种方式来测试
    # 2. w_opt 用来选择到底是用那个weight 来测试
    cls_arr = [NormRelu, CustomRelu]
    f_cls = cls_arr[args.option]
    w = torch.FloatTensor([1., 1.])
    if args.w_opt > 0:
        w = torch.FloatTensor([1., -1.])
    x = torch.FloatTensor([1, 2])
    y = torch.FloatTensor([1])
    run_and_print(f_cls, w, x, y)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=int, default=0)
    parser.add_argument('--w-opt', type=int, default=0)
    t_args = parser.parse_args()
    main(t_args)
