# coding=utf8

import torch

def naive_grad(x, val):
    """ naive test for calc gradient"""
    # x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = (x + val) ** 2
    z = torch.mean(y)
    z.backward()
    return x.grad

def test1():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    grad = naive_grad(x, 2)
    print("test1 result: ")
    print(grad)


def test2():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    grad = naive_grad(x, 3)
    print("test1 result: ")
    print(grad)


def test3():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    grad = naive_grad(x, 2)
    print("test3 stage1 result: ")
    print(grad)
    grad = naive_grad(x, 3)
    print("test3 stage2 result: ")
    print(grad)


if __name__ == '__main__':
    test1()
    test2()
    test3()






def main():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    x = torch.tensor([1.0, 2.0], requires_grad=True)
