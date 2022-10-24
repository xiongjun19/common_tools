# coding=utf8

import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, 2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = lambda x: x.view(-1)
        self.fc1 = nn.Linear(160, 5)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x.register_hook(lambda grad: torch.clamp(grad,  min=0))
        x.register_hook(lambda grad: print("Gradients less than zero: ", bool((grad <0).any())))
        return self.fc1(self.flatten(x))


net = MyNet()


grad_res_dict = {}


def store_gradient_info(grad_res_dict, model):
    named_paras = model.named_parameters()
    for n, p in named_paras:
        if p.requires_grad:
            if n not in grad_res_dict:
                grad_res_dict[n] = []
            if p.grad is None:
                print(f"{n}'s gradient is None")
            else:
                grad_res_dict[n].append(p.grad.abs().mean().cpu().item())



inp = torch.randn(1, 3, 8, 8)
out = net(inp)

(1-out.mean()).backward()
store_gradient_info(grad_res_dict, net)
print("The biases are", net.fc1.bias.grad)
print(grad_res_dict)
