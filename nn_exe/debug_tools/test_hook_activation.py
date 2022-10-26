# coding=utf8

import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, 2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = lambda x: x.view(-1)
        self.seq = nn.Sequential(nn.Linear(5, 3), nn.Linear(3, 2))
        self.fc1 = nn.Linear(160, 5)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.fc1(self.flatten(x))
        x = self.seq(x)
        return x


net = MyNet()
visualisation = {}


def hook_fn(m, i, o):
    visualisation[m] = o


def get_all_layers(net):
    for name, layer in net._modules.items():
        if isinstance(layer, nn.Sequential):
            get_all_layers(layer)
        else:
            layer.register_forward_hook(hook_fn)


get_all_layers(net)
out = net(torch.randn(1, 3, 8, 8))
import ipdb; ipdb.set_trace()
print(visualisation)

