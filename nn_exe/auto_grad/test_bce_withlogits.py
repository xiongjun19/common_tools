# coding=utf8


import torch


target = torch.ones([10, 64], dtype=torch.float32)
output = torch.full([10, 64], 1.5)
pos_weight = torch.ones([64])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss = criterion(output, target)
print(loss)
