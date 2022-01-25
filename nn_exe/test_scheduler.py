# coding=utf8

import seaborn as sns
import math
from transformers import get_linear_schedule_with_warmup
from transformers import BertModel
from transformers import AdamW
import matplotlib.pyplot as plt

def test():
    model = BertModel.from_pretrained('bert-base-chinese')
    optimizer = AdamW(model.parameters(), lr=4e-4)
    total_steps = 10000
    warmup_ratio = 0.01
    warmup_step =math.floor(total_steps * warmup_ratio)
    lr_sch = get_linear_schedule_with_warmup(optimizer, warmup_step, total_steps)
    lr_res = []
    for step in range(total_steps):
        lr_sch.step()
        print(lr_sch.get_last_lr()[0])
        lr_res.append(lr_sch.get_last_lr()[0])
    plt.plot(lr_res)
    plt.show()



if __name__ == '__main__':
    test()
