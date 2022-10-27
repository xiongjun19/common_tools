# coding=utf8
'''
 while self.step < self.train_num_steps:
185
186                 total_loss = 0.
187
188                 for _ in range(self.gradient_accumulate_every):
189                     data = next(self.dl).to(device)
190
191                     with self.accelerator.autocast():
192                         loss = self.model(data)
193                         loss = loss / self.gradient_accumulate_every
194                         total_loss += loss.item()
195
196                     self.accelerator.backward(loss)
197
198                 pbar.set_description(f'loss: {total_loss:.4f}')
199                 lr = [pg["lr"] for pg in self.opt.param_groups]
200     ¦   ¦   ¦   self.wb.log({'train_lr': lr[0],
201     ¦   ¦   ¦   ¦   ¦   ¦   ¦'train_loss': loss,})
'''


class SomeObj(object):
    def __init__(self, model):
        self.model = model
        self.wb = wandb.init(project='you_project_name')
        self.wb.watch(self.model, log='all')

    def log_value(self, key, val):
        self.wb.log({key: val})
