# coding=utf8


import time
import torch
from torch import nn
from functools import partial


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, 'timer has already been started'
        # get_accelerator().synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer is not started'
        # get_accelerator().synchronize()
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + '-time', value, iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                    torch.distributed.get_world_size() - 1):
                print(string, flush=True)
        else:
            print(string, flush=True)


class SimpleNet(nn.Module):
    def __init__(self, dim=10):
        super().__init__()
        self.weight = nn.Linear(dim, 1)
        self.hid2 = nn.Linear(dim * 2, dim)
        self.hid = nn.Linear(dim, 2 * dim)

    def forward(self, x):
        y = self.hid(x)
        y = torch.nn.functional.gelu(y)
        y = torch.nn.functional.gelu(self.hid2(y))
        return self.weight(y)


def pre_forward_hook(timer, name, module, *argv):
    timer(name).start()


def forward_hook(timer, name, module, *argv):
    timer(name).stop()


def sub_hook(timer, module, grad_input, grad_output):
    # import ipdb; ipdb.set_trace()
    # print("module: ", module)
    # print("grad_input: ", grad_input)
    # print("grad_output: ", grad_output)
    print(module)
    timer.log(["backward"], reset=False)


def b_hook(timer, module, grad_input, grad_output):
    # import ipdb; ipdb.set_trace()
    print("module: ", module)
    timer('back_hook').stop()
    # print("grad_input: ", grad_input)
    # print("grad_output: ", grad_output)

def w_hook(name, timer, grad):
    # import ipdb; ipdb.set_trace()
    print("wegith_name: ", name)
    timer.log(["backward"], reset=False)
    # print("grad_input: ", grad_input)
    # print("grad_output: ", grad_output)




def test():
    dim = 1024 
    model = SimpleNet(dim)
    timer = Timers()
    model.register_forward_pre_hook(partial(pre_forward_hook, timer, 'forward_hook'))
    model.register_forward_hook(partial(forward_hook, timer, 'forward_hook'))
    # model.register_backward_hook(partial(b_hook, timer))
    model.weight.register_backward_hook(partial(sub_hook, timer))
    model.weight.weight.register_hook(partial(w_hook, 'out_wei', timer))
    model.hid.register_backward_hook(partial(sub_hook, timer))
    model.hid.weight.register_hook(partial(w_hook, 'hid_wei', timer))
    model.hid2.register_backward_hook(partial(sub_hook, timer))
    model.hid2.weight.register_hook(partial(w_hook, 'hid2_wei', timer))

    x = torch.randn([dim], dtype=torch.float32)
    timer('forward').start()
    y = model(x)
    timer('forward').stop()
    # loss = (1 - y) * (1 - y)
    loss = 1 - y
    # timer('back_hook').start()
    timer('backward').start()
    loss.backward()
    timer('backward').stop()
    # timer.log(['forward', 'backward', 'forward_hook', 'back_hook'])
    timer.log(['forward', 'backward', 'forward_hook'])


if __name__ == '__main__':
    test()
