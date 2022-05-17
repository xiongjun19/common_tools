# coding=utf8

from functools import partial, wraps


def attach_wrapper(obj, func=None):
    # import ipdb; ipdb.set_trace()
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func


def print_str(_str):
    print("in the inner func: ", _str)


def print_extra(extra_str=None):
    str_ = "extra info" if extra_str is None else extra_str
    print("EXTRA: ", str_)


if __name__ == '__main__':
    # import ipdb; ipdb.set_trace()
    new_func1 = attach_wrapper(print_str)
    new_func1(print_extra)("a test")
    new_func2 = attach_wrapper(print_str, print_extra)
    new_func2("a test 2")
    print_str('hello wrapper')
