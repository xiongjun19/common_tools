# coding=utf8


from functools import wraps, partial
import logging


import ipdb; ipdb.set_trace()

def logged(func=None, level=logging.WARN, name=None, message=None):
    if func is None:
        return partial(logged, level=level, name=name, message=message)
    logname = name if name else func.__module__
    log = logging.getLogger(logname)
    log_msg = message if message else func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        log.log(level, log_msg)
        return func(*args, **kwargs)
    return wrapper


@logged
def add(x, y):
    return x + y


@logged(level=logging.CRITICAL, name='example')
def spam():
    print("Spam!")


def test():
    z = add(3, 5)
    print("add res is: ", z)
    spam()


if __name__ == '__main__':
    test()
