# coding=utf8


from threading import Thread
from time import sleep, ctime

time = 0


def func(add_time):
    global time
    time += add_time
    print(f"time is: {time}")


t1 = Thread(target=func, args=(5, ))
t2 = Thread(target=func, args=(3, ))
if __name__ == '__main__':
    t1.start()
    t2.start()
    t1.join()
    t2.join()
