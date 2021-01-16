# coding=utf8


class NumberTree(object):
    def __init__(self):
        pass

    def get_number(self, n):
        if n == 0 or n == 1:
            return 1
        res = 0
        for i in range(n):
            res += (self.get_number(i) * self.get_number(n-i-1))
        return res


if __name__ == "__main__":
    t_n = 4
    t_obj = NumberTree()
    print(t_obj.get_number(t_n))
