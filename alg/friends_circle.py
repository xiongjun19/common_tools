# coding=utf8


class Solution(object):
    def __init__(self):
        pass

    def findCircleNum(self, M):
        num = len(M)
        fa_arr = [i for i in range(num)]
        rank = [0] * num
        for i in range(num):
            for j in range(i+1, num):
                if M[i][j] == 1:
                    self.merge(i, j, fa_arr, rank)
        s = set([self.find_fa(i, fa_arr) for i in range(num)])
        return len(s)

    def find_fa(self, i, fa_arr):
        """
        寻找根节点
        """
        if i == fa_arr[i]:
            return i
        fa_arr[i] = self.find_fa(fa_arr[i], fa_arr)
        return fa_arr[i]

    def merge(self, i, j, fa_arr, rank_arr):
        x = self.find_fa(i, fa_arr)
        y = self.find_fa(j, fa_arr)
        if x == y:
            return
        if rank_arr[x] >= rank_arr[y]:
            fa_arr[y] = x
            if rank_arr[x] == rank_arr[y]:
                rank_arr[x] += 1
        else:
            fa_arr[x] = y


if __name__ == "__main__":
    test_arr = [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 1, 1],
    ]
    t_obj = Solution()
    print(t_obj.findCircleNum(test_arr))