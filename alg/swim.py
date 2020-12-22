# coding=utf8


class Swim(object):
    def swimInWater(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        n = len(grid)
        cur_max = max(grid[0][0], grid[-1][-1]) - 1
        nums = n * n
        fa_arr = [i for i in range(nums)]
        rank_arr = [0] * nums
        while fa_arr[0] != fa_arr[nums - 1]:
            cur_max += 1
            self._cons_set(cur_max, n, fa_arr, rank_arr, grid)
            fa_arr[0] = self.find_fa(0, fa_arr)
            fa_arr[nums - 1] = self.find_fa(nums - 1, fa_arr)
        return cur_max

    def _cons_set(self, time, n, fa_arr, rank_arr, grid):
        for i in range(n):
            for j in range(n):
                if time < grid[i][j]:
                    continue
                self._cons_neigbor(time, n, fa_arr, rank_arr, grid, i, j)

    def _cons_neigbor(self, time, n, fa_arr, rank_arr, grid, i, j):
        neigbors = self._get_neigbors(i, j, n)
        cur_x = self._g_id2float_id(i, j, n)
        for ni, nj in neigbors:
            if time >= grid[ni][nj]:
                n_x = self._g_id2float_id(ni, nj, n)
                self.merge(cur_x, n_x, fa_arr, rank_arr)

    def _get_neigbors(self, i, j, n):
        res = []
        if i < n - 1:
            res.append([i+1, j])
        if j < n - 1:
            res.append([i, j+1])
        return res

    def _g_id2float_id(self, i, j, n):
        res = i * n + j
        return res

    def float_id2g_id(self, x, n):
        i = x // n
        j = x - i * n
        return i, j

    def find_fa(self, i, fa_arr):
        if fa_arr[i] == i:
            return i
        fa_arr[i] = self.find_fa(fa_arr[i], fa_arr)
        return fa_arr[i]

    def merge(self, x1, x2, fa_arr, rank_arr):
        y1 = self.find_fa(x1, fa_arr)
        y2 = self.find_fa(x2, fa_arr)
        if y1 == y2:
            return
        if rank_arr[y1] >= rank_arr[y2]:
            fa_arr[y2] = y1
            if rank_arr[y1] == rank_arr[y2]:
                rank_arr[y1] += 1
        else:
            fa_arr[y1] = y2


if __name__ == "__main__":
    t_obj = Swim()
    t_arr = [[0, 2], [1, 3]]
    t_arr = [
        [0, 1, 2, 3, 4],
        [24, 23, 22, 21, 5],
        [12, 13, 14, 15, 16],
        [11, 17, 18, 19, 20],
        [10, 9, 8, 7, 6]
    ]
    print(t_obj.swimInWater(t_arr))
