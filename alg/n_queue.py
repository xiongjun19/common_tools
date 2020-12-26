# coding=utf8


import copy


class NQueues(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        mat = self._init_mat(n)
        visited = set()
        abs_visited = set()
        tmp_path = []
        res = []
        self.back_track(0, visited, abs_visited, tmp_path, n, res)
        f_res = []
        for idx_arr in res:
            new_mat = self._convert_res(idx_arr, mat)
            f_res.append(new_mat)
        return f_res

    def _init_mat(self, n):
        res = [["."] * n for _ in range(n)]
        return res

    def _convert_res(self, idx_arr, mat):
        new_mat = copy.deepcopy(mat)
        for i, idx in enumerate(idx_arr):
            new_mat[i][idx] = "Q"
            new_mat[i] = "".join(new_mat[i])
        return new_mat

    def back_track(self, level, visited, abs_visited, tmp_path, n, res):
        if level >= n and len(tmp_path) == n:
            return res.append(tmp_path.copy())
        for i in range(n):
            if i not in visited:
                if not self._is_valid(tmp_path, level, i):
                    continue
                tmp_path.append(i)
                visited.add(i)
                self.back_track(level+1, visited, abs_visited, tmp_path, n, res)
                visited.remove(i)
                tmp_path.pop()

    def _is_valid(self, tmp_path, cur_level, i):
        for pre_level, j in enumerate(tmp_path):
            if abs(cur_level - pre_level) == abs(i-j):
                return False
        return True


if __name__ == "__main__":
    import pprint
    q = pprint.PrettyPrinter()
    obj = NQueues()
    res = obj.solveNQueens(4)
    q.pprint(res)
