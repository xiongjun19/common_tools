# coding=utf8


class Divider(object):
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        par_dict, rank_dict, ratio_dict = self._init_set(equations)
        for i, val in enumerate(values):
            x, y = equations[i]
            self.merge(x, y, val, par_dict, rank_dict, ratio_dict)
        res = [0] * len(queries)
        for i, q in enumerate(queries):
            x, y = q
            res[i] = self._calc_query(x, y, par_dict, ratio_dict)
        return res

    def _calc_query(self, x, y, par_dict, ratio_dict):
        fa_x = self.find_fa(x, par_dict, ratio_dict)
        fa_y = self.find_fa(y, par_dict, ratio_dict)
        if not fa_x or not fa_y:
            return -1
        if fa_x != fa_y:
            return -1
        return ratio_dict[y] / ratio_dict[x]

    def _init_set(self, eq_arr):
        """
        构造初始状态的并查集
        """
        par_dict = {}
        rank_dict = {}
        ratio_dict = {}  # father / child 的值, key 是当前的元素
        for eq in eq_arr:
            for x in eq:
                par_dict[x] = x
                rank_dict[x] = 0
                ratio_dict[x] = 1
        return par_dict, rank_dict, ratio_dict

    def find_fa(self, x, par_dict, ratio_dict):
        if x not in par_dict:
            return None
        fa = par_dict[x]
        if fa == x:
            return x
        par_dict[x] = self.find_fa(fa, par_dict, ratio_dict)
        ratio_dict[x] *= ratio_dict[fa]
        return par_dict[x]

    def merge(self, x, y, val, par_dict, rank_dict, ratio_dict):
        fa_x = self.find_fa(x, par_dict, ratio_dict)
        fa_y = self.find_fa(y, par_dict, ratio_dict)
        if fa_x == fa_y:
            return
        if rank_dict[fa_x] >= rank_dict[fa_y]:
            par_dict[fa_y] = fa_x
            ratio_dict[fa_y] = ratio_dict[x] / ratio_dict[y] * val
            if rank_dict[fa_x] == rank_dict[fa_y]:
                rank_dict[fa_x] += 1
        else:
            par_dict[fa_x] = fa_y
            ratio_dict[fa_x] = ratio_dict[y] / ratio_dict[x] / val


if __name__ == "__main__":
    t_obj = Divider()
    t_arr = [["a", "b"], ["b", "c"]]
    t_vals = [2, 3]
    t_qs = [['a', 'a'], ['c', 'a']]
    # t_res = t_obj.calcEquation(t_arr, t_vals, t_qs)
    # print(t_res)
    t_arr = [["a", "b"], ["c", "b"], ["d", "b"], ["w", "x"], ["y", "x"], ["z", "x"], ["w", "d"]]
    t_vals = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    t_qs = [["a", "c"], ["b", "c"], ["a", "e"], ["a", "a"], ["x", "x"], ["a", "z"]]
    t_res = t_obj.calcEquation(t_arr, t_vals, t_qs)
    print(t_res)


