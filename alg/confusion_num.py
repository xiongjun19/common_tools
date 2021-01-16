class Solution(object):
    def confusingNumberII(self, N):
        """
        :type N: int
        :rtype: int
        """
        dict_ = {0: 0, 1: 1, 6: 9, 8: 8, 9: 6}
        res = 0
        max_bit = self._get_max_bit(N)
        for i in range(1, max_bit + 1):
            res += self._find_count(N, i, dict_)
        return res

    def _get_max_bit(self, N):
        num = 0
        x = N
        while x > 0:
            num += 1
            x = x // 10
        return num

    def _find_count(self, max_num, num, digit_dict):
        res = [0]
        tmp_arr = []
        self._back_track(0, max_num, num, tmp_arr, digit_dict, res)
        return res[0]

    def _back_track(self, level, max_num, num, tmp_path, digit_dict, res):
        if level >= num:
            if self._is_valid_num(max_num, tmp_path, digit_dict):
                res[0] += 1
            return
        no_want = None
        if level == 0:
            no_want = 0
        for key in digit_dict.keys():
            if key == no_want:
                continue
            tmp_path.append(key)
            self._back_track(level + 1, max_num, num, tmp_path, digit_dict, res)
            tmp_path.pop()

    def _is_valid_num(self, max_num, tmp_path, digit_dict):
        cf_state = self._is_confusion(tmp_path, digit_dict)
        if not cf_state:
            return False
        candi = self._cvt_digit(tmp_path)
        if candi <= max_num:
            return True
        return False

    def _cvt_digit(self, tmp_path):
        res = 0
        for d in tmp_path:
            res = res * 10 + d
        return res

    def _is_confusion(self, tmp_path, digit_dict):
        begin = 0
        end = len(tmp_path) - 1
        while begin <= end:
            if tmp_path[begin] != digit_dict[tmp_path[end]]:
                return True
            begin += 1
            end -= 1
        return False


if __name__ == "__main__":
    obj = Solution()
    t_res = obj.confusingNumberII(1000000000)
    print(t_res)