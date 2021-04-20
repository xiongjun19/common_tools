# coding=utf8

"""
给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
示例 1：

输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
示例 2：

输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
"""

import numpy as np


class Solution(object):
    def __init__(self):
        super(Solution, self).__init__()

    def longestValidParentheses(self, s: str) -> int:
        if len(s) == 0:
            return 0
        dp = np.zeros(len(s), dtype=np.int32)
        comp_state_arr = np.zeros_like(dp, dtype=np.bool)
        begin_arr = [-1] * len(s)  # 标识每个位置作为结尾， 最长的初始子窜的开始位置， 不是合法的标为-1
        for i, ch in enumerate(s):
            if ch == "(":
                dp[i] = 0
                comp_state_arr[i] = False
                begin_arr[i] = -1
            else:
                if i == 0:
                    continue
                if comp_state_arr[i-1]:
                    j = begin_arr[i-1] - 1
                    if j >= 0:
                        if s[j] == "(":
                            begin = j
                            ii = j - 1
                            if ii > 0:
                                jj = begin_arr[ii]
                                if jj >= 0:
                                    begin = jj
                            comp_state_arr[i] = True
                            dp[i] = i - begin + 1
                            begin_arr[i] = begin
                else:
                    j = i - 1
                    if j >= 0:
                        if s[j] == "(":
                            begin = j
                            ii = j - 1
                            if ii > 0:
                                jj = begin_arr[ii]
                                if jj >= 0:
                                    begin = jj
                            comp_state_arr[i] = True
                            dp[i] = i - begin + 1
                            begin_arr[i] = begin
        return self._get_max(dp)

    def _get_max(self, dp):
        res = int(max(dp))
        return max(res, 0)
