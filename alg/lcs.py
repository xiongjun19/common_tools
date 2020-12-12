# coding=utf8

import numpy as np


def find_lcs(s1, s2):
    """
    寻找最长连续公共子序列
    """
    if not s1 or not s2:
        return 0
    cur_max = 0
    l1 = len(s1)
    l2 = len(s2)
    mtx = np.zeros((len(s1), len(s2)))
    for i in range(l1):
        for j in range(l2):
            if i == j:
                mtx[i][j] = 1 if i == 0 or j == 0 else mtx[i-1][j-1] + 1
            else:
                mtx[i][j] = 0
            if mtx[i][j] > cur_max:
                cur_max = mtx[i][j]
    return cur_max


def find_not_cont_lcs(s1, s2):
    """
    寻找最长公共子序列
    """
    if not s1 or not s2:
        return 0
    cur_max = 0
    l1 = len(s1)
    l2 = len(s2)
    mtx = np.zeros((len(s1), len(s2)))
    for i in range(l1):
        for j in range(l2):
            if s1[i] == s2[j]:
                mtx[i][j] = 1 if i == 0 or j == 0 else mtx[i-1][j-1] + 1
            else:
                if i > 0 and j > 0:
                    mtx[i][j] = max(mtx[i-1][j], mtx[i][j-1])
                elif j > 0:
                    mtx[i][j] = mtx[i][j-1]
                elif i > 0:
                    mtx[i][j] = mtx[i-1][j]
            if mtx[i][j] > cur_max:
                cur_max = mtx[i][j]
    return int(cur_max)


if __name__ == "__main__":
    t1 = "abcde"
    t2 = "ace"
    res = find_not_cont_lcs(t1, t2)
    print(res)
