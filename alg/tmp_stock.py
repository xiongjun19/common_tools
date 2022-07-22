# coding=utf8


import sys

max_int = sys.maxsize
max_val = float("inf")


def find_max_points(arr):
    '''
     找到所有的新最高点， 并且返回新高的值及其下标
    '''
    if len(arr) < 3:
        raise ValueError(" the arr must be longer than 3")
    res = []

    pre_larger = False
    tmp_pre = max_val
    for i, val in enumerate(arr):
        if pre_larger: # 前面的是一个高点
            if val <= tmp_pre: # 目前的点小于和等于前面的高点
                res.append([val, i-1])
                pre_larger = False
                tmp_pre = val
            else: # 更高的点出现
                tmp_pre = val
        else: # 前面的点不是高点
            if val <= tmp_pre: # 目前的点比前面的点还低
                tmp_pre = val
            if val > tmp_pre: # 目前的点比前面的点高，很可能是高点
                pre_larger = True
                tmp_pre = val
    return res


def find_minimum(arr, start_idx, end_idx):
    '''
    寻找最低点
    '''
    tmp = max_val
    tmp_idx = start_idx
    for i in range(start_idx, end_idx):
        if arr[i] < tmp:
            tmp = arr[i]
            tmp_idx = i
    return tmp, tmp_idx


def compute_retreat_and_days(arr, num_days=0):
    """
    计算最大回落天数以及间隔
    """
    if num_days <= 0:
        num_days = max_int
    # 第一步计算新高
    new_high_points = find_max_points(arr)
    # 第二步
    
