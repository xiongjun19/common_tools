# coding=utf8

'''
本文件用来寻找最长的重复出现N次的连续的sub sequence；N次出现的子序列必须是相互连接的
输入：sequence, 重复出现的次数;
输出： 该子序列, 该子序列的起始位置， 结束位置
'''
import json
import argparse


def parse_seq(f_path):
    with open(f_path) as in_:
        dict_ = json.load(in_)
        elems =[0] * len(dict_)
        for key in  dict_.keys():
            key_name, key_num = key.split('_')
            key_num = int(key_num)
            elems[key_num] = key_name
        return elems


def find_repeat(seq_arr, repeat_num):
    m_len = len(seq_arr) // repeat_num
    if m_len < 1:
        return None
    res = {} # begin idx is the key, and val is the length
    for i in range(len(seq_arr) - repeat_num):
        _find_repeat_from_begin(seq_arr, repeat_num, m_len, i, res)
    return res


def _find_repeat_from_begin(seq_arr, repeat_num, m_len, i, res):
    for cur_len in range(1, m_len+1):
        if _is_rep_pat(seq_arr, repeat_num, cur_len, i):
            res[i] = cur_len


def _is_rep_pat(seq_arr, repeat_num, cur_len, i):
    final_point = repeat_num * cur_len + i
    if final_point > len(seq_arr):
        return False
    sub_arr = seq_arr[i:i+cur_len]
    for j in range(1, repeat_num):
        new_start = j * cur_len + i
        new_sub = seq_arr[new_start:new_start+cur_len]
        if not _is_equal(sub_arr, new_sub):
            return False
    if len(seq_arr) >= final_point + cur_len:
        if _is_equal(sub_arr, seq_arr[final_point:final_point+cur_len]):
            return False
    return True


def _is_equal(arr, arr2):
    for x1, x2  in zip(arr, arr2):
        if x1 != x2:
            return False
    return True


def main(args):
    f_path = args.input
    num = args.num
    seqs = parse_seq(f_path)
    print(seqs[:50])
    res = find_repeat(seqs, num)
    print(res)
    print(seqs[44: 44+192])
    print(seqs[45: 45+192])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-n', '--num', type=int, default=48)
    args = parser.parse_args()
    main(args)


