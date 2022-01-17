# coding=utf8

import json
import argparse


LAY_BEG_IDX = 44
LAY_OPS = 192


def parse_seq(f_path):
    with open(f_path) as in_:
        dict_ = json.load(in_)
        elems =[0] * len(dict_)
        for key, val in dict_.items():
            key_name, key_num = key.split('_')
            key_num = int(key_num)
            elems[key_num] = (key, val)
        return elems


def _calc_dif(arr1, l1, arr2, l2):
    s_arr = arr1
    s_l = l1
    l_arr = arr2
    l_l = l2
    if l1 > l2:
        s_arr = arr2
        l_arr = arr1
        s_l = l2
        l_l = l1
    return _calc_dif_impl(s_arr, s_l, l_arr, l_l)


def _calc_dif_impl(s_arr, s_l, l_arr, l_l):
    l_sub_arr = _get_sub_arr(l_arr, s_l, l_l)
    res = []
    for d1, d2 in zip(s_arr, l_sub_arr):
        k1, v1 = d1[0], d1[1]
        k2, v2 = d2[0], d2[1]
        diff_dict = _find_diff(v1, v2)
        res.append([(k1, k2), diff_dict])
    return res


def _get_sub_arr(l_arr, s_l, l_l):
    _sub_arr = l_arr[:LAY_BEG_IDX + s_l * LAY_OPS]
    _sub_arr.extend(l_arr[LAY_BEG_IDX + l_l * LAY_OPS:])
    return _sub_arr


def _find_diff(d1, d2):
    res = {}
    input1 = d1['inputs'].keys()
    input2 = d2['inputs'].keys()
    if is_diff(input1, input2):
        res['input'] = [input1, input2]
    output1 = d1['outputs'].keys()
    output2 = d2['outputs'].keys()
    if is_diff(output1, output2):
        res['output'] = [output1, output2]
    return res


def is_diff(keys1, keys2):
    s1 = set(keys1)
    s2 = set(keys2)
    r_s = s1 - s2
    r_s2 = s2 - s1
    if len(r_s) == 0 and len(r_s2) == 0:
        return False
    return True


def _find_lg_last(seqs):
    elem = seqs[-2]
    info = elem[1]
    out = list(info['outputs'].keys())[0]
    out = int(out)
    res = []
    for elem in seqs:
        key, v_dict = elem
        lg_info = _get_lg_dim(v_dict, out) 
        if len(lg_info) > 0:
            res.append([key, lg_info])
    for  elem in res:
        print(elem)


def _get_lg_dim(v_dict, thres): 
    res = {}
    keys = ['inputs', 'outputs']
    for key in keys:
        _dict = v_dict[key]
        for no in _dict.keys():
            if no.isnumeric():
                no = int(no)
                if no > thres:
                    _arr = res.get(key, [])
                    _arr.append(no)
                    res[key] = _arr
    return res


"""
find_different_layer diff
"""
def _calc_diff_layers(seqs, layers):
    for i in range(layers-1):
        b = LAY_BEG_IDX + i * LAY_OPS
        e = b + LAY_OPS
        e2 = e + LAY_OPS
        sub1 = seqs[b:e]
        sub2 = seqs[e:e2]
        diff_arr = _get_diff_in_arr(sub2, sub1)
        print("***" * 50)
        print(f"difference between layers {i+1} to {i} is :")
        for elem in diff_arr:
            print(elem)
        print("***" * 50)


def _get_diff_in_arr(arr1, arr2):
    res = []
    for d1, d2 in zip(arr1, arr2):
        k1, v1 = d1[0], d1[1]
        k2, v2 = d2[0], d2[1]
        diff_dict = _find_diff(v1, v2)
        res.append([(k1, k2), diff_dict])
    return res


def main(args):
    f_path = args.input
    seqs = parse_seq(f_path)
    seqs2 = parse_seq(args.ref)
    l1 = args.f_ls
    l2 = args.f_ls2
    diff_res = _calc_dif(seqs, l1, seqs2, l2)
    for res in diff_res:
        print(res)
    print("first large than last info is: ")
    _find_lg_last(seqs)
    print("second larger than last info is: ")
    _find_lg_last(seqs2)
    _calc_diff_layers(seqs, 2)
    _calc_diff_layers(seqs2, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-r', '--ref', type=str)
    parser.add_argument('-l1', '--f_ls', type=int, help="the first layers")
    parser.add_argument('-l2', '--f_ls2', type=int, help="the second layers")
    parser.add_argument('-n', '--num', type=int, default=48)
    parser.add_argument('-o', '--output', type=str, help='save the diff result')
    args = parser.parse_args()
    main(args)

