# coding=utf8

"""
this file designed to clean the json file in input_dir, and output to the output_dir
usage:
    python clean_json.py -i /path_to_input_dir -o /path_to_output_dir 
"""

import os
import re

def clean(in_dir, out_dir):
    os.makedirs(out_dir)
    f_list = os.listdir(in_dir)
    arr_list = [os.path.join(in_dir, x) for x in f_list if x.endswith('.json')]
    out_list = [os.path.join(out_dir, x) for x in f_list if x.endswith('.json')]
    for in_f, out_f in zip(arr_list, out_list):
        clean_file(in_f, out_f)

def clean_file(in_f, out_f):
    res_arr = []
    with open(in_f) as in_:
        line_arr = in_.readlines()
        b, e = _get_begin_end(line_arr)
        new_arr = line_arr[b:e+1]
        for line in new_arr:
            new_line = line.replace("+\n", "").rstrip()
            res_arr.append(new_line)
    with open(out_f, "w") as out_:
        for line  in res_arr:
            out_.write(line + "\n")

def _get_begin_end(line_arr):
    b = 0
    e = len(line_arr) - 1

    for i, l in enumerate(line_arr):
        l = l.strip()
        if '[' in l:
            b = i
            break
    j = e 
    while j > 0:
        if line_arr[j].strip() == ']':
            e = j
            break
        j -= 1
    return b, e

            


def main(args):
    in_dir = args.in_dir
    out_dir = args.out_dir
    clean(in_dir, out_dir)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', help='input dir of the json')
    parser.add_argument('-o', '--out_dir', help='output dir of the clean_json')
    t_args = parser.parse_args()
    main(t_args)

