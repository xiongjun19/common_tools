# coding=utf8

import argparse
import pandas as pd
import openpyxl
from openpyxl import Workbook
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import config_parser


def _calc_att_time_per_layer(bs, hid_dim, heads, head_dim, seq_len, tp, cp, FLOPS, bandwidth):
    act_seq = seq_len // cp
    act_heads = heads // tp
    cmp_flops = bs * act_seq * act_seq *  hid_dim / tp
    cmp_time =  cmp_flops / (FLOPS * 1e12) # second
    comm_bytes = 2 *  bs * act_seq * hid_dim / tp
    comm_time = comm_bytes / (1024 ** 3) / bandwidth
    print(cmp_time, comm_bytes / (1024 **2),  comm_time)
    print(f"compute divide by comm_time: {cmp_time / comm_time}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='the input file')
    parser.add_argument('-o', '--output', type=str, help='the output_file')
    parser.add_argument('-v', '--visual_out', type=str, help='the output_file')
    parser.add_argument('--tp_no_overlap', type=float, default=1., help='partition that can not be overlapped in tp')
    parser.add_argument('--is_vis', action='store_true', default=False, help="weather visulization")
    args = parser.parse_args()
    _calc_att_time_per_layer(1, 12288, 96, 128, 2048 * 16, 8, 4, 210, 50)
