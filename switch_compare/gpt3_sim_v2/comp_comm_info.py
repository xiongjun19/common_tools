# coding=utf8

import argparse
import pandas as pd
import openpyxl
from openpyxl import Workbook


def main(input_path, output_path):
    if output_path is None:
        output_path = input_path.strip(".xlsx") + "_comm_info.csv"
    items = _read_input(input_path)
    res = compute_res(items)
    save_res(res, output_path)


def _read_input(input_path):
    df = pd.read_excel(input_path)
    res = df.to_dict('records')
    return res


def compute_res(items):
    res = []
    for item in items:
        new_item = _compute_impl(item)
        res.append(new_item)
    return res


def _compute_impl(item):
    new_item = item.copy()
    tp_comm_size, tp_times, tp_operator = _calc_tp_info(
        item['global_batch_size'], item['seq_len'], item['hidden_dim'],
        item['tot_layers'], item['TP'], item['PP'], item['DP'],
        item['act_intra_bandwidth']
    )
    new_item['tp_comm_size'] = tp_comm_size
    new_item['tp_times'] = tp_times
    new_item['tp_operator'] = tp_operator
    pp_comm_size, pp_times, pp_operator = _calc_pp_info(
        item['global_batch_size'], item['micro_batch_size'],
        item['TP'], item['PP'], item['DP'],
        item['seq_len'], item['hidden_dim'], item['tot_layers'],
        item['act_inter_bandwidth'])
    new_item['pp_comm_size'] = pp_comm_size
    new_item['pp_times'] = pp_times
    new_item['pp_operator'] = pp_operator
    dp_comm_size, dp_times, dp_operator = _calc_dp_info(
        item['weight'], item['TP'], item['DP'],
        item['PP'], item['act_dp_bandwidth'])
    new_item['dp_comm_size'] = dp_comm_size
    new_item['dp_times'] = dp_times
    new_item['dp_operator'] = dp_operator

    comp_time, buble_time = _calc_comp_and_buble(
        item['global_batch_size'], item['micro_batch_size'],
        item['TP'], item['PP'], item['DP'],
        item['seq_len'], item['hidden_dim'], item['tot_layers'],
        item['gpu_flops'], item['gpu_util'])
    new_item['comp_time'] = comp_time
    new_item['buble_time'] = buble_time
    # new_item['tot_time'] = comp_time + buble_time + pp_time + \
    #                        dp_time + tp_comm_time
    new_item['tot_time'] = comp_time + buble_time
    new_item['tot_gpu_num'] = item['TP'] * item['PP'] * item['DP']
    return new_item


def calc_tp_comm_size(bs, seq, h_dim, tot_layers, tp, pp, dp):
    num_layers = tot_layers / pp
    act_bs = bs / dp
    comm_size, times = calc_tp_comm_v2(act_bs, seq, h_dim, num_layers)
    return comm_size, times


def calc_layer_v2(bs, seq_len, h_dim):
    tot_comm = bs * seq_len * h_dim * 2
    # 将Bytes 转成GBytes
    tot_comm = tot_comm / (1024 ** 3)
    times = 4
    return tot_comm, times


def calc_tp_comm_v2(bs, seq, h_dim, num_layers):
    tot_comm, times = calc_layer_v2(bs, seq, h_dim)
    #  最后一层需要和embedding 做通信，因为量比较小；也不属于tp，所以下面的实现会去掉这个
    # tot_comm = num_layers * tot_comm + (bs * seq * h_dim * 2) / (1024 ** 3)
    tot_comm = tot_comm
    tot_times = num_layers * times
    return tot_comm, tot_times


def _calc_tp_info(bs, seq, h_dim, tot_layers, tp, pp, dp, act_intra_band):
    # 注意act_intra_band 指的是一个节点内带宽，通常是pcie 或者nvswitch 带宽
    # 带宽单位这里统一为 GB/s, 注意是Byte
    comm_size, tot_times = calc_tp_comm_size(bs, seq, h_dim, tot_layers, tp, pp, dp)
    # converted to ms by mutiply 1000
    return comm_size, tot_times, "AllReduce"


def _calc_pp_info(tot_batch, micro_batch, tp, pp, dp,
                  seq, h_dim, tot_layers, band_width):
    _data = micro_batch * seq * h_dim * 2 / (1024 ** 3)
    times = 2
    return _data, times, "AllGather"


def _calc_dp_info(weight, tp, dp, pp, band_width):
    _data = weight * 2 / (tp * pp)
    comm = _data
    time = 1
    return comm, time, "AllReduce"


def _calc_comp_and_buble(tot_batch, micro_batch, tp, pp, dp, seq,
                         h_dim, tot_layers, gpu_flops, util_ratio):
    act_batch = tot_batch // dp
    m = act_batch // micro_batch
    act_layers = tot_layers // pp
    compute_flops = calc_micro_comp_flops(micro_batch, seq, h_dim, act_layers)
    act_flops = compute_flops / tp
    comp_time = act_flops / (gpu_flops * 1e12 * util_ratio) * 1e3
    tot_comp_time = m * comp_time
    bub_time = (pp - 1) * comp_time
    return tot_comp_time, bub_time


def calc_micro_comp_flops(micro_batch, seq, h_dim, act_layers):
    res = 96 * micro_batch * seq * act_layers * h_dim * h_dim * (1 + seq / (6 * h_dim))
    return res


def save_res(res, output_path):
    df = pd.DataFrame.from_records(res)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='the input file')
    parser.add_argument('-o', '--output', type=str, help='the output_file')
    args = parser.parse_args()
    main(args.input, args.output)

