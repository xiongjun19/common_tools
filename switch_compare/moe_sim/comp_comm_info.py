# coding=utf8

import argparse
import pandas as pd
import openpyxl
from openpyxl import Workbook


def main(input_path, output_path):
    if output_path is None:
        output_path = input_path.replace(".xlsx", "") + "_comm_info.csv"
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
        item['global_batch_size'], item['micro_batch_size'],
        item['seq_len'], item['hidden_dim'],
        item['layers'], item['MOE'], item['TP'], item['PP'], item['DP'],
        item['act_intra_bandwidth']
    )
    new_item['tp_comm_size'] = tp_comm_size
    new_item['tp_times'] = tp_times
    new_item['tp_operator'] = tp_operator

    moe_comm_size, moe_times, moe_operator = _calc_moe_info(
        item['global_batch_size'], item['micro_batch_size'],
        item['seq_len'], item['hidden_dim'],
        item['moe_layers'], item['TP'],
        item['PP'], item['DP'])
    new_item['moe_comm_size'] = moe_comm_size
    new_item['moe_times'] = moe_times
    new_item['moe_operator'] = moe_operator

    pp_comm_size, pp_times, pp_operator = _calc_pp_info(
        item['global_batch_size'], item['micro_batch_size'],
        item['MOE'], item['PP'], item['DP'],
        item['seq_len'], item['hidden_dim'])
    new_item['pp_comm_size'] = pp_comm_size
    new_item['pp_times'] = pp_times
    new_item['pp_operator'] = pp_operator

    dp_comm_size, dp_times, dp_operator = _calc_dp_info(
        item['hidden_dim'], item['ffn_dim'],
        item['layers'],
        item['moe_layers'],
        item['TP'],
        item['PP'])
    new_item['dp_comm_size'] = dp_comm_size
    new_item['dp_times'] = dp_times
    new_item['dp_operator'] = dp_operator

    comp_time, buble_time = _calc_comp_and_buble(
        item['global_batch_size'], item['micro_batch_size'],
        item['MOE'], item['TP'], item['PP'], item['DP'],
        item['seq_len'], item['hidden_dim'], item['ffn_dim'],
        item['layers'], item['gpu_flops'], item['gpu_util'])
    new_item['comp_time'] = comp_time
    new_item['buble_time'] = buble_time
    new_item['tot_time'] = comp_time + buble_time
    new_item['tot_gpu_num'] = item['TP'] * item['PP'] * item['DP']
    return new_item


def _calc_tp_info(bs, micro_bs, seq, h_dim, tot_layers, moe,
                  tp, pp, dp, act_intra_band):
    # 注意act_intra_band 指的是一个节点内带宽，通常是pcie 或者nvswitch 带宽
    # 带宽单位这里统一为 GB/s, 注意是Byte
    num_layers = tot_layers / pp
    act_bs = bs / dp
    m = act_bs / micro_bs
    comm_size, times = calc_tp_comm_v2(micro_bs, seq, h_dim, num_layers)
    # converted to ms by mutiply 1000
    return comm_size, times * m, "AllReduce"


def calc_layer_v2(bs, seq_len, h_dim):
    tot_comm = bs * seq_len * h_dim * 2
    # 将Bytes 转成GBytes
    tot_comm = tot_comm / (1024 ** 3)
    times = 2
    return tot_comm, times


def calc_tp_comm_v2(bs, seq, h_dim, num_layers):
    tot_comm, times = calc_layer_v2(bs, seq, h_dim)
    #  最后一层需要和embedding 做通信，因为量比较小；也不属于tp，
    #  所以下面的实现会去掉这个
    # tot_comm = num_layers * tot_comm + (bs * seq * h_dim * 2) / (1024 ** 3)
    tot_times = num_layers * times
    return tot_comm, tot_times


def _calc_moe_info(bs, micro_bs, seq_len, hid_dim,
                   moe_layers, tp,
                   pp, dp):
    act_bs = bs / dp
    act_layers = (moe_layers) / pp
    comm_size = micro_bs * seq_len * hid_dim * 2
    comm_size  = comm_size / (1024 **3)
    times = act_bs / micro_bs * act_layers * 2
    return comm_size, times, 'AllToAll'


def _calc_pp_info(tot_batch, micro_batch, moe, pp, dp,
                  seq, h_dim):
    _data = micro_batch * seq * h_dim * 2 / (1024 ** 3)
    times = 2
    return _data, times, "AllGather"


def _calc_dp_info(hid_dim, mid_dim,
                  tot_layers, moe_layers,
                  tp, pp):
    tot_moe_num = moe_layers
    moe_act_layers = tot_layers - tot_moe_num
    moe_act_layers = moe_act_layers / pp
    act_layers = tot_layers / pp
    params = _calc_t5_parm_imp(tp, act_layers, moe_act_layers, hid_dim, mid_dim, False)
    _data = 2 * params / (1024 ** 3)
    return _data, 1, "AllReduce"


def _calc_t5_parm_imp(tp, att_layers, moe_res_layers, hid_dim, ffn_mid, is_dec):
    att_num = 1
    if is_dec:
        att_num = 2
    att_param = att_num * 4 * (hid_dim ** 2) * att_layers
    ffn_param = 3 * hid_dim * ffn_mid * moe_res_layers
    res = att_param + ffn_param
    return res / tp


def _calc_comp_and_buble(tot_batch, micro_batch, moe, tp,  pp, dp, seq,
                         h_dim, mid_dim, tot_layers,
                         gpu_flops, util_ratio):
    act_batch = tot_batch // dp
    m = act_batch // micro_batch
    act_layers = tot_layers // pp
    compute_flops = calc_micro_comp_flops(micro_batch, tp, seq, h_dim, mid_dim, act_layers)
    comp_time = compute_flops / (gpu_flops * 1e12 * util_ratio) * 1e3
    tot_comp_time = m * comp_time
    bub_time = (pp - 1) * comp_time
    return tot_comp_time, bub_time


def calc_micro_comp_flops(micro_batch, tp,  seq, h_dim, mid_dim, act_layers):
    att = 4 * h_dim * h_dim * seq * micro_batch * 2 + \
             + micro_batch * h_dim * seq * seq * 2
    ffn = 3 * h_dim * mid_dim * seq * micro_batch * 2
    res = 3 * (att / tp + ffn)
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

