# coding=utf8


import argparse
import pandas as pd
import openpyxl
from openpyxl import Workbook
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def main(input_path, output_path, is_vis):
    if output_path is None:
        output_path = input_path.replace(".xlsx", "") + "_res.xlsx"
    items = _read_input(input_path)
    res = compute_res(items)
    save_res(res, output_path, is_vis)


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
    tp_comm_size, tp_comm_time = _calc_tp_info(
        item['global_batch_size'], item['micro_batch_size'],
        item['seq_len'], item['hidden_dim'],
        item['layers'], item['MOE'], item['TP'], item['PP'], item['DP'],
        item['act_tp_bandwidth']
    )
    new_item['tp_comm_size'] = tp_comm_size
    new_item['tp_comm_time'] = tp_comm_time

    moe_comm_size, moe_comm_time = _calc_moe_info(
        item['global_batch_size'], item['micro_batch_size'],
        item['seq_len'], item['hidden_dim'],
        item['moe_layers'], item['TP'], item['PP'], item['DP'],
        item['act_intra_bandwidth'])
    new_item['moe_comm_size'] = moe_comm_size
    new_item['moe_comm_time'] = moe_comm_time

    pp_comm_size, pp_time = _calc_pp_info(
        item['global_batch_size'], item['micro_batch_size'],
        item['MOE'], item['TP'], item['PP'], item['DP'],
        item['seq_len'], item['hidden_dim'], item['layers'],
        item['act_inter_bandwidth'])
    new_item['pp_comm_size'] = pp_comm_size
    new_item['pp_comm_time'] = pp_time

    dp_comm_size, dp_time = _calc_dp_info(
        item['hidden_dim'], item['ffn_dim'],
        item['layers'],
        item['moe_layers'],
        item['TP'], item['PP'], item['DP'],
        item['act_intra_bandwidth'])
    new_item['dp_comm_size'] = dp_comm_size
    new_item['dp_comm_time'] = dp_time

    comp_time, buble_time = _calc_comp_and_buble(
        item['global_batch_size'], item['micro_batch_size'],
        item['MOE'], item['TP'], item['PP'], item['DP'],
        item['seq_len'], item['hidden_dim'], item['ffn_dim'],
        item['layers'], item['gpu_flops'], item['gpu_util'])
    new_item['comp_time'] = comp_time
    new_item['buble_time'] = buble_time

    new_item['tot_time'] = comp_time + buble_time + pp_time + \
                           dp_time + tp_comm_time + moe_comm_time
    new_item['tot_gpu_num'] = item['TP'] * item['PP'] * item['DP']
    return new_item


def _calc_tp_info(bs, micro_bs, seq, h_dim, tot_layers,
                  moe, tp, pp, dp, act_intra_band):
    # 注意act_intra_band 指的是一个节点内带宽，通常是pcie 或者nvswitch 带宽
    # 带宽单位这里统一为 GB/s, 注意是Byte
    num_layers = tot_layers / pp
    act_bs = bs / dp
    m = act_bs / micro_bs
    comm_size, times = calc_tp_comm_v2(micro_bs, seq, h_dim, num_layers)
    # converted to ms by mutiply 1000
    comm_time = 2 * (tp - 1) * comm_size / (act_intra_band * tp) * 1000 * times * m
    return comm_size * times * m, comm_time


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
                   pp, dp, bandwidth):
    act_bs = bs / dp
    act_layers = (moe_layers) / pp
    comm_size = micro_bs * seq_len * hid_dim * 2
    comm_size = comm_size / (1024 ** 3)
    times = act_bs / micro_bs * act_layers * 2
    comm_time = times * comm_size / bandwidth * 1000
    return comm_size, comm_time


def _calc_pp_info(tot_batch, micro_batch, moe, tp, pp, dp,
                  seq, h_dim, tot_layers, band_width):
    _data = micro_batch * seq * h_dim * 2 / (1024 ** 3)
    time = 2 * _data / band_width * 1000
    return _data, time


def _calc_dp_info(hid_dim, mid_dim,
                  tot_layers, moe_layers,
                  tp, pp, dp, band_width):
    tot_moe_num = moe_layers
    moe_act_layers = tot_layers - tot_moe_num
    moe_act_layers = moe_act_layers / pp
    act_layers = tot_layers / pp
    params = _calc_t5_parm_imp(tp, act_layers, moe_act_layers, hid_dim, mid_dim, False)
    _data = 2 * params / (1024 ** 3)
    comm = 2 * _data * (dp - 1) / dp
    time = comm / band_width * 1e3
    return comm, time


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


def save_res(res, output_path, is_vis):
    df = pd.DataFrame.from_records(res)
    if is_vis:
        _visual_res(df, output_path)
    df.T.to_excel(output_path)
    # df.to_csv(output_path, index=False)


def _visual_res(df, out_file):
    df['parallel_strategy'] = df['TP'].astype(str) + "_" + \
            df["PP"].astype(str) + "_" + df["DP"].astype(str) + "_" + \
            df["topo"].astype(str)
    seq_len_arr = [2048, 32768]
    global_bs_arr = [16, 32, 64, 128, 2048, 2688, 5376]
    for glob_bs in global_bs_arr:
        for seq_len in seq_len_arr:
            out_path = out_file + f"seq_len-{seq_len}_glob_bs-{glob_bs}.png"
            vis_impl(df, seq_len, glob_bs, out_path)
            out_path = out_file + f"seq_len-{seq_len}_glob_bs-{glob_bs}_stacked.png"
            vis_stack_bar(df, seq_len, glob_bs, out_path)


def vis_impl(df, seq_length, batch_size, out_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    title = f'seq_length: {seq_length}, glob_batch_size: {batch_size}'
    filter_df = df[df.seq_len == seq_length]
    filter_df = filter_df[filter_df.global_batch_size == batch_size]
    if len(filter_df) <= 0:
        return
    filter_df['normalized_ratio'] = filter_df['tot_time'] / (filter_df['tot_time'].iloc[0])
    filter_df['norm_performance'] = 1 / filter_df['normalized_ratio']
    barplot = sns.barplot(data=filter_df, x="parallel_strategy", y="norm_performance")
    for i, p in enumerate(barplot.patches):
        bb = "%.0f%%" % (filter_df['norm_performance'].iloc[i] * 100)
        barplot.annotate(bb,
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 3),
                         textcoords='offset points')

    barplot.set_title(title)
    plt.gca().set_xlabel('parallel strategy')
    plt.gca().set_ylabel('norm performance')
    plt.xticks(rotation=45)
    plt.gca().tick_params(axis='y', which='both', length=0)
    # lg = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    # lg.set_title('Topo_NO')
    plt.savefig(out_path, bbox_inches='tight')
    plt.clf()


def vis_stack_bar(df, seq_length, batch_size, out_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    filter_df = df[df.seq_len == seq_length]
    filter_df = filter_df[filter_df.global_batch_size == batch_size]
    if len(filter_df) <= 0:
        return
    filter_df.set_index('parallel_strategy').plot(kind='bar', stacked=True,
            y=["comp_time", "buble_time", "tp_comm_time", "pp_comm_time", "dp_comm_time", "moe_comm_time"])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(rotation=45)
    plt.savefig(out_path, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='the input file')
    parser.add_argument('-o', '--output', type=str, help='the output_file')
    parser.add_argument('-v', '--visual_out', type=str, help='the output_file')
    parser.add_argument('--is_vis', action='store_true', default=False, help="weather visulization")
    args = parser.parse_args()
    main(args.input, args.output, args.is_vis)
    '''
    python compute_llm_train.py -i dats/gpt3_train_175B.xlsx -o dats/gpt_175B_res.xlsx --is_vis
    '''
