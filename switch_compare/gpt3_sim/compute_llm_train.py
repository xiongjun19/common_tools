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
        output_path = input_path.strip(".xlsx") + "_res.xlsx"
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
        item['global_batch_size'], item['seq_len'], item['hidden_dim'],
        item['tot_layers'], item['TP'], item['PP'], item['DP'],
        item['act_intra_bandwidth']
    )
    new_item['tp_comm_size'] = tp_comm_size
    new_item['tp_comm_time'] = tp_comm_time
    pp_comm_size, pp_time = _calc_pp_info(
        item['global_batch_size'], item['micro_batch_size'],
        item['TP'], item['PP'], item['DP'],
        item['seq_len'], item['hidden_dim'], item['tot_layers'],
        item['act_inter_bandwidth'])
    new_item['pp_comm_size'] = pp_comm_size
    new_item['pp_comm_time'] = pp_time
    dp_comm_size, dp_time = _calc_dp_info(
        item['weight'], item['TP'], item['DP'],
        item['PP'], item['act_dp_bandwidth'])
    new_item['dp_comm_size'] = dp_comm_size
    new_item['dp_comm_time'] = dp_time

    comp_time, buble_time = _calc_comp_and_buble(
        item['global_batch_size'], item['micro_batch_size'],
        item['TP'], item['PP'], item['DP'],
        item['seq_len'], item['hidden_dim'], item['tot_layers'],
        item['gpu_flops'], item['gpu_util'])
    new_item['comp_time'] = comp_time
    new_item['buble_time'] = buble_time
    new_item['tot_time'] = comp_time + buble_time + pp_time + \
                           dp_time + tp_comm_time
    new_item['tot_gpu_num'] = item['TP'] * item['PP'] * item['DP']
    return new_item


def calc_tp_comm_size(bs, seq, h_dim, tot_layers, tp, pp, dp):
    num_layers = tot_layers / pp
    act_bs = bs / dp
    comm_size = 2 * calc_tp_comm_v2(act_bs, seq, h_dim, num_layers) \
        * (tp - 1) / tp
    return comm_size


def calc_layer_v2(bs, seq_len, h_dim):
    tot_comm = 4 * bs * seq_len * h_dim * 2
    # 将Bytes 转成GBytes
    tot_comm = tot_comm / (1024 ** 3)
    return tot_comm


def calc_tp_comm_v2(bs, seq, h_dim, num_layers):
    tot_comm = calc_layer_v2(bs, seq, h_dim)
    #  最后一层需要和embedding 做通信，因为量比较小；也不属于tp，所以下面的实现会去掉这个
    # tot_comm = num_layers * tot_comm + (bs * seq * h_dim * 2) / (1024 ** 3)
    tot_comm = num_layers * tot_comm
    return tot_comm


def _calc_tp_info(bs, seq, h_dim, tot_layers, tp, pp, dp, act_intra_band):
    # 注意act_intra_band 指的是一个节点内带宽，通常是pcie 或者nvswitch 带宽
    # 带宽单位这里统一为 GB/s, 注意是Byte
    comm_size = calc_tp_comm_size(bs, seq, h_dim, tot_layers, tp, pp, dp)
    # converted to ms by mutiply 1000
    comm_time = comm_size / act_intra_band * 1000
    return comm_size, comm_time


def _calc_pp_info(tot_batch, micro_batch, tp, pp, dp,
                  seq, h_dim, tot_layers, band_width):
    _data = micro_batch * seq * h_dim * 2 / (1024 ** 3)
    time = _data / band_width * 1000
    return _data, time


def _calc_dp_info(weight, tp, dp, pp, band_width):
    _data = weight * 2 / (tp * pp)
    comm = 2 * _data * (dp - 1) / dp
    time = comm / band_width * 1e3
    return comm, time


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


def save_res(res, output_path, is_vis):
    df = pd.DataFrame.from_records(res)
    # import ipdb; ipdb.set_trace()
    if is_vis:
        _visual_res(df, output_path)
    df.T.to_excel(output_path)
    # df.to_csv(output_path, index=False)


def _visual_res(df, out_file):
    df['parallel_strategy'] = df['TP'].astype(str) + "_" + \
            df["PP"].astype(str) + "_" + df["DP"].astype(str) + "_" + \
            df["topo"].astype(str)
    seq_len_arr = [2048, 4096, 8192, 32768]
    global_bs_arr = [16, 32, 128, 64, 512, 2048, 1344, 2688]
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
    plt.gca().tick_params(axis='y', which='both', length=0)
    # lg = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    # lg.set_title('Topo_NO')
    plt.xticks(rotation=45)
    # plt.savefig(out_path)
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
            y=["comp_time", "buble_time", "tp_comm_time", "pp_comm_time", "dp_comm_time"])
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
