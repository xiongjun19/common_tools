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
import math


def main(input_path, output_path, is_vis, tp_no_overlap):
    if output_path is None:
        output_path = input_path[:input_path.index('.xlsx')] + "_res.xlsx"
    jobs, items = config_parser.get_job_from_excel(input_path)
    res = compute_res(jobs, items, tp_no_overlap)
    save_res(res, output_path, is_vis)


def compute_res(jobs, items, tp_no_overlap):
    res = []
    for job, item in zip(jobs, items):
        new_item = _compute_impl(job, item, tp_no_overlap)
        res.append(new_item)
    return res


def _calc_comp_and_tp_info(job, no_overlap):
    chunk_layers, vit_stages = _get_num_chunks(job)
    peak_flops = job.sys_cfg.peak_FLOPS
    comp_util = job.sys_cfg.pure_comp_util
    bs = job.train_cfg.micro_bs
    dp = job.train_cfg.dp
    tp = job.train_cfg.tp
    pp = job.train_cfg.pp
    cp = job.train_cfg.cp
    moe = job.train_cfg.moe
    experts = job.model_cfg.experts
    selected_experts = job.model_cfg.selected_experts
    one_flop_info = _calc_one_layer_flops(job, cp, tp, moe, selected_experts)
    tf_tb_one_chunk = chunk_layers * one_flop_info['tot'] / (peak_flops * 1e12 * comp_util)
    glob_bs = job.train_cfg.glob_bs
    m = glob_bs / (dp * bs)
    comp_time = m * vit_stages * tf_tb_one_chunk
    buble_time = (pp - 1) * tf_tb_one_chunk
    tp_msg, tp_times = _calc_one_layer_tp(job, cp, moe,
                                          experts, selected_experts)
    tp_times = tp_times * m * vit_stages * chunk_layers
    tp = job.train_cfg.tp
    tp_bwd = job.sys_cfg.intra_bandwidth
    if tp > job.sys_cfg.local_scale:
        tp_bwd = job.sys_cfg.inter_bandwidth
    if no_overlap >= 1.0:
        tp_comm_time = tp_times * tp_msg * 2 * (tp - 1) / (tp_bwd * tp) / (1024 ** 3)
    else:
        tp_comm_time = tp_times * tp_msg * 2 * (tp - 1) / (tp_bwd * tp) / (1024 ** 3) / tp * 2
    moe_msg, moe_times = _calc_one_layer_moe(job, cp, moe, experts)
    moe_times = moe_times * m * vit_stages * chunk_layers
    moe_comm_time = moe_times * moe_msg / tp_bwd / (1024 ** 3)
    return comp_time, buble_time, tp_comm_time, moe_comm_time


def _get_num_chunks(job):
    is_inter = job.train_cfg.pp_interleaved
    tot_layers = job.model_cfg.layers
    cfg_chunk = job.train_cfg.pp_chunk_layers
    pp = job.train_cfg.pp
    chunk = tot_layers // pp
    if chunk * pp != tot_layers:
        raise ValueError(f"tot_layers: {tot_layers} can not be divided by pp: {pp}")
    if not is_inter:
        return chunk, 1
    res_chunk = min(cfg_chunk, chunk)
    vit_stages = chunk // res_chunk
    if vit_stages * res_chunk != chunk:
        raise ValueError(f"layers {chunk} must be divided by chunks{res_chunk}")
    return min(cfg_chunk, chunk), vit_stages


def _get_recomp_cfg(job):
    return job.train_cfg.recomp


def _calc_one_layer_tp(job, cp, moe, experts, selected_exp):
    bs = job.train_cfg.micro_bs
    seq = job.train_cfg.seq_len
    h = job.model_cfg.hid_dim
    dtype = 2
    msg_size = dtype * bs * seq * h / cp
    times = 2
    if moe is not None and experts is not None:
        times = 2 + 2 * selected_exp
    else:
        times = 2 + 2
    return msg_size, times


def _calc_one_layer_moe(job, cp, moe, experts):
    if moe is None or experts is None:
        return 0, 0
    if moe == 1:
        return 0, 0
    bs = job.train_cfg.micro_bs
    seq = job.train_cfg.seq_len
    h = job.model_cfg.hid_dim
    dtype = 2
    msg_size = dtype * bs * seq * h / cp
    times = 4
    return msg_size, times


def _calc_one_layer_flops(job, cp, tp, moe, select_experts):
    bs = job.train_cfg.micro_bs
    seq = job.train_cfg.seq_len
    seq = seq / cp
    h = job.model_cfg.hid_dim
    ffn_dim = job.model_cfg.ffn_dim
    recomp_str = _get_recomp_cfg(job)
    fwd_att_ln_flops = 4 * bs * seq * (h ** 2) * 2 / tp
    cp_blocks = math.ceil((cp + 1) / 2)
    fwd_att_self_att = 2 * bs * (seq ** 2) * h * 2 * cp_blocks / tp
    cp_comm_size = 2 * bs * seq * h / tp * (cp_blocks - 1)  # 注意这里没有考虑后向的
    fwd_att_flops = fwd_att_ln_flops + fwd_att_self_att
    bwd_att_flops = 2 * fwd_att_flops
    fwd_mlp_flops = 2 * bs * seq * h * ffn_dim * 2 / tp * select_experts
    bwd_mlp_flops = 2 * fwd_mlp_flops
    if recomp_str == 'recomp':
        bwd_att_flops += fwd_att_flops
        bwd_mlp_flops += fwd_mlp_flops
    elif recomp_str == 'selective':
        bwd_att_flops += 2 * bs * (seq ** 2) * h * 2 / tp
    tot = fwd_att_flops + bwd_att_flops + fwd_mlp_flops + bwd_mlp_flops
    return {
        'tot': tot, 'fwd_att': fwd_att_flops,
        'bwd_att': bwd_att_flops, 'fwd_mlp': fwd_mlp_flops,
        'bwd_mlp': bwd_mlp_flops, 'fwd_att_linear': fwd_att_ln_flops,
        'fwd_att_self_att': fwd_att_self_att,
        'fwd_cp_comm_bytes': cp_comm_size,
    }


def _compute_impl(job, item, no_overlap):
    new_item = item.copy()
    comp_time, buble_time, tp_comm_time, moe_comm_time = _calc_comp_and_tp_info(job, no_overlap)
    new_item['tp_comm_time'] = tp_comm_time
    new_item['moe_comm_time'] = moe_comm_time
    pp_time = _calc_pp_info(job)
    new_item['pp_comm_time'] = pp_time
    dp_time = _calc_dp_info(job)
    new_item['dp_comm_time'] = dp_time

    new_item['comp_time'] = comp_time
    new_item['buble_time'] = buble_time
    new_item['tot_time'] = comp_time + buble_time + pp_time + \
                           dp_time + tp_comm_time + moe_comm_time
    new_item['tot_gpu_num'] = item['TP'] * item['PP'] * item['DP'] * job.train_cfg.cp
    return new_item


def _calc_pp_info(job):
    bs = job.train_cfg.micro_bs
    seq = job.train_cfg.seq_len
    h = job.model_cfg.hid_dim
    pp = job.train_cfg.pp
    dtype = 2
    msg_size = dtype * bs * seq * h
    tp_bwd = job.sys_cfg.inter_bandwidth
    pp_comm_time = 2 * (pp - 1) * msg_size / tp_bwd / (1024 ** 3)
    return pp_comm_time


def _calc_dp_info(job):
    dtype = 4
    voc = job.model_cfg.voc
    h = job.model_cfg.hid_dim
    dp = job.train_cfg.dp
    dp_weight = max(dtype * (voc * h), dtype * (h * h))
    pp = job.train_cfg.pp
    tp = job.train_cfg.tp
    _data = dp_weight * dtype / (pp * tp)
    comm = 2 * _data * (dp - 1) / dp
    dp_time = comm / job.sys_cfg.inter_bandwidth / (1024 ** 3)
    return dp_time


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
            df["MOE"].astype(str) + "_" + \
            df["topo"].astype(str)
    # seq_len_arr = [2048, 4096, 8192, 32768]
    seq_len_arr = set(df['seq_len'].tolist())
    # global_bs_arr = [16, 32, 128, 64, 512, 2048, 1344, 2688]
    global_bs_arr = set(df['global_batch_size'].tolist())
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
    plt.gca().set_xlabel('parallel strategy:TP_PP_DP_EP')
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
            y=["comp_time", "buble_time", "tp_comm_time", "moe_comm_time","pp_comm_time", "dp_comm_time"])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(rotation=45)
    plt.savefig(out_path, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='the input file')
    parser.add_argument('-o', '--output', type=str, help='the output_file')
    parser.add_argument('-v', '--visual_out', type=str, help='the output_file')
    parser.add_argument('--tp_no_overlap', type=float, default=1., help='partition that can not be overlapped in tp')
    parser.add_argument('--is_vis', action='store_true', default=False, help="weather visulization")
    args = parser.parse_args()
    main(args.input, args.output, args.is_vis, args.tp_no_overlap)
    '''
    python compute_llm_train.py -i dats/gpt3_train_175B.xlsx -o dats/gpt_175B_res.xlsx --is_vis
    '''
