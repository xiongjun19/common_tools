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
    moe_comm_size, moe_times, moe_operator = _calc_moe_info(
        item['global_batch_size'], item['micro_batch_size'],
        item['seq_len'], item['hidden_dim'],
        item['enc_moe_layers'], item['dec_moe_layers'],
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
        item['enc_layers'], item['dec_layers'],
        item['enc_moe_layers'], item['dec_moe_layers'],
        item['PP'])
    new_item['dp_comm_size'] = dp_comm_size
    new_item['dp_times'] = dp_times
    new_item['dp_operator'] = dp_operator

    comp_time, buble_time = _calc_comp_and_buble(
        item['global_batch_size'], item['micro_batch_size'],
        item['MOE'], item['PP'], item['DP'],
        item['seq_len'], item['hidden_dim'], item['ffn_dim'],
        item['enc_layers'], item['dec_layers'],
        item['gpu_flops'], item['gpu_util'])
    new_item['comp_time'] = comp_time
    new_item['buble_time'] = buble_time
    new_item['tot_time'] = comp_time + buble_time
    new_item['tot_gpu_num'] = item['PP'] * item['DP']
    return new_item


def _calc_moe_info(bs, micro_bs, seq_len, hid_dim,
                   moe_enc_layers, moe_dec_layers,
                   pp, dp):
    act_bs = bs / dp
    act_layers = (moe_enc_layers + moe_dec_layers) / pp
    comm_size = micro_bs * seq_len * hid_dim * 2
    comm_size  = comm_size / (1024 **3)
    times = act_bs / micro_bs * act_layers * 2
    return comm_size, times, 'AllToAll'


def _calc_pp_info(tot_batch, micro_batch, moe, pp, dp,
                  seq, h_dim):
    _data = micro_batch * seq * h_dim * 2 / (1024 ** 3)
    times = 2
    return _data, times, "AllGather"


def _calc_dp_info(hid_dim, mid_dim, enc_num, dec_num,
                  enc_moe_num, dec_moe_num, pp):
    tot_layers = (enc_num + dec_num)
    tot_moe_num = enc_moe_num + dec_moe_num
    act_layers = tot_layers - tot_moe_num
    act_layers = act_layers / pp
    params = _calc_t5_parm_imp(act_layers, hid_dim, mid_dim, True)
    _data = 2 * params / (1024 ** 3)
    return _data, 1, "AllReduce"


def _calc_t5_parm_imp(dec_num, hid_dim, ffn_mid, is_dec):
    att_num = 1
    if is_dec:
        att_num = 2
    att_param = att_num * 4 * (hid_dim ** 2) * dec_num
    ffn_param = 3 * hid_dim * ffn_mid * dec_num
    res = att_param + ffn_param
    return res


def _calc_comp_and_buble(tot_batch, micro_batch, moe, pp, dp, seq,
                         h_dim, mid_dim, enc_num, dec_num,
                         gpu_flops, util_ratio):
    act_batch = tot_batch // dp
    m = act_batch // micro_batch
    tot_layers = enc_num + dec_num
    act_layers = tot_layers // pp
    compute_flops = calc_micro_comp_flops(micro_batch, seq, h_dim, mid_dim, act_layers)
    comp_time = compute_flops / (gpu_flops * 1e12 * util_ratio) * 1e3
    tot_comp_time = m * comp_time
    bub_time = (pp - 1) * comp_time
    return tot_comp_time, bub_time


def calc_micro_comp_flops(micro_batch, seq, h_dim, mid_dim, act_layers):
    att = 4 * h_dim * h_dim * seq * micro_batch * 2 + \
             + micro_batch * h_dim * seq * seq * 2
    ffn = 3 * h_dim * mid_dim * seq * micro_batch * 2
    res = 3 * (2 * att + ffn)
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

