# coding=utf8

import argparse
import pandas as pd
import openpyxl
from openpyxl import Workbook


def main(in_file, comm_file, res_file):
    input_items = _read_in_file(in_file)
    comm_items = _read_in_file(comm_file)
    res = _merge_info(input_items, comm_items)
    save_res(res, res_file)
    pass


def _read_in_file(in_file):
    df = pd.read_csv(in_file)
    _dict = df.to_dict('records')
    return _dict


def _merge_info(input_items, comm_items):
    res = []
    for topo in range(1, 4):
        for item in input_items:
            new_item = item.copy()
            tp_time = _get_tp_lat(new_item, comm_items, topo)
            pp_time = _get_pp_lat(new_item, comm_items, topo)
            dp_time = _get_pp_lat(new_item, comm_items, topo)
            overall_time = tp_time + pp_time + dp_time + new_item['tot_time']
            new_item['overall_time'] = overall_time
            new_item['tp_comm_time'] = tp_time
            new_item['pp_comm_time'] = pp_time
            new_item['dp_comm_time'] = dp_time
            new_item['Topo_NO'] = topo
            res.append(new_item)
    return res


def _query_lat(new_item, comm_items, topo, parallel):
    prefix = parallel.lower()
    times_key = prefix + "_" + "times"
    size_key = prefix + "_" + "comm_size"
    for comm_item in comm_items:
        if comm_item['TOPO'] == topo and comm_item['FLAG'] == parallel:
            if comm_item['GPU'] == new_item[parallel] \
                    and comm_item['SIZE (GB)'] == new_item[size_key]:
                curr_time = comm_item["LATENCY (us)"] / 1000
                res_time = curr_time * new_item[times_key]
                return res_time
    return None


def _get_tp_lat(new_item, comm_items, topo):
    tp_lat = _query_lat(new_item, comm_items, topo, 'TP')
    return tp_lat


def _get_dp_lat(new_item, comm_items, topo):
    tp_lat = _query_lat(new_item, comm_items, topo, 'DP')
    return tp_lat


def _get_pp_lat(new_item, comm_items, topo):
    tp_lat = _query_lat(new_item, comm_items, topo, 'PP')
    return tp_lat


def save_res(res, output_path):
    df = pd.DataFrame.from_records(res)
    # df.T.to_excel(output_path)
    df.to_excel(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='the input file')
    parser.add_argument('-c', '--comm_file', type=str, help='the input file')
    parser.add_argument('-o', '--output', type=str, help='the output_file')
    args = parser.parse_args()
    main(args.input, args.comm_file, args.output)

