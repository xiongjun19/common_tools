# coding=utf8


import re
import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def comp_plot(_dict, f_path, key_pattern=None):
    sns.set()
    for k, _arr in _dict.items():
        if _is_need(key_pattern, k):
            x = range(len(_arr))
            plt.plot(x, _arr, label=k)
    plt.legend(loc='upper left')
    plt.savefig(f_path)


def _is_need(key_pattern, key):
    if key_pattern is None:
        return True
    return key_pattern in key


def main(args):
    fig_path = args.fig
    dict_path = args.input
    key_pat = args.key
    with open(dict_path) as in_:
        _dict = json.load(in_)
        comp_plot(_dict, fig_path, key_pat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',  type=str, help='the path of the dict') 
    parser.add_argument('--fig', type=str, default='fig.png', help='where to store your plotted figure')
    parser.add_argument('--key', type=str, default=None, help='where to store your plotted figure')
    args = parser.parse_args()
    main(args)


