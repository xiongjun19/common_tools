# coding=utf8

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style('whitegrid')


def vis(_dic, x_key, y_key, f_path='bar_img.jpg'):
    df = pd.DataFrame.from_dict(_dic)
    g = sns.barplot(x=x_key, y=y_key, data=df)
    for i, row in df.iterrows():
        g.text(row.name, row[y_key], row[y_key])
    plt.save_fig(f_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='bar_img.jpg')
    args = parser.parse_args()

