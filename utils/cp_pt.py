# coding=utf8

import os
import argparse
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm


def copy(src, dst):
    src_files = find_file_by_pat("*.pt", src, include_dir=False)
    src_files2 = find_file_by_pat("*.pth", src, include_dir=False)
    src_files.extend(src_files2)
    for f_path in tqdm(src_files):
        dst_path = os.path.join(dst, f_path)
        mk_folder_for_file(dst_path)
        copyfile(os.path.join(src, f_path), dst_path)


def find_file_by_pat(f_pat, _dir, include_dir=True):
    res = []
    for path in Path(_dir).rglob(f_pat):
        f_path = str(path)
        if not include_dir:
            idx = len(_dir)
            f_path = f_path[idx+1:]
        res.append(f_path)
    return res


def mk_folder_for_file(f_path):
    dir_ = os.path.dirname(f_path)
    os.makedirs(dir_, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()
    copy(args.input, args.output)
