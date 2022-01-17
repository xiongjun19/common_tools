# coding=utf8


import json
import argparse




def parse_seq(f_path):
    with open(f_path) as in_:
        dict_ = json.load(in_)
        elems =[0] * len(dict_)
        for key, val in dict_.items():
            key_name, key_num = key.split('_')
            key_num = int(key_num)
            elems[key_num] = (key, val)
        return elems


def main(args):
    f_path = args.input
    num = args.num
    seqs = parse_seq(f_path)
    for seq in seqs[:30]:
        print("inputs is: ")
        print(seq[1]['inputs'])
        print("outputs is: ")
        print(seq[1]['outputs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-n', '--num', type=int, default=48)
    args = parser.parse_args()
    main(args)


