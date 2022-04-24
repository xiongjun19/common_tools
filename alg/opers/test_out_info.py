# coding=utf8


import json
import argparse

def inspect(in_file):
    with open(in_file) as in_:
        dict_ = json.load(in_)
        for key, val in dict_.items():
            out_dict = val.get('outputs', dict())
            if len(out_dict) > 1:
                print(key)
                print(val)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    args = parser.parse_args()
    inspect(args.input)
