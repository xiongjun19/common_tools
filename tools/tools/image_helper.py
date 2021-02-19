# coding=utf8

import base64
import argparse


def img2str(img_file):
    with open(img_file, "rb") as in_:
        str_ = base64.b64encode(in_.read())
        return str_


def str2img(str_, img_file):
    img_data = base64.b64decode(str_)
    with open(img_file, "wb") as out_:
        out_.write(img_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/swish_act.png")
    args = parser.parse_args()
    test_img = args.input
    test_str = img2str(test_img)
    print("the str of img:")
    print(test_str)