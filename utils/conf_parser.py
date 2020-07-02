# coding=utf8

import os
import json
import collections


def parse_conf(f_path):
    with open(f_path) as in_:
        cont = in_.read()
        obj_ = json.loads(cont)
        return obj_

