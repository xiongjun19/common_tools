# coding=utf8

"""
本文件主要封住用来执行命令行的工具
call is blocking:

call('notepad.exe')
print('hello')  # only executed when notepad is closed
Popen is non-blocking:

Popen('notepad.exe')
print('hello')  # immediately executed
"""

import subprocess
import shlex
from subprocess import Popen


def exe_cmd(cmd_line):
    # to execute command async
    cmd_args = shlex.split(cmd_line)
    Popen(cmd_args)

def exe_cmd_sync(cmd_line):
    # call is blocking:
    cmd_args = shlex.split(cmd_line)
    subprocess.call(cmd_args)


def exe_cmd_raw(cmd_line):
    """
    :param
    """
    Popen(cmd_line, shell=True)


def get_cmd_args(cmd_line):
    return shlex.split(cmd_line)


def exe_c_args(args):
    return subprocess.call(args)
