#!/usr/bin/env python3

import argparse
from cvi_toolkit.numpy_helper import *
npz_tool_func = {
    "compare": npz_compare,
    'dump':npz_dump,
    "extract": npz_extract,
    "rename": npz_rename,
    "to_bin": npz_to_bin,
    "bf16_to_fp32": npz_bf16_to_fp32,
}

def main():
    args_list = sys.argv
    if len(args_list) < 2:
        print("Usage: {} {} ".format(args_list[0], npz_tool_func.keys()))
        exit(-1)

    def NoneAndRaise(func):
        raise RuntimeError("No support {} Method".format(func))

    npz_tool_func.get(args_list[1], lambda x: NoneAndRaise(args_list[1]))(args_list[2:])


if __name__ == "__main__":
    main()