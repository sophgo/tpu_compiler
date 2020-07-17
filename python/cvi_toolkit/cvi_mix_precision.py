#!/usr/bin/env python3
import csv
import argparse
import os
import numpy as np
import time

from cvi_toolkit.data.preprocess import get_preprocess_parser, preprocess
from cvi_toolkit.utils.mlir_shell import gen_bf16_mlir
from cvi_toolkit.mix_precision.MixPrecision import MixPrecisior

import logging


from cvi_toolkit.utils.log_setting import setup_logger
logger = setup_logger('root')
log_flag = logger.level <= logging.DEBUG

def parse_args():
    parser = argparse.ArgumentParser(description="Generate bf16 table")
    parser.add_argument('fp32_cali_mlir_file', metavar='fp32_cali_mlir_file', help='Model file')
    parser.add_argument('image_list_file', metavar='image_list_file', help='Input image list file')
    parser.add_argument('output_bf16_table', metavar='output_bf16_layer_file', help='Output bf16 layer table')
    parser.add_argument('--output_mlir', metavar='output mix precision mlir', help='output mix precision mlir')
    parser.add_argument('--model_name', metavar='model_name', help='Model name', default='generic')
    parser.add_argument('--number_bf16', metavar='number of swich int8 to bf16', help='number of bf16 layer', type=int, default=10)
    parser.add_argument('--input_num', metavar='input_num', help='Calibration data number', type=int, default=10)
    parser = get_preprocess_parser(existed_parser=parser)
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = parse_args()

    if (args.model_name == 'generic'):
        preprocessor = preprocess()
        preprocessor.config(net_input_dims=args.net_input_dims,
                    resize_dims=args.image_resize_dims,
                    mean=args.mean,
                    mean_file=args.mean_file,
                    input_scale=args.input_scale,
                    raw_scale=args.raw_scale,
                    std=args.std,
                    rgb_order=args.model_channel_order)
        # read with opencv, bgr, hwc
        p_func = lambda input_tensor: preprocessor.run(input_tensor, input_channel_order="bgr", input_data_format="hwc",
                        output_channel_order=args.model_channel_order, input_type='tensor')
    else:
        assert(False)


    mix_precisior = MixPrecisior(args.fp32_cali_mlir_file, args.image_list_file, precrocess_func=p_func, input_num=args.input_num)


    sort_bf16_layers = mix_precisior.run()
    for idx, layer in enumerate(sort_bf16_layers):
        print("No.{:<4}: Layer: {:<30} SQNR: {}".format(idx, layer[0], layer[1]))


    with open(args.output_bf16_table, "w") as f:
        sort_bf16_layers = sort_bf16_layers[:args.number_bf16]
        for i in sort_bf16_layers:
            f.write("{}\n".format(i[0]))
    print("Output bf16 table to {}".format(args.output_bf16_table))

    if args.output_mlir:
        gen_bf16_mlir(args.fp32_cali_mlir_file , args.output_mlir, args.output_bf16_table)
        print("gen bf16 mix precision mlir => {}".format(args.output_bf16_table))
