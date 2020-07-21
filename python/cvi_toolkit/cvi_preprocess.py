#!/usr/bin/env python3
import csv
import argparse
import os
import numpy as np
import time

from cvi_toolkit.data.preprocess import get_preprocess_parser, preprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Do image preprocess")
    parser = get_preprocess_parser(existed_parser=parser)
    parser.add_argument('--image_file', help='image_file')
    parser.add_argument('--npz_name', help='output npz name, default: input_fp32.npz', default='input_fp32.npz')
    parser.add_argument(
        '--input_name', help='input data name, default: input', default='input')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    preprocessor = preprocess()
    preprocessor.config(net_input_dims=args.net_input_dims,
                        resize_dims=args.image_resize_dims,
                        mean=args.mean,
                        mean_file=args.mean_file,
                        input_scale=args.input_scale,
                        raw_scale=args.raw_scale,
                        std=args.std,
                        rgb_order=args.model_channel_order)
    preprocessor.run(args.image_file, output_npz=args.npz_name,
                     input_name=args.input_name)

