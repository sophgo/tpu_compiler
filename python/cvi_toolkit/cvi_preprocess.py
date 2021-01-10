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
    parser.add_argument('-o', '--output_npz',  help="output npz file", required=True)
    parser.add_argument('--npz_name', dest='output_npz')
    parser.add_argument('--resize_only_npz', default=None, help="only output resized image data to npz")
    parser.add_argument(
        '--input_name', help='input data name, default: input', default='input')
    parser.add_argument(
        '--output_data_format', help='output data format, default: hwc, chw', default='hwc')
    parser.add_argument("--batch_size", type=int, default=1,
                        help="preprocess store batch size")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    preprocessor = preprocess()
    if args.resize_only_npz:
        preprocessor.config(net_input_dims=args.net_input_dims,
                            resize_dims=args.resize_dims,
                            keep_aspect_ratio=args.keep_aspect_ratio,
                            pixel_format=args.pixel_format,
                            data_format=args.data_format,
                            gray=args.gray)
        x = preprocessor.run(args.image_file, batch=args.batch_size)
        np.savez(args.resize_only_npz, **{(args.input_name + "_resize"): x})

    preprocessor.config(**vars(args))
    x = preprocessor.run(args.image_file, batch=args.batch_size)
    np.savez(args.output_npz, **{args.input_name: x})
