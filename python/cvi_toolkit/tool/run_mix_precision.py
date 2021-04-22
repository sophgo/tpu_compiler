#!/usr/bin/env python3
import csv
import argparse
import os, math
import numpy as np
import time
import logging
from pathlib import Path
import random

from cvi_toolkit.data.preprocess import get_preprocess_parser, preprocess
from cvi_toolkit.calibration.MixPrecision import MixPrecSearcher

def random_select_images(dataset_path, num):
    full_list = []
    for file in Path(dataset_path).glob('**/*'):
        if file.is_file():
            full_list.append(str(file))
    random.shuffle(full_list)
    num = num if len(full_list) > num else len(full_list)
    if num == 0:
        num = len(full_list)
    return full_list[:num]

def generate_image_list(image_list_file, dataset_path, image_num):
    image_list = []
    if image_list_file:
        with open(image_list_file, 'r') as f:
            for line in f.readlines():
                image_list.append(line.strip())
    else:
        if not dataset_path:
            raise RuntimeError("Please specific dataset path by --dataset")
        image_list = random_select_images(dataset_path, image_num)
    return image_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate bf16 table")
    parser.add_argument('model_file', help='fp32 mlir file')
    parser.add_argument('--model_name', default='generic', help='Model name')
    parser.add_argument('--dataset', help='dataset path for mix precision searching')
    parser.add_argument('--input_num', type=int, default=10, help='num of images for mix precision searching')
    parser.add_argument("--image_list", help="specify a file with images's absolute path for mix precision searching")
    parser.add_argument('--calibration_table', required=True, help='calibration table generated by calibration or tune tool')
    parser.add_argument('--max_bf16_layers', type=int, default=10, help='number of bf16 layers for mix precision quantization')
    parser.add_argument('--loss_table', default='full_loss_table.txt',
                        help="output all loss of layers if each layer is quantized to bf16")
    parser.add_argument('-o', '--mix_precision_table', required=True,
                        help='output searched bf16 layer table')

    parser = get_preprocess_parser(existed_parser=parser)
    args = parser.parse_args()

    preprocessor = preprocess()
    preprocessor.load_config(args.model_file, 0)

    image_list = generate_image_list(args.image_list, args.dataset, args.input_num)

    p_func = lambda input_tensor: preprocessor.run(input_tensor)

    searcher = MixPrecSearcher(args.model_file,
                               args.calibration_table,
                               image_list,
                               precrocess_func=p_func,
                               input_num=args.input_num)

    sort_bf16_layers = searcher.run()

    with open(args.loss_table, "w") as f:
        for idx, layer in enumerate(sort_bf16_layers):
            loss_msg = "No.{:<4}: Layer: {:<50}\t\tLoss: {}".format(idx, layer[0], layer[1])
            f.write("{}\n".format(loss_msg))
            print(loss_msg)

    with open(args.mix_precision_table, "w") as f:
        sort_bf16_layers = sort_bf16_layers[:args.max_bf16_layers]
        for i in sort_bf16_layers:
            f.write("{}\n".format(i[0]))
    print("Output bf16 table to {}".format(args.mix_precision_table))
