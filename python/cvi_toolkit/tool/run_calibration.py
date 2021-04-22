#!/usr/bin/env python3
##
# Copyright (C) Cristal Vision Technologies Inc.
# All Rights Reserved.
##

import argparse
import sys
import os
import cv2
import numpy as np
from pathlib import Path
import random

from cvi_toolkit.calibration.kld_calibrator import KLD_Calibrator
from cvi_toolkit.calibration.tuner import AutoTuner, AutoTunerPlus
from cvi_toolkit import preprocess
from cvi_toolkit.data.preprocess import get_preprocess_parser

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
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', metavar='model_file', help='Model file')
    parser.add_argument('--dataset', type=str, help='dataset for calibration or auto-tune')
    parser.add_argument('--input_num', type=int, required=True, default=100, help='num of images for calibration')
    parser.add_argument('--image_list', help='Input image list file')
    parser.add_argument('--calibrator', type=str, default='KLD', help='Calibration method')
    parser.add_argument('--histogram_bin_num', type=int, default=2048,
                        help='Specify histogram bin numer for kld calculate')
    parser.add_argument('--custom_op_plugin',
                        help='set file path of custom op plugin', default='')
    parser.add_argument('-o', '--calibration_table', type=str,
                        help='output threshold table')

    parser = get_preprocess_parser(existed_parser=parser)
    args = parser.parse_args()

    preprocessor = preprocess()
    preprocessor.load_config(args.model_file, 0)
    def p_func(input_file): return preprocessor.run(input_file)

    image_list = generate_image_list(args.image_list, args.dataset, args.input_num)

    # calibration
    if args.calibrator != 'KLD':
        raise RuntimeError("Now only support kld calibrator")
    calibrator = KLD_Calibrator(image_list=image_list,
                                mlir_file=args.model_file,
                                preprocess_func=p_func,
                                histogram_bin_num=args.histogram_bin_num,
                                custom_op_plugin=args.custom_op_plugin)
    calibrator.run(args.calibration_table)