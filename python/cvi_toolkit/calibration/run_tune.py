#!/usr/bin/env python3
##
## Copyright (C) Cristal Vision Technologies Inc.
## All Rights Reserved.
##

import argparse
import sys
import os
import cv2
import numpy as np
from pathlib import Path
import random
from cvi_toolkit.calibration.tuner import AutoTuner, AutoTunerPlus
from cvi_toolkit import preprocess


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', metavar='model_file', help='Model file')
    parser.add_argument('--custom_op_plugin', help='set file path of custom op plugin')
    parser.add_argument('--dataset', required=True, help='dataset for auto-tune')
    parser.add_argument('--input_num', type=int, default=10, help='number of images for auto-tune')
    parser.add_argument('--calibration_table', required=True, help='input calibration table for auto-tune')
    parser.add_argument('--mix_precision_table', help='input mix precision table for auto-tune')
    parser.add_argument('--tuned_table', type=str, required=True, help='output tuned calibration table file')
    parser.add_argument('--tune_iteration', type=int, default=30,
                        help='''iteration for tool to find better threhold,
                                The larger the value, the longer it will take''')
    parser.add_argument('--threshold_update_factor', default=0.01,
                        help='threshold update factor')
    parser.add_argument('--strategy', choices=['improve-per-op', 'improve-outputs'],
                        default='improve-outputs')
    parser.add_argument('--evaluation_method', choices=['cosine', 'euclid'], default='cosine',
                        help='evaluation method, cosine is "cosine similarity",'
                             'euclid is "eculidean distance')

    args = parser.parse_args()
    preprocessor = preprocess()
    preprocessor.load_config(args.model_file, 0)
    def p_func(input_file): return preprocessor.run(input_file)

    image_list = random_select_images(args.dataset, args.input_num)
    kargs = {
        'model_file': args.model_file,
        'input_calib_table': args.calibration_table,
        'mix_precision_table': args.mix_precision_table,
        'output_tune_table': args.tuned_table,
        'tune_image_list': image_list,
        'tune_image_num': args.input_num,
        'tune_iteration': args.tune_iteration,
        'preprocess_func': p_func,
        'threshold_update_factor': args.threshold_update_factor,
        'evaluation_method': args.evaluation_method
    }

    if args.strategy == 'improve-per-op':
        tuner = AutoTuner(**kargs)
    else: # 'improve-outputs'
        tuner = AutoTunerPlus(**kargs)
    tuner.run()
