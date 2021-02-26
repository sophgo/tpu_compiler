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


from cvi_toolkit.calibration.base_calibrator import Base_Calibrator
from cvi_toolkit.calibration.kld_calibrator import KLD_Calibrator
from cvi_toolkit.calibration.tuner import Tuner_v2
from cvi_toolkit import preprocess
from cvi_toolkit.data.preprocess import get_preprocess_parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', metavar='model_file', help='Model file')
    parser.add_argument(
        'image_list_file', metavar='image_list_file', help='Input image list file')
    parser.add_argument(
        '--output_file', metavar='output_file', help='Output file')
    parser.add_argument(
        '--output_tune_file', metavar='output_tune_file', help='Output tune file', default="", type=str)
    parser.add_argument('--output_density_table', metavar='output_density_table',
                        help='Output density file for look-up table')
    parser.add_argument('--model_name', metavar='model_name',
                        help='Model name', default='generic')
    parser.add_argument('--calibrator', metavar='calibrator',
                        help='Calibration method', default='KLD')
    parser.add_argument('--math_lib_path', metavar='math_path',
                        help='Calibration math library path', default='calibration_math.so')
    parser.add_argument('--input_num', metavar='input_num',
                        help='Calibration data number', default=10)
    parser.add_argument('--histogram_bin_num', metavar='histogram_bin_num', help='Specify histogram bin numer for kld calculate',
                        default=2048)
    parser.add_argument('--auto_tune', action='store_true',
                        help='Enable auto tune or not')
    parser.add_argument('--tune_iteration', metavar='iteration', type=int,
                        help='The number of data using in tuning process', default=10)
    parser.add_argument('--dataset_file_path', metavar='dataset_file_path',
                        type=str, help='file path that recode dataset images path')
    parser.add_argument('--custom_op_plugin', metavar='custom_op_plugin',
                        help='set file path of custom op plugin', default='')
    parser.add_argument('--create_calibration_table', type=int,
                        default=1, help="create_calibration_table, default: 1")
    parser = get_preprocess_parser(existed_parser=parser)
    args = parser.parse_args()

    if (args.model_name == 'generic'):
        preprocessor = preprocess()
        preprocessor.config(**vars(args))
        def p_func(input_file): return preprocessor.run(input_file)
    else:
        assert(False)

    threshold_table = args.output_file if args.output_file else \
                      '{}_threshold_table'.format(args.model_name)

    if args.auto_tune != True:
        if args.calibrator == 'KLD':
            calibrator = KLD_Calibrator(image_list_file=args.image_list_file,
                                        mlir_file=args.model_file,
                                        preprocess_func=p_func,
                                        input_num=args.input_num,
                                        histogram_bin_num=args.histogram_bin_num,
                                        math_lib_path=args.math_lib_path,
                                        custom_op_plugin=args.custom_op_plugin)
            calibrator.create_calibration_table(threshold_table)
        else:
            raise RuntimeError("Now only support kld calibrator")
    else: # tune
        args.input_threshold_table = threshold_table
        tuner = Tuner_v2(args.model_file, threshold_table, args.image_list_file,
                         tune_iteration=args.tune_iteration, preprocess_func=p_func,
                         tune_table=args.output_tune_file)
        tuner.run_tune()

if __name__ == '__main__':
    main()
