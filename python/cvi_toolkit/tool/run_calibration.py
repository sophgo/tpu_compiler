#!/usr/bin/env python3
##
# Copyright (C) Cristal Vision Technologies Inc.
# All Rights Reserved.
##
import os, sys
import re
import argparse
from cvi_toolkit.utils.version import declare_toolchain_version
from cvi_toolkit.utils.log_setting import setup_logger
logger = setup_logger('root')
from cvi_toolkit.calibration.images_selector import ImageSelector
from cvi_toolkit.calibration.kld_calibrator import ActivationCalibrator, WeightCalibrator

def buffer_size_type(arg):
    try:
        val = re.match('(\d+)G', arg).groups()[0]
    except:
        raise argparse.ArgumentTypeError("must be [0-9]G")
    return val

if __name__ == '__main__':
    declare_toolchain_version()
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', metavar='model_file', help='Model file')
    parser.add_argument('--dataset', type=str, help='dataset for calibration or auto-tune')
    parser.add_argument('--input_num', type=int, required=True, default=100,
                        help='num of images for calibration')
    parser.add_argument('--image_list', help='Input image list file')
    parser.add_argument('--histogram_bin_num', type=int, default=2048,
                        help='Specify histogram bin numer for kld calculate')
    parser.add_argument('--buffer_size', type=buffer_size_type, default='2G',
                        help='buffer size for activations to speedup calibration, "G" stands for GigaByte')
    parser.add_argument('--custom_op_plugin',
                        help='set file path of custom op plugin', default='')
    parser.add_argument('-o', '--calibration_table', type=str,
                        help='output threshold table')
    args = parser.parse_args()

    buffer_size = int(args.buffer_size) * 0x40000000

    selector = ImageSelector(args.dataset, args.input_num,
                             args.image_list, debug=False)
    selector.dump('{}_images_list.txt'.format(args.calibration_table.split('.')[-1]))

    # calibration
    calibrator = ActivationCalibrator(args.model_file, selector.image_list,
                                args.histogram_bin_num, buffer_size,
                                custom_op_plugin=args.custom_op_plugin)
    calibrator.run(args.calibration_table)

    if False:
        wcalibrator = WeightCalibrator(args.model_file, args.histogram_bin_num)
        wcalibrator.run()