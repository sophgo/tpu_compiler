#!/usr/bin/env python3
#
# Copyright (C) Cristal Vision Technologies Inc.
# All Rights Reserved.
##

import cv2
import numpy as np
import sys
import os
import copy
import math
import logging
from tqdm import tqdm
from cvi_toolkit.utils.log_setting import setup_logger

logger = setup_logger('root')


def is_all_zero(data):
    return np.all((data == 0))


class Base_Calibrator(object):
    def __init__(self,
                 image_list_file,
                 mlir_file,
                 preprocess_func,
                 input_num=200,
                 is_symmetric_quantization=True):
        logger.info("Reading {} file".format(image_list_file))
        with open(image_list_file, 'r') as fp:
            self.images = fp.readlines()

        logger.info("Images list number: {}".format(len(self.images)))
        if len(self.images) == 0:
            raise IOError("ERROR: No calibration data detect."
                          " Please check the input file: {}".format(image_list_file))
        self.input_num = int(input_num)
        if len(self.images) < self.input_num:
            logger.warning(
                "There are {} number in {}, less than input_num ({}), set input_num to {}"
                .format(len(self.images), image_list_file, self.input_num, len(self.images)))
            self.input_num = len(self.images)

        self.preprocess_func = preprocess_func
        self.model = mlir_file
        self.tensor_max = {}
        self.tensor_min = {}
        self.is_symmetric_quantization = is_symmetric_quantization

    def do_find_min_max(self):
        tensor_min_max_dict = dict()

        pbar = tqdm(self.images, total=self.input_num, position=0, leave=True)
        for img_path in pbar:
            pbar.set_description("{}".format(img_path))
            x = self.preprocess_func(img_path)
            self.model.run(x)
            all_activation_data = self.model.get_all_tensor()

            for op_name, activation in all_activation_data.items():
                if op_name not in tensor_min_max_dict:
                    tensor_min_max_dict[op_name] = (0, 0)
                min_value = np.min(activation)
                max_value = np.max(activation)
                tensor_min_max_dict[op_name] = (
                    min(tensor_min_max_dict[op_name][0], min_value),
                    max(tensor_min_max_dict[op_name][1], max_value),
                )

            # check max is zero
            for op_name, (_min, _max) in tensor_min_max_dict.items():
                if _max == 0:
                    # customer may have network output all zero, change it to 1e-5 for them.
                    logging.warning("WARNING: layer {} is all zeros. Please check the input data "
                                    "correctness.".format(op_name))
                    tensor_min_max_dict[op_name] = (_min, 1e-5)

        return tensor_min_max_dict

    def do_calibration(self):
        return self.do_find_min_max()


    def dump_threshold_table(self, threshold_table, op_threshold_dict):
        op_layer = self.model.op_info

        with open(threshold_table, 'w') as writer:
            for op_dict in op_layer:
                op_name = op_dict['name']
                threshold = op_threshold_dict[op_name][0]
                threshold_info = "{} {}\n".format(op_name, threshold)
                writer.write(threshold_info)

