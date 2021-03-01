#!/usr/bin/env python3
#
# Copyright (C) Cristal Vision Technologies Inc.
# All Rights Reserved.
##

import cv2
import numpy as np
import sys
import os
import abc
import copy
import math
import logging
import pymlir

from tqdm import tqdm
from cvi_toolkit.utils.log_setting import setup_logger

logger = setup_logger('root')


def is_all_zero(data):
    return np.all((data == 0))


class Base_Calibrator(object):
    def __init__(self,
                 image_list,
                 mlir_file,
                 preprocess_func,
                 is_symmetric_quantization=True,
                 custom_op_plugin=''):
        if type(image_list) == str:
            with open(image_list, 'r') as fp:
                self.images = fp.readlines()
        else:
            self.images = image_list

        logger.info("Images list number: {}".format(len(self.images)))
        if len(self.images) == 0:
            raise IOError("ERROR: No calibration image detect.")

        self.input_num = len(self.images)

        self.preprocess_func = preprocess_func
        self.model = pymlir.module()
        self.model.load(mlir_file)
        self.model.set_plugin(custom_op_plugin)
        self.tensor_max = {}
        self.tensor_min = {}
        self.is_symmetric_quantization = is_symmetric_quantization

    def do_find_min_max(self):
        tensor_min_max_dict = dict()

        pbar = tqdm(self.images, total=self.input_num, position=0, leave=True)
        for idx, img_path in enumerate(pbar):
            pbar.set_description("Find Min Max")
            pbar.update(1)
            if idx >= self.input_num:
                break
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
        pbar.close()
        return tensor_min_max_dict

    @abc.abstractmethod
    def do_calibration(self):
        return NotImplemented

    @abc.abstractmethod
    def create_threshold_table(self):
        return NotImplemented
