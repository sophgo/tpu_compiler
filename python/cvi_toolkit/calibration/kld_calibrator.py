#!/usr/bin/env python3
##
# Copyright (C) Cristal Vision Technologies Inc.
# All Rights Reserved.
##

import cv2
import numpy as np
import sys
import os
import copy
import math
import pymlir
import logging
from base_calibrator import Base_Calibrator
from ctypes import *
from tqdm import tqdm
import datetime


from cvi_toolkit.utils.log_setting import setup_logger

logger = setup_logger('root')


class KLD_Calibrator(Base_Calibrator):
    def __init__(self,
                 image_list_file,
                 mlir_file,
                 preprocess_func,
                 input_num=200,
                 histogram_bin_num=2048,
                 math_lib_path='calibration_math.so',
                 custom_op_plugin=''):
        super().__init__(image_list_file, mlir_file, preprocess_func,
                         input_num, custom_op_plugin=custom_op_plugin)
        if not self.is_symmetric_quantization:
            raise RuntimeError(
                "KLD_Calibrator only support symmetric quantization")

        self.histogram_bin_num = int(histogram_bin_num)
        self.calibration_math = CDLL(math_lib_path)
        self.calibration_math.kl_diversity.restype = c_float
        self.calibration_math.kl_diversity_hist.restype = c_float

    def KLD_hist(self, data, width):
        return self.calibration_math.kl_diversity_hist(
            data.ctypes.data_as(POINTER(c_int)), c_float(width),
            c_longlong(self.histogram_bin_num))

    def do_histogram(self, data_max):
        data_hist = {}
        width_hist = {}
        logger.info("calculate histogram, histogram number: {}".format(
            self.histogram_bin_num))
        pbar = tqdm(self.images, total=self.input_num, position=0, leave=True)
        for idx, img_path in enumerate(pbar):
            img_name = img_path.split("/")[-1]
            pbar.set_description("histogram: {}".format(img_name))
            pbar.update(1)
            if idx >= self.input_num:
                break

            x = self.preprocess_func(img_path)
            self.model.run(x)
            all_tensor = self.model.get_all_tensor()

            for op_name, activation in all_tensor.items():
                t = np.abs(activation.flatten())
                t = t[t != 0]

                width = data_max[op_name] / (self.histogram_bin_num - 1)
                if t.size > 0:
                    hist, _ = np.histogram(np.floor(t / width + 0.5),
                                           bins=self.histogram_bin_num,
                                           range=(
                        0, self.histogram_bin_num-1),
                        density=False)
                else:
                    hist = np.zeros(self.histogram_bin_num)

                hist = hist.astype(np.int32)
                if op_name not in data_hist:
                    data_hist[op_name] = hist
                    width_hist[op_name] = width
                else:
                    # update histogram
                    data_hist[op_name] += hist

        return data_hist, width_hist

    def do_calibration(self):
        op_tensor_min_max = self.do_find_min_max()

        # In symmetric, find max(abs(min_value), abs(max_value))
        abs_value = {}
        for k, v in op_tensor_min_max.items():
            abs_value[k] = max(abs(v[0]), abs(v[1]))
        data_hist, width_hist = self.do_histogram(abs_value)

        thresholds = {}
        for item in data_hist:
            thresholds[item] = self.KLD_hist(
                data_hist[item], width_hist[item])

        return thresholds

    def create_calibration_table(self, calibration_table_file):
        # step 1: find min max
        op_tensor_min_max = self.do_find_min_max()

        # In symmetric, find max(abs(min_value), abs(max_value))
        abs_value = {}
        for k, v in op_tensor_min_max.items():
            abs_value[k] = max(abs(v[0]), abs(v[1]))

        # step 2: get histogram
        data_hist, width_hist = self.do_histogram(abs_value)

        # step3 calculate kld
        thresholds = {}
        for item in data_hist:
            thresholds[item] = self.KLD_hist(
                data_hist[item], width_hist[item])

        # step4 dump to calibration table
        op_layer = self.model.op_info

        with open(calibration_table_file, 'w') as writer:

            calibration_infomation = "###\n# file: {}, genetated time {}\n# histogram number: {}\n###\n".format(
                calibration_table_file, datetime.datetime.now(), self.histogram_bin_num)
            writer.write(calibration_infomation)
            calibration_format = "# op_name    threshold    min    max"
            writer.write(calibration_format)
            for op_dict in op_layer:
                op_name = op_dict['name']
                threshold = thresholds[op_name]
                min_value = op_tensor_min_max[op_name][0]
                max_value = op_tensor_min_max[op_name][1]
                threshold_info = "{} {:.5f} {:.5f} {:.5f}\n".format(
                    op_name, threshold, min_value, max_value)
                writer.write(threshold_info)
