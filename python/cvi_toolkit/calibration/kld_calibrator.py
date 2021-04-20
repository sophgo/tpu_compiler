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
                 image_list,
                 mlir_file,
                 preprocess_func,
                 histogram_bin_num=2048,
                 math_lib_path='calibration_math.so',
                 custom_op_plugin='',
                 buffer_size=0x40000000):
        super().__init__(image_list, mlir_file, preprocess_func,
                         custom_op_plugin=custom_op_plugin)
        if not self.is_symmetric_quantization:
            raise RuntimeError(
                "KLD_Calibrator only support symmetric quantization")

        self.histogram_bin_num = int(histogram_bin_num)
        self.calibration_math = CDLL(math_lib_path)
        self.calibration_math.kl_diversity.restype = c_float
        self.calibration_math.kl_diversity_hist.restype = c_float
        self.buffered_tensors = dict()
        self.free_buffer_size = buffer_size
        logger.info("images: {}".format(self.images))

    def __activations_size(self, tensors):
        size = 0
        for k, v in tensors.items():
            size += v.size
        return size

    def __is_npz(self, image):
        return True if image.split('.')[-1] == 'npz' else False

    def __activations_generator(self):
        for image in self.images:
            if image not in self.buffered_tensors:
                if self.__is_npz(image):
                    x = np.load(image)
                    for k,v in x.items():
                        self.model.set_tensor(k, v)
                        self.model.invoke()
                else:
                    x = self.preprocess_func(image)
                    self.model.run(x)

                activations = self.model.get_all_tensor()
                size = self.__activations_size(activations)
                if size <= self.free_buffer_size:
                    self.buffered_tensors[image] = activations
                    self.free_buffer_size -= size
            else:
                activations = self.buffered_tensors[image]
            yield image, activations

    def find_min_max(self):
        tensor_min_max_dict = dict()

        pbar = tqdm(self.images, total=self.input_num, position=0, leave=True)
        for image, activations in self.__activations_generator():
            pbar.set_description("Find Min Max *{}".format(self.free_buffer_size))
            pbar.update(1)

            for op_name, activation in activations.items():
                if op_name not in tensor_min_max_dict:
                    tensor_min_max_dict[op_name] = (0, 0)
                min_value = np.min(activation)
                max_value = np.max(activation)
                minimum, maximum = tensor_min_max_dict[op_name]
                tensor_min_max_dict[op_name] = (min(minimum, min_value),
                                                max(maximum, max_value))

            # check max is zero
            for op_name, (_min, _max) in tensor_min_max_dict.items():
                if _max == 0:
                    # if network outputs are all zero, change it to 1e-5 for them.
                    logging.warning("WARNING: layer {} is all zeros. Please check the input data "
                                    "correctness.".format(op_name))
                    tensor_min_max_dict[op_name] = (_min, 1e-5)
        pbar.close()
        return tensor_min_max_dict

    def do_histogram(self, data_max):
        data_hist = {}
        width_hist = {}
        logger.info("calculate histogram, histogram number: {}".format(
            self.histogram_bin_num))
        pbar = tqdm(self.images, total=self.input_num, position=0, leave=True)

        for image, activations in self.__activations_generator():
            img_name = image.split("/")[-1]
            pbar.set_description("histogram: {}".format(img_name))
            pbar.update(1)

            for op_name, activation in activations.items():
                t = np.abs(activation.flatten())
                t = t[t != 0]

                width = data_max[op_name] / (self.histogram_bin_num - 1)
                if t.size > 0:
                    hist, _ = np.histogram(np.floor(t / width + 0.5),
                                           bins=self.histogram_bin_num,
                                           range=(0, self.histogram_bin_num-1),
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

    def __find_threshold_by_kld(self, data, width):
        return self.calibration_math.kl_diversity_hist(
            data.ctypes.data_as(POINTER(c_int)), c_float(width),
            c_longlong(self.histogram_bin_num))

    def run(self, output_calibration_table):
        # step 1: find min max
        activations_min_max_map = self.find_min_max()

        # In symmetric, find max(abs(min_value), abs(max_value))
        abs_value = {}
        for k, v in activations_min_max_map.items():
            abs_value[k] = max(abs(v[0]), abs(v[1]))

        # step 2: get histogram
        data_hist, width_hist = self.do_histogram(abs_value)

        # step3 calculate kld
        thresholds = {}
        for item in data_hist:
            thresholds[item] = self.__find_threshold_by_kld(data_hist[item], width_hist[item])

        # step4 dump to calibration table
        op_layer = self.model.op_info
        with open(output_calibration_table, 'w') as writer:
            top_format = "###\n# file: {}\n# genetated time: {}\n"
            top_format += "# histogram number: {}\n# sample number: {}\n###\n"
            cali_info = top_format.format(output_calibration_table,
                                          datetime.datetime.now(),
                                          self.histogram_bin_num,
                                          self.input_num)
            writer.write(cali_info)

            calibration_format = "# op_name    threshold    min    max\n"
            writer.write(calibration_format)
            for op_dict in op_layer:
                op_name = op_dict['name']
                threshold = thresholds[op_name]
                min_value = activations_min_max_map[op_name][0]
                max_value = activations_min_max_map[op_name][1]
                threshold_info = "{} {:.5f} {:.5f} {:.5f}\n".format(
                    op_name, threshold, min_value, max_value)
                writer.write(threshold_info)