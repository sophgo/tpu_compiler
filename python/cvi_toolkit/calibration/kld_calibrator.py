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
from ctypes import *
from tqdm import tqdm
import datetime

from cvi_toolkit.data.preprocess import preprocess
from cvi_toolkit.utils.mlir_parser import MlirParser
from cvi_toolkit.utils.log_setting import setup_logger
logger = setup_logger('root')

class BaseKldCalibrator:
    def __init__(self, math_lib_path='calibration_math.so'):
        self.calib_lib = CDLL(math_lib_path)
        self.calib_lib.kl_diversity.restype = c_float
        self.calib_lib.kl_diversity_hist.restype = c_float

    def histogram(self, ndarray, abs_max, bin_num):
        t = np.abs(ndarray.flatten())
        t = t[t != 0]
        width = abs_max / (bin_num - 1)

        if t.size > 0:
            hist, _ = np.histogram(np.floor(t / width + 0.5),
                                    bins=bin_num,
                                    range=(0, bin_num-1),
                                    density=False)
        else:
            hist = np.zeros(bin_num)
        hist = hist.astype(np.int32)
        return hist, width

    def kld_threshold(self, hist, width, bin_num):
        threshold = self.calib_lib.kl_diversity_hist(
                        hist.ctypes.data_as(POINTER(c_int)),
                        c_float(width),
                        c_longlong(bin_num))
        return threshold


class ActivationCalibrator(BaseKldCalibrator):
    def __init__(self, mlir_file, image_list, histogram_bin_num,
                 buffer_size, custom_op_plugin=None):
        super().__init__()
        self.images = image_list
        self.input_num = len(self.images)
        self.preprocessor = preprocess()
        self.preprocessor.load_config(mlir_file, 0)

        self.model = pymlir.module()
        if custom_op_plugin:
            self.model.set_plugin(custom_op_plugin)
        self.model.load(mlir_file)

        self.tensor_max = {}
        self.tensor_min = {}
        self.histogram_bin_num = histogram_bin_num

        self.buffered_activations = dict()
        self.free_buffer_size = buffer_size
        logger.info("Max buffer size: {} Bytes".format(buffer_size))

    def _activations_size(self, tensors):
        size = 0
        for k, v in tensors.items():
            size += v.size
        return size

    def _is_npz(self, image):
        return True if image.split('.')[-1] == 'npz' else False

    def _activations_generator(self):
        for image in self.images:
            if image not in self.buffered_activations:
                if self._is_npz(image):
                    x = np.load(image)
                    for k,v in x.items():
                        self.model.set_tensor(k, v)
                        self.model.invoke()
                else:
                    x = self.preprocessor.run(image)
                    self.model.run(x)

                activations = self.model.get_all_tensor()
                size = self._activations_size(activations)
                if size <= self.free_buffer_size:
                    self.buffered_activations[image] = activations
                    self.free_buffer_size -= size
            else:
                activations = self.buffered_activations[image]
            yield image, activations

    def find_min_max(self):
        tensor_min_max_dict = dict()

        pbar = tqdm(self.images, total=self.input_num, position=0, leave=True)
        for image, activations in self._activations_generator():
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
        logger.info("calculate histogram, histogram number: {}".format(
                    self.histogram_bin_num))

        data_hist = {}
        width_hist = {}

        pbar = tqdm(self.images, total=self.input_num, position=0, leave=True)
        for image, activations in self._activations_generator():
            img_name = image.split("/")[-1]
            pbar.set_description("histogram: {}".format(img_name))
            pbar.update(1)

            for op_name, activation in activations.items():
                hist, width = self.histogram(activation, data_max[op_name],
                                      self.histogram_bin_num)
                if op_name not in data_hist:
                    data_hist[op_name] = hist
                    width_hist[op_name] = width
                else:
                    # update histogram
                    data_hist[op_name] += hist

        return data_hist, width_hist

    def find_threshold(self, data_hist, width_hist):
        thresholds = {}
        num = len(data_hist)
        pbar = tqdm(range(num), total=num, position=0, leave=True)
        for item in data_hist:
            pbar.set_description("threshold: {}".format(item))
            pbar.update(1)
            thresholds[item] = self.kld_threshold(data_hist[item], width_hist[item],
                                                  self.histogram_bin_num)
        pbar.close()
        return thresholds

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
        thresholds = self.find_threshold(data_hist, width_hist)

        # step4 dump to calibration table
        op_layer = self.model.op_info
        with open(output_calibration_table, 'w') as f:
            top_format = "###\n# file: {}\n# genetated time: {}\n"
            top_format += "# histogram number: {}\n# sample number: {}\n###\n"
            cali_info = top_format.format(output_calibration_table,
                                          datetime.datetime.now(),
                                          self.histogram_bin_num,
                                          self.input_num)
            f.write(cali_info)

            calibration_format = "# op_name    threshold    min    max\n"
            f.write(calibration_format)
            for op_dict in op_layer:
                op_name = op_dict['name']
                threshold = thresholds[op_name]
                min_value = activations_min_max_map[op_name][0]
                max_value = activations_min_max_map[op_name][1]
                threshold_info = "{} {:.5f} {:.5f} {:.5f}\n".format(
                    op_name, threshold, min_value, max_value)
                f.write(threshold_info)


class WeightCalibrator(BaseKldCalibrator):
    def __init__(self, mlir_file, histogram_bin_num):
        super().__init__()
        self.bin_num = histogram_bin_num
        self.parser = MlirParser(mlir_file)
        weight_npz = self.parser.get_weight_file_name()
        self.weights = np.load(weight_npz)

    def min_max_vals(self, ndarray):
        min_val = np.min(ndarray)
        max_val = np.max(ndarray)
        abs_max = max(abs(min_val), abs(max_val))
        return (min_val, max_val, abs_max)

    def run(self):
        for w_name in self.weights.files:
            ndarray = self.weights[w_name]
            min_, max_, abs_max_ = self.min_max_vals(ndarray)
            hist, width = self.histogram(ndarray, abs_max_, self.bin_num)
            threshold = self.kld_threshold(hist, width, self.bin_num)
            logger.info("{}: {} {} {} {}".format(w_name, threshold, min_, max_, abs_max_))
