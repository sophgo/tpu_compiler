#!/usr/bin/env python3
##
## Copyright (C) Cristal Vision Technologies Inc.
## All Rights Reserved.
##

import cv2
import numpy as np
import sys, os, copy, math
import pymlir
import logging
from ctypes import *

from cvi_toolkit.utils.log_setting import setup_logger

logger = setup_logger('root')


def is_all_zero(data):
    for num in data:
        if num != 0:
            return False
    return True

# customer may have network output all zero, change it to 1e-5 for them.
def warn_zeros(layer_name, t):
    print("WARNING: layer {} is all zeros. Please check the input data "
          "correctness.".format(layer_name))
    print("WARNING: Set zero value to 1e-5.")
    t[0] = 1e-5

class KLD_Calibrator(object):
    def __init__(self, 
            image_list_file, 
            mlir_model,
            preprocess_func, 
            input_num=200, 
            histogram_bin_num=2048, 
            math_lib_path='calibration_math.so'):
        with open(image_list_file,'r') as fp:
            self.all_lines = fp.readlines()

        if len(self.all_lines) == 0:
            print("ERROR: No calibration data detect."
                  " Please check the input file: {}".format(image_list_file))
            exit(-1)

        self.input_num = int(input_num)
        self.preprocess_func = preprocess_func

        self.module = mlir_model
        self.histogram_bin_num = int(histogram_bin_num)

        self.calibration_math = CDLL(math_lib_path)
        self.calibration_math.kl_diversity.restype = c_float
        self.calibration_math.kl_diversity_hist.restype = c_float
        self.data_min = {}

    def KLD_hist(self, data, width):
        return self.calibration_math.kl_diversity_hist(
            data.ctypes.data_as(POINTER(c_int)), c_float(width),
            c_longlong(self.histogram_bin_num))

    def do_find_min_max(self):
        data_max = {}
        idx = 0
        for line in self.all_lines:
            print('Calculating max at iteration: ', str(idx))

            x = self.preprocess_func(line)
            _ = self.module.run(x)
            data = self.module.get_all_tensor()

            for item in data:
                if item not in data_max:
                    data_max[item] = 0
                    self.data_min[item] = 0

                t = np.abs(data[item].flatten())

                if t.size > 0:
                    if is_all_zero(t):
                        warn_zeros(item, t)
                    data_max[item] = max(data_max[item], np.max(t))
                    self.data_min[item] = min(self.data_min[item], np.min(data[item].flatten()))

            idx += 1
            if idx >= self.input_num:
                break

        return self.data_min, data_max

    def get_raw_min(self):
        return self.data_min

    def do_histogram(self, data_max):
        data_hist = {}
        width_hist = {}
        idx = 0
        for line in self.all_lines:
            # print('Generating histogram at iteration: ', str(idx))
            x = self.preprocess_func(line)
            _ = self.module.run(x)
            data = self.module.get_all_tensor()

            for item in data:
                t = np.abs(data[item].flatten())
                t = t[t!=0]

                width = data_max[item] / (self.histogram_bin_num - 1)
                if t.size > 0:
                    hist, bins = np.histogram(np.floor(t / width + 0.5),
                                              bins=self.histogram_bin_num,
                                              range=(0, self.histogram_bin_num-1),
                                              density=False)
                else:
                    hist = np.zeros(self.histogram_bin_num)
                hist = hist.astype(np.int32)

                if item not in data_hist:
                    data_hist[item] = hist
                    width_hist[item] = width
                else:
                    data_hist[item] += hist

            idx += 1
            if idx >= self.input_num:
                break

        return data_hist, width_hist

    def do_calibration(self):
        self.data_min, data_max = self.do_find_min_max()
        data_hist, width_hist = self.do_histogram(data_max)

        thresholds = {}
        for item in data_hist:
            thresholds[item] = [self.KLD_hist(data_hist[item], width_hist[item])]

        return thresholds


    def dump_threshold_table(self, threshold_table, thresholds):
        op_layer = self.module.op_info
        with open(threshold_table, 'w') as outfile:
            for op_dict in op_layer:
                line = op_dict['name']
                for num in thresholds[op_dict['name']]:
                    line += ' ' + str(num)
                outfile.write(line)
                outfile.write('\n')

    def dump_density_table(self, density_table, low, high):
        op_layer = self.module.op_info
        with open(density_table, 'w') as outfile:
            for op_dict in op_layer:
                line = op_dict['name']
                for num in high[op_dict['name']]:
                    line += ' ' + str(low[op_dict['name']]) + ' ' + str(num)
                outfile.write(line)
                outfile.write('\n')
