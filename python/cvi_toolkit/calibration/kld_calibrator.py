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
from base_calibrator import Base_Calibrator
from ctypes import *

from cvi_toolkit.utils.log_setting import setup_logger

logger = setup_logger('root')

class KLD_Calibrator(Base_Calibrator):
    def __init__(self,
            image_list_file,
            mlir_model,
            preprocess_func,
            input_num=200,
            histogram_bin_num=2048,
            math_lib_path='calibration_math.so'):
        super().__init__(image_list_file, mlir_model, preprocess_func, input_num)

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
        idx = 0
        for line in self.all_lines:
            # print('Generating histogram at iteration: ', str(idx))
            x = self.preprocess_func(line)
            _ = self.module.run(x)
            data = self.module.get_all_tensor()

            for item in data:
                t = np.abs(data[item].flatten())
                t = t[t!=0]

                width = data_max[item][0] / (self.histogram_bin_num - 1)
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
        abs_data_max = super().do_calibration()
        data_hist, width_hist = self.do_histogram(abs_data_max)

        thresholds = {}
        for item in data_hist:
            thresholds[item] = [self.KLD_hist(data_hist[item], width_hist[item])]

        return thresholds
