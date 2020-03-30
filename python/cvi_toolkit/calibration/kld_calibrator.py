#!/usr/bin/env python3
##
## Copyright (C) Cristal Vision Technologies Inc.
## All Rights Reserved.
##

import cv2
import numpy as np
import sys, os, copy, math
import pymlir
from ctypes import *

def is_all_zero(data):
    for num in data:
        if num != 0:
            return False
    return True

def warn_zeros(layer_name):
    print("WARNING: layer {} is all zeros. Please check the input data "
          "correctness.".format(layer_name))
    exit(-1)

class KLD_Calibrator(object):
    def __init__(self, args, preprocess_func):
        with open(args.image_list_file,'r') as fp:
            self.all_lines = fp.readlines()

        if len(self.all_lines) == 0:
            print("ERROR: No calibration data detect."
                  " Please check the input file: {}".format(args.image_list_file))
            exit(-1)

        self.input_num = int(args.input_num)
        self.preprocess_func = preprocess_func

        self.module = pymlir.module()
        self.module.load(args.model_file)
        self.histogram_bin_num = int(args.histogram_bin_num)

        self.calibration_math = CDLL(args.math_lib_path)
        self.calibration_math.kl_diversity.restype = c_float
        self.calibration_math.kl_diversity_hist.restype = c_float
        self.args = args

    def KLD_hist(self, data, width):
        return self.calibration_math.kl_diversity_hist(
            data.ctypes.data_as(POINTER(c_int)), c_float(width),
            c_longlong(self.histogram_bin_num))

    def do_find_max(self):
        data_max = {}
        idx = 0
        for line in self.all_lines:
            print('Calculating max at iteration: ', str(idx))

            x = self.preprocess_func(line, self.args)
            _ = self.module.run(x)
            data = self.module.get_all_tensor()

            for item in data:
                if item not in data_max:
                    data_max[item] = 0

                t = np.abs(data[item].flatten())
                t = t[t!=0]

                if t.size > 0:
                    if is_all_zero(t):
                        warn_zeros(item)
                    data_max[item] = max(data_max[item], np.max(t))

            idx += 1
            if idx >= self.input_num:
                break

        return data_max

    def do_histogram(self, data_max):
        data_hist = {}
        width_hist = {}
        idx = 0
        for line in self.all_lines:
            print('Generating histogram at iteration: ', str(idx))

            x = self.preprocess_func(line, self.args)
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
        data_max = self.do_find_max()
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


class KLD_Calibrator_v2(object):
    def __init__(self, image_list_file, preprocess_func, model,
            input_num=1, histogram_bin_num=2048, auto_tune=True, tune_iteration=10
            , math_lib_path="calibration_math.so"):
        with open(image_list_file,'r') as fp:
            self.all_lines = fp.readlines()

        if len(self.all_lines) == 0:
            print("ERROR: No calibration data detect."
                  " Please check the input file: {}".format(image_list_file))
            exit(-1)

        self.input_num = input_num
        self.preprocess_func = preprocess_func

        self.model = model
        self.histogram_bin_num = histogram_bin_num

        self.calibration_math = CDLL(math_lib_path)
        self.calibration_math.kl_diversity.restype = c_float
        self.calibration_math.kl_diversity_hist.restype = c_float

    def KLD_hist(self, data, width):
        return self.calibration_math.kl_diversity_hist(
            data.ctypes.data_as(POINTER(c_int)), c_float(width),
            c_longlong(self.histogram_bin_num))

    def do_find_max(self):
        data_max = {}
        idx = 0
        for line in self.all_lines:
            print('Calculating max at iteration: ', str(idx))

            x = self.preprocess_func(line.rstrip())
            self.model.inference(x)
            data = self.model.get_all_tensor()
            print(line)
            for item in data:
                if item not in data_max:
                    data_max[item] = 0

                t = np.abs(data[item].flatten())
                t = t[t!=0]

                if t.size > 0:
                    if is_all_zero(t):
                        warn_zeros(item)
                    data_max[item] = max(data_max[item], np.max(t))

            idx += 1
            if idx >= self.input_num:
                break

        return data_max

    def do_histogram(self, data_max):
        data_hist = {}
        width_hist = {}
        idx = 0
        for line in self.all_lines:
            print('Generating histogram at iteration: ', str(idx))

            x = self.preprocess_func(line)
            _ = self.model.inference(x)
            data = self.model.get_all_tensor()

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

    def do_calibration(self, threshold_table=None):
        data_max = self.do_find_max()
        data_hist, width_hist = self.do_histogram(data_max)

        thresholds = {}
        for item in data_hist:
            thresholds[item] = [self.KLD_hist(data_hist[item], width_hist[item])]

        if threshold_table:
            op_layer = self.model.get_op_info()
            with open(threshold_table, 'w') as outfile:
                for op_dict in op_layer:
                    line = op_dict['name']
                    for num in thresholds[op_dict['name']]:
                        line += ' ' + str(num)
                    outfile.write(line)
                    outfile.write('\n')
