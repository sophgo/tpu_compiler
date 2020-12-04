#!/usr/bin/env python3
##
## Copyright (C) Cristal Vision Technologies Inc.
## All Rights Reserved.
##

import cv2
import numpy as np
import sys, os, copy, math
import logging
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


class Base_Calibrator(object):
    def __init__(self, 
            image_list_file, 
            mlir_model, 
            preprocess_func, 
            input_num=200):

        with open(image_list_file,'r') as fp:
            self.all_lines = fp.readlines()

        if len(self.all_lines) == 0:
            print("ERROR: No calibration data detect."
                  " Please check the input file: {}".format(image_list_file))
            exit(-1)

        self.input_num = int(input_num)
        self.preprocess_func = preprocess_func

        self.module = mlir_model
        self.data_min = {}

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

    def do_calibration(self):
        self.data_min, data_max = self.do_find_min_max()

        thresholds = {}
        for item in data_max:
            thresholds[item] = [data_max[item]]

        return thresholds

    def get_raw_min(self):
        return self.data_min

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
