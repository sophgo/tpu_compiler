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
    return np.all((data == 0))

# Just use min and max as the quantization range
class Base_Calibrator(object):
    def __init__(self,
            image_list_file,
            mlir_model,
            preprocess_func,
            input_num=200,
            is_symmetric_quantization=True):

        with open(image_list_file,'r') as fp:
            self.all_lines = fp.readlines()

        if len(self.all_lines) == 0:
            print("ERROR: No calibration data detect."
                  " Please check the input file: {}".format(image_list_file))
            exit(-1)

        self.input_num = int(input_num)
        self.preprocess_func = preprocess_func

        self.module = mlir_model
        self.tensor_max = {}
        self.tensor_min = {}
        self.is_symmetric_quantization = is_symmetric_quantization

    def do_find_min_max(self):
        idx = 0
        for line in self.all_lines:
            print('Calculating max at iteration: ', str(idx))

            x = self.preprocess_func(line)
            _ = self.module.run(x)
            data = self.module.get_all_tensor()

            for item in data:
                if item not in self.tensor_max:
                    self.tensor_max[item] = 0
                    self.tensor_min[item] = 0

                if data[item].size > 0:
                    if np.all((data[item] == 0)):
                        # customer may have network output all zero, change it to 1e-5 for them.
                        print("WARNING: layer {} is all zeros. Please check the input data "
                            "correctness.".format(item))
                        self.tensor_max[item] = max(self.tensor_max[item], 1e-5)
                    else:
                        self.tensor_max[item] = max(self.tensor_max[item], np.max(data[item]))

                    self.tensor_min[item] = min(self.tensor_min[item], np.min(data[item]))

            idx += 1
            if idx >= self.input_num:
                break

        return self.tensor_min, self.tensor_max

    def do_calibration(self):
        self.tensor_min, self.tensor_max = self.do_find_min_max()

        thresholds = {}
        if self.is_symmetric_quantization:
            for item in self.tensor_max:
                thresholds[item] = [max(abs(self.tensor_max[item]), abs(self.tensor_min[item]))]
        else:
            for item in self.tensor_max:
                thresholds[item] = [self.tensor_min[item], self.tensor_max[item]]

        return thresholds

    def get_raw_min(self):
        return self.tensor_min

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
