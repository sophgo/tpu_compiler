#!/usr/bin/env python3
##
## Copyright (C) Cristal Vision Technologies Inc.
## All Rights Reserved.
##

import cv2
import numpy as np
import sys, os, copy, math
import pymlir

def is_all_zero(data):
    for num in data:
        if num != 0:
            return False
    return True

def warn_zeros(layer_name):
    print("WARNING: layer {} is all zeros. Please check the input data "
          "correctness.".format(layer_name))
    exit(-1)

class Asym_Calibrator(object):
    def __init__(self, args, preprocess_func):
        with open(args.image_list_file,'r') as fp:
            self.all_lines = fp.readlines()

        if len(self.all_lines) == 0:
            print("ERROR: No calibration data detect. "
                  "Please check the input file: {}".format(args.image_list_file))
            exit(-1)

        self.input_num = int(args.input_num)
        self.preprocess_func = preprocess_func

        self.module = pymlir.module()
        self.module.load(args.model_file)

    def do_find_min_max(self):
        data_max = {}
        data_min = {}
        idx = 0
        for line in self.all_lines:
            print('Calculating min/max at iteration: ', str(idx))

            x = self.preprocess_func(line)
            _ = self.module.run(x)
            data = self.module.get_all_tensor()

            for item in data:
                if item not in data_max:
                    data_max[item] = sys.float_info.min
                if item not in data_min:
                    data_min[item] = sys.float_info.max

                t = data[item].flatten()
                if is_all_zero(t):
                    warn_zeros(item)
                data_max[item] = max(data_max[item], np.max(t))
                data_min[item] = min(data_min[item], np.min(t))

            idx += 1
            if idx >= self.input_num:
                break

        return data_min, data_max

    def do_calibration(self):
        data_min, data_max = self.do_find_min_max()

        thresholds = {}
        for item in data_min:
            thresholds[item] = []
            thresholds[item].append(data_min[item])
            thresholds[item].append(data_max[item])

        return thresholds


    @staticmethod
    def dump_threshold_table(threshold_table, thresholds):
        with open(threshold_table, 'w') as outfile:
            for layer in thresholds:
                line = layer
                for num in thresholds[layer]:
                    line += ' ' + str(num)
                outfile.write(line)
                outfile.write('\n')


