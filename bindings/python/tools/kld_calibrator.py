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

def second(elem):
  return elem[1]
def get_topk(a, k):
  idx = np.argpartition(-a.ravel(),k)[:k]
  # return np.column_stack(np.unravel_index(idx, a.shape))
  topk = list(zip(idx, np.take(a, idx)))
  #return topk
  topk.sort(key=second, reverse=True)
  return topk

class KLD_Calibrator(object):
    def __init__(self, args, preprocess_func):
        with open(args.input_file,'r') as fp:
            self.all_lines = fp.readlines()
        self.input_num = int(args.input_num)
        self.preprocess_func = preprocess_func

        self.module = pymlir.module()
        self.module.load(args.model_path)
        self.histogram_bin_num = args.histogram_bin_num

        self.calibration_math = CDLL(args.math_lib_path)
        self.calibration_math.kl_diversity.restype = c_float
        self.calibration_math.kl_diversity_hist.restype = c_float

    def KLD_hist(self, data, width):
        return self.calibration_math.kl_diversity_hist(data.ctypes.data_as(POINTER(c_int)), c_float(width), c_longlong(self.histogram_bin_num))

    def do_find_max(self):
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

                t = np.abs(data[item].flatten())
                t = t[t!=0]

                if t.size > 0:
                    if is_all_zero(t):
                        warn_zeros(layer_caffe.name, b.name)
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
            _ = self.module.run(x)
            data = self.module.get_all_tensor()

            for item in data:
                t = np.abs(data[item].flatten())
                t = t[t!=0]

                width = data_max[item] / (self.histogram_bin_num - 1)
                if t.size > 0:
                    hist, bins = np.histogram(np.floor(t / width + 0.5), bins=self.histogram_bin_num, range=(0, self.histogram_bin_num-1), density=False)
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
            thresholds[item] = self.KLD_hist(data_hist[item], width_hist[item])

        return thresholds

