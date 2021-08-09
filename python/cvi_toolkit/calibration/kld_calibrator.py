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
import pytuner

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


class ThresholdTable:
    def __init__(self, candidate_threshold_map, activations_statistics):
        self.candidate_threshold_map = self.transform_threshold_map(candidate_threshold_map)
        self.activations_statistics = activations_statistics
        self.thresholds_map = {}
        for k, v in self.candidate_threshold_map.items():
            self.thresholds_map[k] = v[0]

    def transform_threshold_map(self, candidate_threshold_map):
        threshold_list_map = {}
        for k, _map in candidate_threshold_map.items():
            for op, threshold in _map.items():
                if op not in threshold_list_map:
                    threshold_list_map[op] = [threshold]
                else:
                    threshold_list_map[op].append(threshold)
        return threshold_list_map

    def update_to(self, dest_table, target_op, new_threshold):
        with open(dest_table, "w") as f:
            for k, v in self.thresholds_map.items():
                _min, _max, _ = self.activations_statistics[k]
                if k == target_op:
                    threshold = new_threshold
                else:
                    threshold = v
                f.write("{} {:.7f} {:.7f} {:.7f}\n".format(
                        k, threshold, _min, _max))

    def update(self, target_op, best_threshold):
        self.thresholds_map[target_op] = best_threshold

    def candidate_thresholds(self, target_op):
        return self.candidate_threshold_map[target_op]


class SimpleTuner:
    def __init__(self, fp32_mlir, thresholds_map, activations_statistics,
                 images, image_num, preprocessor):
        image_num = min(len(images), image_num)
        self.images = images[:image_num]
        self.fp32_mlir = fp32_mlir
        self.threshold_table = ThresholdTable(thresholds_map,
                                             activations_statistics)
        self.tuner = pytuner.tuner(len(self.images))
        self._pre_run(preprocessor)

    def _load_data(self, preprocessor):
        def is_npz(image):
            return True if image.split('.')[-1] == 'npz' else False
        for i, image in enumerate(self.images):
            if is_npz(image):
                x = np.load(image)
                for k, v in x.items():
                    self.tuner.set_data(k, v, i)
            else:
                x = preprocessor.run(image)
                self.tuner.set_data(x, i)

    def _pre_run(self, preprocessor):
        self.tuner.load(self.fp32_mlir)
        self.tuner.build()
        self._load_data(preprocessor)
        self.tuner.invoke()
        self.ref_fp32_activations = {}
        for i in range(len(self.images)):
            self.ref_fp32_activations[i] = self.tuner.get_all_tensors(i)

    def run(self):
        tmp_table = 'tmp_table.txt'
        op_info = self.tuner.op_info()
        pbar = tqdm(op_info)
        for op in pbar:
            op_name = op['name']
            op_type = op['type']
            pbar.set_description("tune: {}".format(op_name))
            pbar.update(1)

            if op_type == 'tpu.input':
                continue

            candidates = self.threshold_table.candidate_thresholds(op_name)
            minimum_distance = float('inf')
            best_threshold = candidates[0]
            for threshold in candidates:
                if threshold == 0:
                    continue
                self.threshold_table.update_to(tmp_table, op_name, threshold)
                self.tuner.load(self.fp32_mlir)
                self.tuner.quantize(tmp_table)
                self.tuner.build(op_name)
                self.tuner.invoke(op_name)
                data_type = self.tuner.get_tensor_type(op_name)
                distance = 0
                for i in range(len(self.images)):
                    target_activations = self.tuner.get_tensor(op_name, i)
                    if data_type == 'INT8':
                        scale = threshold / 127.0
                        target_activations = target_activations * scale
                    target_fp32_activations = self.ref_fp32_activations[i][op_name]
                distance += np.linalg.norm(target_fp32_activations.flatten() -
                                           target_activations.flatten())
                distance /= len(self.images)
                # tqdm.write("tuning {}, threshold: {}, distance:{}".format(op_name,
                #            threshold, distance))
                if distance < minimum_distance:
                    best_threshold = threshold
                    minimum_distance = distance
            tqdm.write("{} => threshold: {}\n".format(op_name, best_threshold))
            self.threshold_table.update(op_name, best_threshold)
        pbar.close()
        return self.threshold_table.thresholds_map


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
        self.fp32_mlir = mlir_file

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
        return size * 4

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

    def _clean_resource(self):
        del self.buffered_activations
        self.buffered_activations = None
        del self.model
        self.model = None

    def find_min_max_abs(self):
        activations_statistics = dict()

        pbar = tqdm(self.images, total=self.input_num, position=0, leave=True)
        for image, activations in self._activations_generator():
            pbar.set_description("Find Min Max *{}".format(self.free_buffer_size))
            pbar.update(1)

            for op_name, activation in activations.items():
                if op_name not in activations_statistics:
                    minimum, maximum = np.min(activation), np.max(activation)
                else:
                    minimum, maximum, _ = activations_statistics[op_name]
                min_value = min(np.min(activation), minimum)
                max_value = max(np.max(activation), maximum)
                abs_value = max(abs(min_value), abs(max_value))
                activations_statistics[op_name] = (min_value, max_value, abs_value)

        # check max is zero
        for k, v in activations_statistics.items():
            _min, _max, _abs = v
            if _abs == 0:
                # if network outputs are all zero, change it to 1e-5 for them.
                activations_statistics[k] = (-1e-5, 1e-5, 1e-5)
                logger.warning("WARNING: layer {} is all zeros. Please check the "
                               "input data correctness.".format(k))
        pbar.close()
        return activations_statistics

    def calc_thresholds(self, activations_statistics, hist_bin_nums):
        logger.info("calculate histogram..")

        histogram_data_map = {}
        histogram_width_map = {}
        for bin_num in hist_bin_nums:
            histogram_data_map[bin_num] = {}
            histogram_width_map[bin_num] = {}

        pbar = tqdm(self.images, total=self.input_num, position=0, leave=True)
        for image, activations in self._activations_generator():
            img_name = image.split("/")[-1]
            pbar.set_description("histogram: {}".format(img_name))
            pbar.update(1)

            for op_name, activation in activations.items():
                _, _, abs_value = activations_statistics[op_name]
                for bin_num in histogram_data_map.keys():
                    hist, width = self.histogram(activation, abs_value, bin_num)
                    if op_name not in histogram_data_map[bin_num]:
                        histogram_data_map[bin_num][op_name] = hist
                        histogram_width_map[bin_num][op_name] = width
                    else:
                        histogram_data_map[bin_num][op_name] += hist
            del activations
        pbar.close()


        thresholds_map = {}
        for bin_num in histogram_data_map.keys():
            thresholds_map[bin_num] = self.find_threshold(histogram_data_map[bin_num],
                                                          histogram_width_map[bin_num],
                                                          bin_num)
        thresholds_map['abs_max'] = {}
        for k, v in activations_statistics.items():
            _, _, abs_val = v
            thresholds_map['abs_max'][k] = abs_val

        return thresholds_map

    def find_threshold(self, histogram_data_map,
                       histogram_width_map,
                       histogram_bin_num):
        thresholds = {}
        num = len(histogram_data_map)
        pbar = tqdm(range(num), total=num, position=0, leave=True)
        for item in histogram_data_map:
            pbar.set_description("threshold: {}".format(item))
            pbar.update(1)
            thresholds[item] = self.kld_threshold(histogram_data_map[item],
                                                  histogram_width_map[item],
                                                  histogram_bin_num)
        pbar.close()
        return thresholds

    def run(self, output_calibration_table):
        # step 1: find min max
        op_layers = self.model.op_info
        activations_statistics = self.find_min_max_abs()

        # step 2: set histogram bins
        hist_bin_nums = [ (2 ** i) * 512 for i in range(7)]
        if self.histogram_bin_num not in hist_bin_nums:
            hist_bin_nums.append(self.histogram_bin_num)

        # step 3: calculate threshold with histogram bins
        thresholds_map = self.calc_thresholds(activations_statistics, hist_bin_nums)
        self._clean_resource()

        # step 6: dump threshold table of default histogram bins
        with open(output_calibration_table + '.1', 'w') as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# histogram number: {}\n".format(self.histogram_bin_num))
            f.write("# sample number: {}\n###\n".format(self.input_num))
            f.write("# op_name    threshold    min    max\n")
            for op in op_layers:
                op_name = op['name']
                threshold = thresholds_map[self.histogram_bin_num][op_name]
                min_value, max_value, _ = activations_statistics[op_name]
                f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold,
                                                         min_value, max_value))


        # setp 4: tune to get better threshold of each layers.
        self.tuner = SimpleTuner(self.fp32_mlir, thresholds_map, activations_statistics,
                                 self.images, 5, self.preprocessor)
        thresholds = self.tuner.run()

        # step 5: dump threshold table after tuning
        with open(output_calibration_table, 'w') as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# sample number: {}\n###\n".format(self.input_num))
            f.write("# op_name    threshold    min    max\n")
            for op in op_layers:
                op_name = op['name']
                threshold = thresholds[op_name]
                min_value, max_value, _ = activations_statistics[op_name]
                f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold,
                                                           min_value, max_value))


        # step 7: dump all thresholds to csv files
        with open(output_calibration_table + '.csv', 'w') as f:
            f.write("name,default({}),best,min,max,abs,{}\n".format(self.histogram_bin_num,
                   ','.join([str(x) for x in hist_bin_nums])))
            for op in op_layers:
                op_name = op['name']
                default_threshold = thresholds_map[self.histogram_bin_num][op_name]
                best_threshold = thresholds[op_name]
                min_value, max_value, abs_value = activations_statistics[op_name]
                f.write("{},{:.7f},{:.7f},{:.7f},{:.7f},{:5f},".format(
                        op_name, default_threshold,
                        best_threshold, min_value, max_value, abs_value))
                _str_thres = []
                for x in hist_bin_nums:
                    _str_thres.append('{:.7f}'.format(thresholds_map[x][op_name]))
                f.write(','.join(_str_thres))
                f.write('\n')
