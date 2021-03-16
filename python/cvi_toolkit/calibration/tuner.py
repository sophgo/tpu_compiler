#!/usr/bin/env python3
##
# Copyright (C) Cristal Vision Technologies Inc.
# All Rights Reserved.
##

import numpy as np
import sys
import os
import copy
import math
import shutil
import time
import pymlir
import shutil
from tqdm import tqdm

from ..utils.mlir_shell import mlir_int8_quant
from cvi_toolkit.utils.log_setting import setup_logger
logger = setup_logger('root')

general_skip_op = [
#    'tpu.concat',
    'tpu.crop',
    'tpu.clip',
    'tpu.detectionoutput',
    'tpu.dummy',
    'tpu.exp',
    'tpu.frcn_detection',
    'tpu.load_weight',
    'tpu.input',
    'tpu.interp',
    'tpu.pad',
    'tpu.permute',
    'tpu.pixelshuffle',
    'tpu.pool_max_2d',
    'tpu.pool_mask',
    'tpu.power',
    'tpu.preprocess',
    'tpu.priorbox',
    'tpu.proposal',
    'tpu.quant',
    'tpu.quadratic_sum'
    'tpu.requant',
    'tpu.reshape',
    'tpu.reorg',
    'tpu.retinaface_detection',
    'tpu.reverse',
    'tpu.roi_pooling',
    'tpu.shuffle_channel',
    'tpu.sigmoid',
    'tpu.slice',
    'tpu.softmax',
    'tpu.square',
    'tpu.swap_channel',
    'tpu.tanh',
    'tpu.tile',
    'tpu.upsample',
    'tpu.weight_file',
    'tpu.yolo_detection',
]


class CalibrationTable:
    def __init__(self, table):
        self.headers, self.thresholds_map = self.parse(table)

    def parse(self, table):
        thresholds_map = dict()
        headers = []
        with open(table, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('#'):
                    headers.append(line)
                    continue
                # op_name    threshold    min    max
                fields = line.split(' ')
                if len(fields) != 4:
                    logger.error(
                        "Table format should be 'op_name, threshold, min, max'")
                    raise RuntimeError("Error with parse {} in {}".format(line, table))

                op_name, threshold, _min, _max = fields
                thresholds_map[op_name] = (float(threshold), float(_min), float(_max))
        return headers, thresholds_map

    def dump(self, dest_table):
        with open(dest_table, "w") as f:
            for line in self.headers:
                f.write(line + "\n")
            for k, v in self.thresholds_map.items():
                f.write("{} {:.5f} {:.5f} {:.5f}\n".format(k, *v))

    def update(self, target_op, new_threshold):
        threshold, _min, _max = self.thresholds_map[target_op]
        self.thresholds_map[target_op] = (new_threshold, _min, _max)

    def update_to(self, dest_table, target_op, new_threshold):
        with open(dest_table, "w") as f:
            for line in self.headers:
                f.write(line + "\n")
            for k, v in self.thresholds_map.items():
                if k == target_op:
                    f.write("{} {:.5f} {:.5f} {:.5f}\n".format(
                            k, new_threshold, v[1], v[2]))
                else:
                    f.write("{} {:.5f} {:.5f} {:.5f}\n".format(k, *v))


class QuantedMlirModel:
    def __init__(self, fp32_mlir, calib_table):
        self.model = None
        self.fp32_mlir = fp32_mlir
        self.calib_table = calib_table
        self.quanted_mlir_file = '{}.quanted.tune.mlir'.format(fp32_mlir)

    def __enter__(self):
        ret = mlir_int8_quant(self.fp32_mlir, self.quanted_mlir_file,
                              self.calib_table)
        if ret != 0:
            raise RuntimeError("generate quanted mlir model failed")
        self.model = pymlir.module()
        self.model.load(self.quanted_mlir_file)
        return self.model

    def __exit__(self, type, value, traceback):
        mlir_weight_file = self.model.get_weight_file_path()
        del self.model
        os.remove(self.quanted_mlir_file)
        os.remove(mlir_weight_file)


class AutoTuner(object):
    def __init__(self, model_file, input_calib_table,
                 image_list, tune_image_num, tune_iteration=10,
                 preprocess_func=None, output_tune_table="",
                 threshold_update_factor=0.03):

        self.skip_op = general_skip_op
        self.tuned_table = CalibrationTable(input_calib_table)
        self.output_tune_table = output_tune_table

        self.threshold_update_factor = threshold_update_factor
        self.tune_iteration = tune_iteration

        self.__choose_tune_images__(image_list, tune_image_num)
        self.preprocess_func = preprocess_func
        self.input_data_buffer = dict()

        self.fp32_mlir = model_file
        self.fp32_model = pymlir.module()
        self.__prepare_for_compare__()

    def __choose_tune_images__(self, image_list, tune_image_num):
        if type(image_list) == str:
            with open(image_list, 'r') as f:
                self.images = f.readlines()
        else:
            self.images = image_list
        self.images_num = min(len(self.images), tune_image_num)
        self.images = self.images[:self.images_num]
        logger.info("Selected images:")
        for i, image in enumerate(self.images):
            logger.info("[{}] {}".format(i, image))


    def __input_data_generator__(self):
        for image in self.images:
            if image not in self.input_data_buffer:
                x = self.preprocess_func(image)
                self.input_data_buffer[image] = x
            else:
                x = self.input_data_buffer[image]
            yield x

    def __prepare_for_compare__(self):
        # restore input calib table's data to
        # output tune table firstly.
        self.tuned_table.dump(self.output_tune_table)
        # get fp32 target layer activation
        self.fp32_model.load(self.fp32_mlir)
        self.all_fp32_activations_map = {}
        for idx, data in enumerate(self.__input_data_generator__()):
            self.fp32_model.run(data)
            self.all_fp32_activations_map[idx] = self.fp32_model.get_all_tensor()

    def tune_layer(self, target_layer, threshold):
        # search up
        logger.info("tuning op: {}, search up".format(target_layer))
        tune_distance = self.find_better_threshold(
            threshold, -1, target_layer, True)
        # search down
        # original threshold is calculated above, adjust threshold directly
        logger.info("tuning op: {}, search down".format(target_layer))
        tune_distance = self.find_better_threshold(
            threshold, tune_distance, target_layer, False)
        return tune_distance

    def find_better_threshold(self, prev_threshold, prev_distance,
                              target_layer, search_up):
        try_cnt = 0
        fail_cnt = 0
        cur_threshold = prev_threshold

        delta_sum = 0
        while fail_cnt < self.tune_iteration:
            if prev_distance != -1:
                delta_sum += cur_threshold * self.threshold_update_factor
                delta = delta_sum / (try_cnt + 1)
                if search_up:
                    cur_threshold += delta
                else:
                    cur_threshold -= delta
            else:
                cur_threshold = prev_threshold
            if cur_threshold < 0:
                break
            try_cnt += 1

            # make tmp table
            tmp_table = self.output_tune_table + '.tmp'
            self.tuned_table.update_to(tmp_table, target_layer, cur_threshold)

            # import and inference
            with QuantedMlirModel(self.fp32_mlir, tmp_table) as model:
                cur_distance = 0
                for idx, data in enumerate(self.__input_data_generator__()):
                    # inference until target layer
                    model.run(data, target_layer)

                    # dequant int8 to fp32
                    target_int8_activation = model.get_tensor(target_layer)
                    scale = cur_threshold / 127.0
                    dequant_target_activation = target_int8_activation * scale
                    # sqrt(sum(x^2))
                    target_layer_fp32_activation = self.all_fp32_activations_map[idx][target_layer]
                    cur_distance += np.linalg.norm(target_layer_fp32_activation -
                                                   dequant_target_activation) / self.images_num

                # if distance is small than old one, update it
                logger.info("cur[{}] threshold: {:5f}, distance: {:5f}".format(
                            try_cnt, cur_threshold, cur_distance))

                if prev_distance == -1:
                    prev_distance = cur_distance
                elif cur_distance < prev_distance:
                    logger.info(
                        "### tuning {}, find a better threshold: {:5f} -> {:5f}, distance: {:5f} -> {:5f}".format(
                        target_layer, prev_threshold, cur_threshold, prev_distance, cur_distance))

                    self.tuned_table.update(target_layer, cur_threshold)
                    self.tuned_table.dump(self.output_tune_table)
                    prev_distance = cur_distance
                    prev_threshold = cur_threshold
                else:
                    fail_cnt += 1
        return prev_distance

    def run(self):
        pbar = tqdm(self.fp32_model.op_info)
        for info in pbar:
            op_name = info["name"]
            pbar.set_description("tune: {}".format(op_name))
            pbar.update(1)
            if info['type'] in self.skip_op:
                continue

            if op_name not in self.tuned_table.thresholds_map:
                continue

            # tune layer
            threshold = self.tuned_table.thresholds_map[op_name][0]
            tune_distance = self.tune_layer(op_name, threshold)
            logger.info("tuning op: {} finish, tune_distance: {}".format(op_name, tune_distance))

        logger.info("all tuning down, table path : {}".format(self.tune_table))
