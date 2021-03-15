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

def remove_mlir_with_weight(mlir_file, mlir_weight_file):
    os.remove(mlir_file)
    os.remove(mlir_weight_file)


def parse_threshold_table(threshold_table):
    with open(threshold_table, 'r') as f:
        op_infos = f.readlines()

    op_threshold_dict = {}
    for op_info in op_infos:
        op_info = op_info.rstrip()
        if op_info.startswith("#"):
            continue
        else:
            # op_name    threshold    min    max
            op_info_list = op_info.split()
            if len(op_info_list) != 4:
                logger.error(
                    "information are op_name, threshold, min, max")
                raise RuntimeError("Error with parse {}, {}".format(
                    op_info, threshold_table))

            op_name, threshold, _min, _max = op_info_list
            op_threshold_dict[op_name] = (
                float(threshold), float(_min), float(_max))
    return op_threshold_dict


def update_calibration_table(old_calibration_table, new_calibration_table,
                             target_op_name=None, new_threshold=None):
    new_calibration_txt = ""
    with open(old_calibration_table, "r") as f:
        op_infos = f.readlines()

    for op_info in op_infos:
        _op_info = op_info.rstrip()
        if _op_info.startswith("#"):
            new_calibration_txt += op_info
        else:
            # op_name    threshold    min    max
            op_info_list = _op_info.split()
            if len(op_info_list) != 4:
                logger.error(
                    "information are op_name, threshold, min, max")
                raise RuntimeError("Error with parse {}".format(op_info))

            op_name, threshold, _min, _max = op_info_list
            if target_op_name == None:
                new_calibration_txt += "{} {} {} {}\n".format(
                    op_name, threshold, _min, _max)
                continue
            if op_name == target_op_name:
                new_calibration_txt += "{} {:.5f} {} {}\n".format(
                    target_op_name, new_threshold, _min, _max)
            else:
                new_calibration_txt += op_info
    with open(new_calibration_table, "w") as f:
        f.write(new_calibration_txt)


def generate_quanted_mlir_model(fp32_mlir, calibration_table):
    file_name = fp32_mlir.split(".")[0]
    quanted_mlir = '{}_quanted_tune.mlir'.format(file_name)

    ret = mlir_int8_quant(fp32_mlir, quanted_mlir, calibration_table)
    if ret != 0:
        raise RuntimeError("generate quanted mlir model failed")
    return quanted_mlir


class AutoTuner(object):
    def __init__(self, model_file, input_calibration_table,
                 image_list, tune_image_num, tune_iteration=10,
                 preprocess_func=None, tune_table="",
                 threshold_update_factor=0.03):
        self.fp32_mlir = model_file
        self.fp32_model = pymlir.module()
        self.fp32_model.load(model_file)

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

        if tune_table:
            self.tune_table = tune_table
        else:
            self.tune_table = "{}_tune_table".format(
                model_file.split(".")[0])

        self.skip_op = general_skip_op
        self.orignal_calibration_table = input_calibration_table
        self.thresholds_map = parse_threshold_table(input_calibration_table)

        self.threshold_update_factor = threshold_update_factor
        self.tune_iteration = tune_iteration

        self.preprocess_func = preprocess_func
        self.input_data_buffer = dict()

    def __input_data_generator(self):
        for image in self.images:
            if image not in self.input_data_buffer:
                x = self.preprocess_func(image)
                self.input_data_buffer[image] = x
            else:
                x = self.input_data_buffer[image]
            yield x

    def tune_layer(self, target_layer, threshold):
        # search up
        tune_distance = self.get_layer_best_threshold(
            threshold, -1, target_layer, True)
        logger.info(
            "tuning op: {}, enlarge tuning  down".format(target_layer))
        # search down
        # original threshold is calculated above, adjust threshold directly
        tune_distance = self.get_layer_best_threshold(
            threshold, tune_distance, target_layer, False)
        logger.info(
            "tuning op: {}, reduce tuning reduce down".format(target_layer))

        return tune_distance

    def get_layer_best_threshold(self, original_threshold, original_distance,
                                 target_layer, search_up):
        fail_count = 0

        inference_count = 0
        new_threshold = original_threshold

        delta_sum = 0
        cnt = 0
        while fail_count < self.tune_iteration:
            cnt += 1

            if original_distance != -1:
                delta_sum += new_threshold * self.threshold_update_factor
                delta = delta_sum / cnt
                if search_up:
                    new_threshold += delta
                else:
                    new_threshold -= delta
            else:
                new_threshold = original_threshold

            if new_threshold < 0:
                break
            # make tmp table
            tmp_table = "{}_tmp".format(self.tune_table)
            update_calibration_table(
                self.tune_table, tmp_table,
                target_op_name=target_layer,
                new_threshold=new_threshold)

            # import and inference
            quanted_model_mlir = generate_quanted_mlir_model(self.fp32_mlir, tmp_table)
            quanted_model = pymlir.module()
            quanted_model.load(quanted_model_mlir)

            distance = 0
            for idx, data in enumerate(self.__input_data_generator()):
                quanted_model.run(data, target_layer)
                inference_count += 1
                target_int8_activation = quanted_model.get_tensor(target_layer)

                # dequant int8 to fp32
                dequant_target_activation = target_int8_activation * new_threshold / 127.0
                # sqrt(sum(x^2))
                target_layer_fp32_activation = self.all_fp32_activations_map[idx][target_layer]
                distance += np.linalg.norm(target_layer_fp32_activation -
                                           dequant_target_activation) / self.images_num

            # if distance is small than old one, update it
            logger.info("current [{}] distance {}".format(cnt, distance))
            if original_distance == -1:
                original_distance = distance
            elif original_distance > distance:
                logger.info(
                    "tuning op: {}, tmp distance: {} < original_distance: {}, update".format(
                    target_layer, distance, original_distance))
                logger.info("tuning op: {}, old_threshold: {}, new_threshold: {}, update table: {}".format(
                    target_layer, original_threshold, new_threshold, self.tune_table))
                update_calibration_table(
                    self.tune_table, self.tune_table,
                    target_op_name=target_layer,
                    new_threshold=new_threshold)
                original_distance = distance
            else:
                fail_count += 1

            mlir_weight_file = quanted_model.get_weight_file_path()
            remove_mlir_with_weight(quanted_model_mlir, mlir_weight_file)

        logger.info("all inference count in {} op: {}".format(
            target_layer, inference_count))
        return original_distance

    def run_tune(self):
        update_calibration_table(
            self.orignal_calibration_table, self.tune_table)

        # get fp32 target layer activation
        self.all_fp32_activations_map = {}
        for idx, data in enumerate(self.__input_data_generator()):
            self.fp32_model.run(data)
            self.all_fp32_activations_map[idx] = self.fp32_model.get_all_tensor()

        pbar = tqdm(self.fp32_model.op_info)
        for info in pbar:
            op_name = info["name"]
            pbar.set_description("tune: {}".format(op_name))
            pbar.update(1)
            if info['type'] in self.skip_op:
                continue

            if op_name not in self.thresholds_map:
                continue

            # tune layer
            threshold = self.thresholds_map[op_name][0]
            tune_distance = self.tune_layer(op_name, threshold)
            logger.info("tuning op: {} finish, tune_distance: {}".format(op_name, tune_distance))

        logger.info("all tuning down, table path : {}".format(self.tune_table))
