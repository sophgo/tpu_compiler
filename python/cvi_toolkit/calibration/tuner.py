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

from ..utils.mlir_shell import mlir_import_calibration, mlir_tpu_quant, mlir_opt
from cvi_toolkit.utils.log_setting import setup_logger
logger = setup_logger('root')

general_skip_op = [
    'tpu.concat',
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


def get_mlir_weight_file(mlir_file):
    model = pymlir.module()
    model.load(mlir_file)
    weight_file = model.get_weight_file_path()
    del model
    return weight_file


def remove_mlir_with_weight(mlir_file):
    weight_file = get_mlir_weight_file(mlir_file)
    os.remove(weight_file)
    os.remove(mlir_file)


def get_op_threshold_from_calibration_tabel(calibration_table, target_op):
    op_threshold_dict = parse_threshold_table(calibration_table)
    if target_op not in op_threshold_dict:
        raise KeyError("{} op not in {}".format(target_op, calibration_table))
    return op_threshold_dict[target_op][0]


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


def update_calibration_table(old_calibration_table, new_calibration_table, target_op_name=None, new_threshold=None):
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


def import_calibration_get_int8_mlir(calibration_table, fp32_mlir_file):
    fp32_mlir_opt_file = "{}_tune_opt.mlir".format(
        fp32_mlir_file.split(".")[0])

    mlir_opt(fp32_mlir_file, fp32_mlir_opt_file)
    calibration_mlir = "{}_tune_cali.mlir".format(
        fp32_mlir_file.split(".")[0])

    ret = mlir_import_calibration(
        fp32_mlir_opt_file, calibration_mlir, calibration_table)

    if ret != 0:
        raise RuntimeError("import table failed")

    int8_tune_mlir_file = "{}_tune_int8.mlir".format(
        fp32_mlir_file.split(".")[0])
    ret = mlir_tpu_quant(calibration_mlir, int8_tune_mlir_file, "no_use")

    # weight_file can not remove, tpu_quant will use
    os.remove(calibration_mlir)
    if ret != 0:
        raise RuntimeError("quant failed")

    return int8_tune_mlir_file


class Tuner_v2(object):
    def __init__(self, model_file, input_calibration_table,
                 image_list_file, output_path="./", tune_iteration=10,
                 preprocess_func=None):
        self.fp32_mlir_file = model_file
        self.fp32_mlir_opt_file = "{}_tune_opt.mlir".format(
            model_file.split(".")[0])
        self.fp32_model = pymlir.module()
        self.fp32_model.load(model_file)

        with open(image_list_file, 'r') as f:
            self.images = f.readlines()

        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.tune_table = "{}_tune_table".format(
            model_file.split(".")[0])
        self.int8_model = os.path.join(self.output_path, 'tune-int8.mlir')

        self.cali_model = "{}_tune_cali.mlir".format(model_file.split(".")[0])
        self.skip_op = general_skip_op
        self.orignal_calibration_table = input_calibration_table
        self.thresholds = parse_threshold_table(input_calibration_table)

        self.enlarge_factor = 0.5
        self.reduce_factor = -0.5

        self.best_threshold = 0
        self.best_diff = 0
        self.limit = min(len(self.images), tune_iteration)

        self.preprocess_func = preprocess_func


    def tune_layer(self, target_layer, threshold):
        original_distance = sys.float_info.max

        # search up
        tune_distance = self.get_layer_best_threshold(
            threshold, original_distance, target_layer, self.enlarge_factor)
        logger.info(
            "tuning op: {}, enlarge tuning  down".format(target_layer))
        # search down
        # original threshold is calculated above, adjust threshold directly
        tune_distance = self.get_layer_best_threshold(
            threshold, tune_distance, target_layer, self.reduce_factor)
        logger.info(
            "tuning op: {}, reduce tuning reduce down".format(target_layer))

        return tune_distance

    def get_layer_best_threshold(self, original_threshold, original_distance, target_layer, factor):
        fail_count = 0

        inference_count = 0
        new_threshold = original_threshold
        # get fp32 target layer activation
        target_layer_fp32_activation = {}
        for idx, image in enumerate(self.images):
            x = self.preprocess_func(image)
            self.fp32_model.run(x)
            target_layer_fp32_activation[idx] = self.fp32_model.get_tensor(
                target_layer)
            if idx >= self.limit:
                break

        while fail_count < 3:
            distance = 0
            new_threshold += factor
            if new_threshold < 0:
                break
            # make tmp table
            tmp_table = "{}_tmp".format(self.tune_table)
            update_calibration_table(
                self.tune_table, tmp_table, target_op_name=target_layer, new_threshold=new_threshold)

            # import and inference
            tmp_int8_mlir_file = import_calibration_get_int8_mlir(
                tmp_table, self.fp32_mlir_file)

            self.int8_model = pymlir.module()
            self.int8_model.load(tmp_int8_mlir_file)

            for idx, image in enumerate(self.images):
                x = self.preprocess_func(image)
                self.int8_model.run(x)
                inference_count += 1
                target_int8_activation = self.int8_model.get_tensor(
                    target_layer)

                # dequant int8 to fp32
                dequant_target_activation = target_int8_activation * new_threshold / 127.0
                distance += np.linalg.norm(
                    target_layer_fp32_activation[idx] - dequant_target_activation) / self.limit
                if idx >= self.limit:
                    break

            # if distance is small than old one, update it
            if original_distance > distance:
                logger.info(
                    "tuning op: {}, tmp distance: {} < original_distance: {}, update".format(target_layer, distance, original_distance))
                logger.info("tuning op: {}, old_threshold: {}, new_threshold: {}, update table: {}".format(
                    target_layer, original_threshold, new_threshold, self.tune_table))
                update_calibration_table(
                    self.tune_table, self.tune_table, target_op_name=target_layer, new_threshold=new_threshold)
                original_distance = distance
            else:
                fail_count += 1

            remove_mlir_with_weight(tmp_int8_mlir_file)

        logger.info("all inference count in {} op: {}".format(
            target_layer, inference_count))
        return original_distance

    def run_tune(self):
        update_calibration_table(
            self.orignal_calibration_table, self.tune_table)
        int8_tune_mlir = import_calibration_get_int8_mlir(
            self.orignal_calibration_table, self.fp32_mlir_file)
        tmp_int8_module = pymlir.module()
        tmp_int8_module.load(int8_tune_mlir)
        pbar = tqdm(tmp_int8_module.op_info)
        for info in pbar:
            op_name = info["name"]
            pbar.set_description("tune: {}".format(op_name))
            pbar.update(1)
            if info['type'] in self.skip_op:
                continue
            # tune layer
            tune_distance = self.tune_layer(
                op_name, get_op_threshold_from_calibration_tabel(self.tune_table, op_name))
            logger.info(
                "tuning op: {} finish, tune_distance: {}".format(op_name, tune_distance))

        logger.info("all tuning down, table path : {}".format(self.tune_table))
