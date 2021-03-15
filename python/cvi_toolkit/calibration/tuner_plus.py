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
from scipy import spatial

from ..utils.mlir_shell import mlir_import_calibration, mlir_mix_quant, mlir_tpu_quant, mlir_opt
from cvi_toolkit.utils.log_setting import setup_logger
logger = setup_logger('root')


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


def import_calibration_get_int8_mlir(calibration_table, mix_table, fp32_mlir_file):
    mix_tune_mlir_file = "{}_tune_mix.mlir".format(
        fp32_mlir_file.split(".")[0])
    ret = mlir_mix_quant(fp32_mlir_file, mix_tune_mlir_file, calibration_table, mix_table)
    if ret != 0:
        raise RuntimeError("quant failed")
    return mix_tune_mlir_file


class TunerPlus(object):
    def __init__(self, model_file, input_calibration_table, image_list, tune_image_num,
                 tune_iteration=10, preprocess_func=None, tune_table="", mix_table="", skip_op_list=[]):
        self.fp32_mlir_file = model_file
        self.fp32_model = pymlir.module()
        self.fp32_model.load(model_file)

        if type(image_list) == str:
            with open(image_list, 'r') as f:
                self.images = f.readlines()
        else:
            self.images = image_list
        self.images_num = len(self.images)

        self.tune_table = tune_table
        update_calibration_table(input_calibration_table, self.tune_table)

        self.mix_table = mix_table
        self.skip_op_list = skip_op_list

        self.enlarge_factor = 0.5
        self.reduce_factor = -0.5

        self.tune_iteration = tune_iteration
        self.preprocess_func = preprocess_func
        self.image_cache = dict()

    def __image_proprocessor(self):
        for image in self.images:
            if image not in self.image_cache:
                x = self.preprocess_func(image)
                self.image_cache[image] = x
            else:
                x = self.image_cache[image]
            yield x

    def tune_layer(self, target_op, tune_op):
        threshold = get_op_threshold_from_calibration_tabel(self.tune_table, tune_op['name'])
        # search up
        tune_distance = self.get_layer_best_threshold(
            threshold, -1, target_op, tune_op, self.enlarge_factor)
        logger.info(
            "tuning op: {}-{}, enlarge tuning down".format(target_op['name'], tune_op['name']))

        # search down
        # original threshold is calculated above, adjust threshold directly
        tune_distance = self.get_layer_best_threshold(
            threshold, tune_distance, target_op, tune_op, self.reduce_factor)
        logger.info(
            "tuning op: {}-{}, reduce tuning reduce down".format(target_op['name'], tune_op['name']))

        return tune_distance

    def get_layer_best_threshold(self, original_threshold, current_distance, target_op, tune_op, factor):
        fail_count = 0

        inference_count = 0
        new_threshold = original_threshold

        delta_sum = 0
        cnt = 0
        while fail_count < self.tune_iteration:
            cnt += 1

            if current_distance == -1:
                new_threshold = original_threshold
            else:
                delta_sum += new_threshold * 0.05
                delta = delta_sum / cnt
                if factor > 0:
                    new_threshold += delta
                else:
                    new_threshold -= delta

            if new_threshold <= 0:
                break

            # make tmp table
            tmp_table = "{}_tmp".format(self.tune_table)
            update_calibration_table(
                self.tune_table, tmp_table, target_op_name=tune_op['name'], new_threshold=new_threshold)

            # import and inference
            tmp_int8_mlir_file = import_calibration_get_int8_mlir(
                tmp_table, self.mix_table, self.fp32_mlir_file)

            self.int8_model = pymlir.module()
            self.int8_model.load(tmp_int8_mlir_file)

            distance = 0
            for idx, image in enumerate(self.__image_proprocessor()):
                x = self.preprocess_func(image)
                self.int8_model.run(x)
                inference_count += 1
                target_int8_activation = self.int8_model.get_tensor(
                    target_op['name'])

                # dequant int8 to fp32
                if ()
                dequant_target_activation = target_int8_activation * new_threshold / 127.0
                # sqrt(sum(x^2))
                distance += spatial.distance.cosine(
                                self.target_layer_fp32_activation[idx].flatten(),
                                target_int8_activation.flatten()) / self.images_num

            # if distance is small than old one, update it
            logger.info("current [{}] distance {}".format(cnt, distance))
            if cnt == 1:
                current_distance = distance

            if current_distance > distance:
                logger.info(
                    "tuning op: {}-{}, tmp distance: {} < current_distance: {}, update".format(
                    target_op['name'], tune_op['name'], distance, current_distance))
                logger.info("tuning op: {}-{}, old_threshold: {}, new_threshold: {}, update table: {}".format(
                    target_op['name'], tune_op['name'], original_threshold, new_threshold, self.tune_table))
                update_calibration_table(
                    self.tune_table, self.tune_table, target_op_name=tune_op['name'], new_threshold=new_threshold)
                current_distance = distance
            else:
                fail_count += 1

            remove_mlir_with_weight(tmp_int8_mlir_file)

        logger.info("all inference count in {} op: {}".format(
            target_op['name'], inference_count))
        return current_distance

    def run_tune(self):
        outputs = self.fp32_module.get_output_details()
        target_op = [x for x.name == outputs[0] in self.fp32_module.op_info][0]

        # get fp32 target layer activation
        self.target_layer_fp32_activation = {}
        for idx, data in enumerate(self.__image_proprocessor()):
            self.fp32_model.run(idx)
            self.target_layer_fp32_activation[idx] = self.fp32_model.get_tensor(
                target_op['name'])

        pbar = tqdm(self.fp32_module.op_info)
        for tune_op in pbar:
            pbar.set_description("tune: {}".format(tune_op['name']))
            pbar.update(1)

            if tune_op['name'] in self.skip_op_list:
                continue

            # tune layer
            tuned_distance = self.tune_layer(target_op, tune_op)
            logger.info(
                "tuning op: {}-{} finish, tune_distance: {}".format(
                    target_op['name'], tune_op['name'], tuned_distance))
