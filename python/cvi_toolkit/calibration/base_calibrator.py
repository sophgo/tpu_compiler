#!/usr/bin/env python3
#
# Copyright (C) Cristal Vision Technologies Inc.
# All Rights Reserved.
##

import cv2
import numpy as np
import sys
import os
import abc
import copy
import math
import logging
import pymlir

from tqdm import tqdm
from cvi_toolkit.utils.log_setting import setup_logger

logger = setup_logger('root')


class Base_Calibrator(object):
    def __init__(self,
                 image_list,
                 mlir_file,
                 preprocess_func,
                 is_symmetric_quantization=True,
                 custom_op_plugin=''):
        if type(image_list) == str:
            with open(image_list, 'r') as fp:
                self.images = fp.readlines()
        else:
            self.images = image_list

        logger.info("Images list number: {}".format(len(self.images)))
        if len(self.images) == 0:
            raise IOError("ERROR: No calibration image detect.")

        self.input_num = len(self.images)

        self.preprocess_func = preprocess_func
        self.model = pymlir.module()
        self.model.set_plugin(custom_op_plugin)
        self.model.load(mlir_file)

        self.tensor_max = {}
        self.tensor_min = {}
        self.is_symmetric_quantization = is_symmetric_quantization

    @abc.abstractmethod
    def run(self):
        return NotImplemented
