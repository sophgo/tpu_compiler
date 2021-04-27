#!/usr/bin/env python3
##
# Copyright (C) Cristal Vision Technologies Inc.
# All Rights Reserved.
##
import sys
import os
import random
import pathlib
from cvi_toolkit.utils.log_setting import setup_logger
logger = setup_logger('root')

class ImageSelector:
    def __init__(self, dataset, num, image_list_file=None, debug=False):
        self.image_list = []
        if image_list_file:
            with open(image_list_file, 'r') as f:
                for line in f.readlines():
                    self.image_list.append(line.strip())
        elif dataset:
            self.image_list = self._random_select(dataset, num)
        else:
            raise RuntimeError("Please specific dataset path by --dataset")
        if debug:
            self._print()

    def _random_select(self, dataset_path, num):
        full_list = []
        for file in pathlib.Path(dataset_path).glob('**/*'):
            if file.is_file():
                full_list.append(str(file))
        random.shuffle(full_list)
        num = num if len(full_list) > num else len(full_list)
        if num == 0:
            num = len(full_list)
        return full_list[:num]

    def _print(self):
        logger.info("Random selected images:")
        for i, img in enumerate(self.image_list):
            print(" <{}> {}".format(i, img))

    def dump(self, file):
        with open(file, 'w') as f:
            for img in self.image_list:
                f.write(img + '\n')
