#!/usr/bin/env python3
##
## Copyright (C) Cristal Vision Technologies Inc.
## All Rights Reserved.
##

import argparse
import os, cv2

import pymlir
import numpy as np

from tuner import Tuner

def preprocess_func(image_path):
  imgnet_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
  image = cv2.imread(str(image_path).rstrip())
  image = image.astype(np.float32)
  image -= imgnet_mean
  x = cv2.resize(image, (224, 224))
  x = np.transpose(x, (2, 0, 1))

  return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # params below used for auto tuning
    parser.add_argument('model_file', metavar='fp32_model', help='Path to the fp32 mlir model')
    parser.add_argument('input_threshold_table', metavar='input_threshold_table', help='Path to the threshold table')
    parser.add_argument('image_list_file', metavar='image_list_file', help='Input image list file')
    parser.add_argument('binary_path', metavar='binary_path', help='MLIR binary path')
    parser.add_argument('--out_path', metavar='output-path', help='Output directory', default='./')
    parser.add_argument('--tune_iteration', metavar='iteration', help='The test iteration number', default=10)

    args = parser.parse_args()

    tune = Tuner(args, preprocess_func)
    tune.run_tune(args)
