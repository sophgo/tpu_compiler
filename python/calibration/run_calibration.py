#!/usr/bin/env python3
##
## Copyright (C) Cristal Vision Technologies Inc.
## All Rights Reserved.
##

import argparse
import sys, os, cv2
import numpy as np

from kld_calibrator import KLD_Calibrator
from asym_calibrator import Asym_Calibrator
from tuner import Tuner

def preprocess_func_ssd300_face(image_path):
  imgnet_mean = np.array([104, 177, 123], dtype=np.float32)
  image = cv2.imread(str(image_path).rstrip())
  image = image.astype(np.float32)
  image -= imgnet_mean
  x = cv2.resize(image, (300, 300))
  x = np.transpose(x, (2, 0, 1))
  x = np.expand_dims(x, axis=0)
  return x

def preprocess_func(image_path):
  imgnet_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
  image = cv2.imread(str(image_path).rstrip())
  image = image.astype(np.float32)
  image -= imgnet_mean
  x = cv2.resize(image, (224, 224))
  x = np.transpose(x, (2, 0, 1))
  x = np.expand_dims(x, axis=0)
  return x


def preprocess_yolov3(image_path, net_input_dims=(416,416)):
  bgr_img = cv2.imread(str(image_path).rstrip())
  yolo_w = net_input_dims[1]
  yolo_h = net_input_dims[0]
  rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
  rgb_img = rgb_img / 255.0

  ih = rgb_img.shape[0]
  iw = rgb_img.shape[1]

  scale = min(float(yolo_w) / iw, float(yolo_h) / ih)
  rescale_w = int(iw * scale)
  rescale_h = int(ih * scale)

  resized_img = cv2.resize(rgb_img, (rescale_w, rescale_h), interpolation=cv2.INTER_LINEAR)
  new_image = np.full((yolo_h, yolo_w, 3), 0, dtype=np.float32)
  paste_w = (yolo_w - rescale_w) // 2
  paste_h = (yolo_h - rescale_h) // 2

  new_image[paste_h:paste_h + rescale_h, paste_w: paste_w + rescale_w, :] = resized_img
  new_image = np.transpose(new_image, (2, 0, 1))      # row to col, (HWC -> CHW)
  new_image = np.expand_dims(new_image, axis=0)
  return new_image

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name', metavar='model-name', help='Model name')
  parser.add_argument('model_path', metavar='model-path', help='Model path')
  parser.add_argument('input_file', metavar='input_file', help='Input data file')
  parser.add_argument('--calibrator', metavar='calibrator', help='Calibration method', default='KLD')
  parser.add_argument('--out_path', metavar='path', help='Output path', default='./result')
  parser.add_argument('--math_lib_path', metavar='math_path', help='Calibration math library path', default='./lib/calibration_math.so')
  parser.add_argument('--input_num', metavar='input_num', help='Calibration data number', default=10)
  parser.add_argument('--histogram_bin_num', metavar='histogram_bin_num', help='Specify histogram bin numer for kld calculate',
                      default=2048)
  parser.add_argument('--auto_tune', action='store_true', help='Enable auto tune or not')
  parser.add_argument('--binary_path', metavar='binary_path', help='MLIR binary path')
  parser.add_argument('--tune_iteration', metavar='iteration', help='The number of data using in tuning process', default=10)
  args = parser.parse_args()

  if not os.path.isdir(args.out_path):
    os.mkdir(args.out_path)

  if (args.model_name == 'resnet50'):
    preprocess = preprocess_func
  elif (args.model_name == 'yolo_v3'):
    preprocess = preprocess_yolov3
  elif (args.model_name == 'ssd300_face'):
    preprocess = preprocess_func_ssd300_face
  else:
    assert(False)

  if args.calibrator == 'KLD':
    calibrator = KLD_Calibrator(args, preprocess)
  elif args.calibrator == 'Asym':
    calibrator = Asym_Calibrator(args, preprocess)
  else:
    assert(False)

  thresholds = calibrator.do_calibration()
  threshold_table = '{}/{}_threshold_table'.format(args.out_path, args.model_name)
  calibrator.dump_threshold_table(threshold_table, thresholds)

  if args.auto_tune == True:
    args.input_threshold_table = threshold_table
    tune = Tuner(args, preprocess_func)
    tune.run_tune(args)
