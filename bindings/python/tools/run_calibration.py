#!/usr/bin/env python3

import argparse
import sys, os, cv2
import numpy as np

from kld_calibrator import KLD_Calibrator

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

  parser.add_argument('model_name', metavar='model-name', help='model name')
  parser.add_argument('model_path', metavar='model-path', help='model path')
  parser.add_argument('input_file', metavar='input_file', help='input data file')
  parser.add_argument('--out_path', metavar='path', help='output path', default='./')
  parser.add_argument('--math_lib_path', metavar='math_path', help='math library path', default='./calibration_math.so')
  parser.add_argument('--input_num', metavar='iteration', help='Calibration data number', default=10)
  parser.add_argument('--histogram_bin_num', metavar='histogram_bin_num', help='specify histogram bin numer for kld calculate',
                      default=2048)
  args = parser.parse_args()

  Calibrator = KLD_Calibrator(args, preprocess_func)
  thresholds = Calibrator.do_calibration()

  threshold_table = '{}/{}_threshold_table'.format(args.out_path, args.model_name)
  with open(threshold_table, 'w') as outfile:
    for layer in thresholds:
      line = layer + ' ' + str(thresholds[layer])
      outfile.write(line)
      outfile.write('\n')
