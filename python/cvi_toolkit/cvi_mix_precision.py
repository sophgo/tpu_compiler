#!/usr/bin/env python
import csv
import argparse
import os
import numpy as np

#
# function comes from bmcompress, plz refre [here](http://10.34.33.3:8480/toolchain/bmcompress/blob/master/experiment/imagenet/mobilenet_v1_pytorch/sqnr_mobilenet_v1.py) for more details
#
from mix_precision.gen_imagenet_list import imagenet_generator
from mix_precision.inference_demo_mobilenet import preprocess
from mix_precision.get_csv_val import get_csv_val, get_rows_by_column

import pymlir

import logging
from utils.log_setting import setup_logger
logger = setup_logger('root')
log_flag = logger.level <= logging.DEBUG

def parse_args():
  parser = argparse.ArgumentParser(description="find bf16 layers in int8 net")
  parser.add_argument(
      "--all_layers_name_csv_file",
      help="file record all layers name in this network, may be like mobilenetv3_pytorch1_op_info.csv",
      required=True
  )
  parser.add_argument(
      "--layers_column_name",
      help="column name in --all_layers_name_csv_file",
      default="input"
  )
  parser.add_argument(
      "--net_name",
      help="net name, we could create folder for save temp data",
      required=True
  )
  parser.add_argument(
      "--gen_cmd_script",
      help="script to generate cmd, the possible file like gen_mix_precision.sh",
      required=True
  )
  parser.add_argument(
      "--model",
      help="mlir model name such like mobilenetv3_pytorch1_op_info.mlir",
      required=True
  )
  args = parser.parse_args()
  return args


skip_ops = ['tpu.input', 'tpu.quant']
g_val_data_count = 100

def create_bf16_layers(args, layers, exclude_layers = []):
  with open(args.bf16_quant_layers_file, 'w') as myfile:
    for bf16_layer in layers:
      if bf16_layer not in exclude_layers:
        myfile.write(bf16_layer + "\n")

def gen_mlir(args):
  cmd = '{} {} {}'.format(
    args.gen_cmd_script,
    args.net_name,
    args.bf16_quant_layers_file,
    )

  if os.system(cmd) != 0:
    print('Cmd {} execute failed'.format(cmd))
    exit(-1)

def cal_sqnr(signal_raw, signal_dequant):
  raw = signal_raw.flatten()
  dequant = signal_dequant.flatten()
  noise = raw - dequant

  avg_raw = np.sum(raw) / raw.size
  avg_noise = np.sum(noise) / noise.size

  raw_zero_mean = raw - avg_raw
  noise_zero_mean = noise - avg_noise

  var_raw_zero_mean = np.var(raw_zero_mean)
  var_noise_zero_mean = np.var(noise_zero_mean)
  # print(np.max(signal_raw), np.max(signal_dequant))
  # print(var_raw_zero_mean, var_noise_zero_mean)

  if var_noise_zero_mean == 0.0:
      return 2^31 - 1

  sqnr = 10 * np.log10(var_raw_zero_mean / var_noise_zero_mean)

  return sqnr


# default all set to bf16 than turn off some layer to find it
def sqnr_mean_one_output(args, layers):
  sqnr_list = list()
  pred_fp32 = list()

  # pred_fp32, mean of fp32
  print('Collect pred_fp32...')

  # get layers
  create_bf16_layers(args, layers)
  gen_mlir(args)
  module = pymlir.module()
  module.load(args.model)

  op_info = module.op_info
  for x, _, _ in imagenet_generator(generate_count=g_val_data_count, preprocess=preprocess):
    res = module.run(x)
    y = module.get_all_tensor()
    y = y[module.op_info[-1]['name']].flatten()
    pred_fp32.append(y)

  for info in op_info:
    if info['type'] in skip_ops:
      continue

    layer_name = info['name']
    # pred_int8
    print('Collect pred_int8...', layer_name)
    create_bf16_layers(args, layers, [layer_name])

    gen_mlir(args)
    module = pymlir.module()
    module.load(args.model)

    sqnr = 0
    data_count = 0
    for x, _, _ in imagenet_generator(generate_count=g_val_data_count, preprocess=preprocess):
      res = module.run(x)
      y = module.get_all_tensor()
      # get last output
      y = y[module.op_info[-1]['name']].flatten()
      #y = y.values()[0].flatten()
      sqnr += cal_sqnr(pred_fp32[data_count], y)
      data_count += 1

    sqnr_list.append((layer_name, sqnr / g_val_data_count))

  sqnr_list = sorted(sqnr_list, cmp=lambda x, y: cmp(x[1], y[1]))

  for layer_sqnr in sqnr_list:
    print('{}, {}'.format(layer_sqnr[0], layer_sqnr[1]))

if __name__ == '__main__':
  args = parse_args()
  args.bf16_quant_layers_file = "{}_bf16_layers".format(args.net_name)

  layer_names = get_rows_by_column(args.all_layers_name_csv_file,
      [args.layers_column_name])
  if log_flag:
    print("all layer:", layer_names)

  sqnr_mean_one_output(args, layer_names)
